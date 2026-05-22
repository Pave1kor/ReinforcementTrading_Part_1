import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

# ----------------------------------------------------------------------
def compute_metrics(equity_curve, initial_equity=100000.0):
    full_equity = np.array([initial_equity] + equity_curve)
    returns = np.diff(full_equity) / full_equity[:-1]
    if len(returns) == 0:
        return 0.0, 0.0
    annual_factor = np.sqrt(252 * 39)   # 10‑минутные бары
    sharpe = annual_factor * returns.mean() / (returns.std() + 1e-8)
    peak = np.maximum.accumulate(full_equity)
    drawdown = (peak - full_equity) / peak
    max_dd = np.max(drawdown)
    return sharpe, max_dd


def evaluate_model(model, eval_env, deterministic=True):
    obs = eval_env.reset()
    lstm_states = None
    episode_starts = np.ones((eval_env.num_envs,), dtype=bool)
    equity_curve = []
    try:
        initial_equity = eval_env.get_attr("initial_equity_usd")[0]
    except:
        initial_equity = 100000.0
    equity_curve.append(initial_equity)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)
        episode_starts = dones
        if dones[0]:
            equity_curve.append(float(infos[0]["equity_usd"]))
            break
        equity_curve.append(float(infos[0]["equity_usd"]))
    return equity_curve, equity_curve[-1]


# ----------------------------------------------------------------------
def main():
    # Путь к объединённому CSV (должен содержать все данные 2018-2025)
    DATA_PATH = "data/SBER_2018_2025.csv"
    # Отдельный тестовый файл 2024-2025 (опционально)
    TEST_PATH = "data/SBER_test_2023_daily"

    # 1. Загружаем и рассчитываем признаки на ВСЁМ датасете (один раз)
    df_full, feature_cols = load_and_preprocess_data(DATA_PATH)

    # 2. Определяем внешний тестовый период (2024-2025)
    if os.path.exists(TEST_PATH):
        test_df_full, _ = load_and_preprocess_data(TEST_PATH)
        print("Загружен отдельный тестовый файл 2024-2025")
    else:
        test_start = pd.Timestamp("2024-01-03 09:50:00")
        test_end = pd.Timestamp("2025-12-30 23:40:00")
        test_df_full = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)].copy()
        print("Отдельный тестовый файл не найден, использую данные за 2024-2025 из основного файла")

    # 3. Данные до 2024 года для walk‑forward
    train_val_df = df_full[df_full.index < pd.Timestamp("2024-01-01")].copy()
    print(f"Период walk‑forward: {train_val_df.index.min()} – {train_val_df.index.max()}")

    # 4. Формируем окна walk‑forward (train / val / test) без утечки
    #    Теперь train_val_df уже содержит все признаки,
    #    поэтому мы просто выбираем непрерывные интервалы.
    years = sorted(set(train_val_df.index.year))
    years = [y for y in years if y >= 2018 and y <= 2023]   # убираем 2024

    windows = []
    # Для каждого среднего года создаём окно: train = всё до val_year, val = val_year, test = следующий год
    for i in range(1, len(years)-1):
        val_year = years[i]
        test_year = years[i+1]

        train_end = pd.Timestamp(f"{val_year}-01-01")               # граница train/val
        val_start = pd.Timestamp(f"{val_year}-01-01")
        val_end   = pd.Timestamp(f"{test_year}-01-01")
        test_start_w = val_end
        test_end_w   = pd.Timestamp(f"{test_year+1}-01-01") if test_year+1 <= 2023 else train_val_df.index.max()

        train_df = train_val_df[train_val_df.index < train_end].copy()
        val_df   = train_val_df[(train_val_df.index >= val_start) & (train_val_df.index < val_end)].copy()
        test_inner_df = train_val_df[(train_val_df.index >= test_start_w) & (train_val_df.index < test_end_w)].copy()

        if len(train_df) > 500 and len(val_df) > 200 and len(test_inner_df) > 200:
            windows.append({
                'name': f"train_until_{val_year-1}_val_{val_year}_test_{test_year}",
                'train_df': train_df,
                'val_df': val_df,
                'test_inner_df': test_inner_df,
            })

    if not windows:
        print("Не удалось создать ни одного окна. Проверьте диапазон дат.")
        return

    # ------------------------------------------------------------------
    # Параметры обучения
    WIN = 60
    NUM_ENVS = 4
    TOTAL_TIMESTEPS = 500_000          # на одно окно (можно увеличить)

    all_results = []

    # ------------------------------------------------------------------
    # 5. Обучение по каждому окну
    for win in windows:
        print(f"\n===== Окно: {win['name']} =====")
        train_df = win['train_df']
        val_df   = win['val_df']
        test_inner_df = win['test_inner_df']

        # Нормализация на тренировочных данных
        train_features = train_df[feature_cols].values.astype(np.float32)
        train_mean = np.mean(train_features, axis=0)
        train_std  = np.std(train_features, axis=0)

        # Фабрики сред
        def make_train_env():
            return ForexTradingEnv(
                df=train_df, window_size=WIN, feature_columns=feature_cols,
                spread_pips=1.0, commission_pips=0.0, max_slippage_pips=1.0,
                random_start=True, min_episode_steps=300, episode_max_steps=800,
                feature_mean=train_mean, feature_std=train_std,
                risk_per_trade=0.005, base_sl_pips=40.0, base_tp_pips=80.0,
                k_sl=0.3, k_tp=0.6, open_penalty_pips=0.0, time_penalty_pips=0.0005,
                trailing_atr_mult=2.0, min_atr_pips=5.0, slope_div_reward_scale=0.002,
                open_bonus_pips=5.0, reward_scale=0.01, pip_value=0.01, lot_size=1.0, leverage=1.0
            )

        def make_eval_env(df, mean, std):
            return ForexTradingEnv(
                df=df, window_size=WIN, feature_columns=feature_cols,
                spread_pips=1.0, commission_pips=0.0, max_slippage_pips=1.0,
                random_start=False, episode_max_steps=None,
                feature_mean=mean, feature_std=std,
                risk_per_trade=0.005, base_sl_pips=40.0, base_tp_pips=80.0,
                k_sl=0.3, k_tp=0.6, open_penalty_pips=0.0, time_penalty_pips=0.0005,
                trailing_atr_mult=2.0, min_atr_pips=5.0, slope_div_reward_scale=0.002,
                open_bonus_pips=5.0, reward_scale=0.01, pip_value=0.01, lot_size=1.0, leverage=1.0
            )

        # Обучающая среда с нормализацией
        train_vec_env = SubprocVecEnv([lambda i=i: make_train_env() for i in range(NUM_ENVS)])
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True,
                                     clip_obs=10.0, gamma=0.99)

        # Валидационная среда (без тренировочной нормализации reward)
        val_env = DummyVecEnv([lambda: make_eval_env(val_df, train_mean, train_std)])
        val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, training=False)

        # Политика
        policy_kwargs = dict(
            net_arch=dict(shared=[128], pi=[64], vf=[32]),
            lstm_hidden_size=48, n_lstm_layers=1, activation_fn=nn.Tanh
        )

        model = RecurrentPPO(
            policy="MlpLstmPolicy", env=train_vec_env, verbose=1,
            tensorboard_log=f"./tensorboard_log/{win['name']}", policy_kwargs=policy_kwargs,
            ent_coef=0.05, learning_rate=1e-4, n_steps=4096, batch_size=256, n_epochs=8,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.3, max_grad_norm=0.7
        )

        os.makedirs(f"./checkpoints/{win['name']}", exist_ok=True)
        checkpoint_callback = CheckpointCallback(save_freq=50_000,
                                                  save_path=f"./checkpoints/{win['name']}",
                                                  name_prefix="RecurrentPPO_sber")
        eval_callback = EvalCallback(val_env, best_model_save_path=f"./best_model_{win['name']}",
                                     log_path=f"./eval_log_{win['name']}", eval_freq=10000,
                                     deterministic=True, render=False)

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback])
        model.save(f"model_sber_{win['name']}")
        train_vec_env.save(f"vec_normalize_{win['name']}.pkl")

        # Оценка на внутреннем тесте (один эпизод)
        test_env = DummyVecEnv([lambda: make_eval_env(test_inner_df, train_mean, train_std)])
        test_env = VecNormalize.load(f"vec_normalize_{win['name']}.pkl", test_env)
        test_env.training = False
        test_env.norm_reward = False

        test_curve, test_final = evaluate_model(model, test_env)
        test_sharpe, test_mdd = compute_metrics(test_curve)
        print(f"Inner test {win['name']}: final equity = {test_final:.2f}, "
              f"Sharpe = {test_sharpe:.3f}, MaxDD = {test_mdd*100:.2f}%")
        all_results.append({'window': win['name'], 'sharpe': test_sharpe,
                            'mdd': test_mdd, 'final': test_final})

    # ------------------------------------------------------------------
    # 6. Финальное обучение на ВСЕХ данных до 2023 года
    print("\n===== Финальное обучение на 2018-2023 =====")
    final_train_df = train_val_df
    final_train_features = final_train_df[feature_cols].values.astype(np.float32)
    final_mean = np.mean(final_train_features, axis=0)
    final_std  = np.std(final_train_features, axis=0)

    def make_final_train_env():
        return ForexTradingEnv(
            df=final_train_df, window_size=WIN, feature_columns=feature_cols,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=1.0,
            random_start=True, min_episode_steps=300, episode_max_steps=800,
            feature_mean=final_mean, feature_std=final_std,
            risk_per_trade=0.005, base_sl_pips=40.0, base_tp_pips=80.0,
            k_sl=0.3, k_tp=0.6, open_penalty_pips=0.0, time_penalty_pips=0.0005,
            trailing_atr_mult=2.0, min_atr_pips=5.0, slope_div_reward_scale=0.002,
            open_bonus_pips=5.0, reward_scale=0.01, pip_value=0.01, lot_size=1.0, leverage=1.0
        )

    final_train_vec_env = SubprocVecEnv([lambda i=i: make_final_train_env() for i in range(NUM_ENVS)])
    final_train_vec_env = VecNormalize(final_train_vec_env, norm_obs=True, norm_reward=True,
                                       clip_obs=10.0, gamma=0.99)

    final_model = RecurrentPPO(
        policy="MlpLstmPolicy", env=final_train_vec_env, verbose=1,
        tensorboard_log="./tensorboard_log/final", policy_kwargs=policy_kwargs,
        ent_coef=0.05, learning_rate=1e-4, n_steps=4096, batch_size=256, n_epochs=8,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.3, max_grad_norm=0.7
    )
    final_model.learn(total_timesteps=1_000_000)
    final_model.save("model_sber_final")
    final_train_vec_env.save("vec_normalize_final.pkl")

    # Сохраняем метаданные финальной модели
    with open("final_metadata.json", "w") as f:
        json.dump({
            'mean': final_mean.tolist(),
            'std': final_std.tolist(),
            'feature_cols': feature_cols,
            'train_start': str(final_train_df.index.min()),
            'train_end': str(final_train_df.index.max())
        }, f)

    # ------------------------------------------------------------------
    # 7. Финальный тест на 2024-2025
    def make_final_test_env(df, mean, std):
        return ForexTradingEnv(
            df=df, window_size=WIN, feature_columns=feature_cols,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=1.0,
            random_start=False, episode_max_steps=None,
            feature_mean=mean, feature_std=std,
            risk_per_trade=0.005, base_sl_pips=40.0, base_tp_pips=80.0,
            k_sl=0.3, k_tp=0.6, open_penalty_pips=0.0, time_penalty_pips=0.0005,
            trailing_atr_mult=2.0, min_atr_pips=5.0, slope_div_reward_scale=0.002,
            open_bonus_pips=5.0, reward_scale=0.01, pip_value=0.01, lot_size=1.0, leverage=1.0
        )

    test_final_env = DummyVecEnv([lambda: make_final_test_env(test_df_full, final_mean, final_std)])
    test_final_env = VecNormalize.load("vec_normalize_final.pkl", test_final_env)
    test_final_env.training = False
    test_final_env.norm_reward = False

    final_test_curve, final_test_equity = evaluate_model(final_model, test_final_env)
    final_sharpe, final_mdd = compute_metrics(final_test_curve)

    print(f"\n========== ФИНАЛЬНЫЙ ТЕСТ 2024-2025 ==========")
    print(f"Итоговая эквити: {final_test_equity:.2f} RUB")
    print(f"Sharpe (годовой): {final_sharpe:.3f}, MaxDD: {final_mdd*100:.2f}%")

    # Сохраняем результаты
    with open("walkforward_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    plt.figure(figsize=(12,6))
    plt.plot(final_test_curve, label="Equity 2024-2025")
    plt.title("Final Test Equity Curve 2024-2025")
    plt.xlabel("10-min bars")
    plt.ylabel("Equity (RUB)")
    plt.grid(True)
    plt.legend()
    plt.savefig("final_test_curve.png")
    plt.close()

    print("Аутсемплинг завершён. Модель сохранена как model_sber_final")


if __name__ == "__main__":
    main()