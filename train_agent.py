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

def compute_metrics(equity_curve, initial_equity=100000.0):
    full_equity = np.array([initial_equity] + equity_curve)
    returns = np.diff(full_equity) / full_equity[:-1]
    if len(returns) == 0:
        return 0.0, 0.0
    # 10-минутные бары: ~39 баров в день, 252 торговых дня
    annual_factor = np.sqrt(252 * 39)
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
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)
        episode_starts = dones
        if dones[0]:
            equity_curve.append(float(infos[0]["equity_usd"]))
            break
        equity_curve.append(float(infos[0]["equity_usd"]))
    return equity_curve, equity_curve[-1]

def main():
    DATA_PATH = "data/SBER_10min.csv"
    df_full, feature_cols = load_and_preprocess_data(DATA_PATH)

    # Разбиение по годам для walk‑forward
    # Определим список лет, которые будут использоваться как валидационные и тестовые
    years = sorted(set(df_full.index.year))
    # Убираем слишком маленькие года
    years = [y for y in years if y >= 2018 and y <= 2025]
    # Для walk‑forward: тренируемся на данных до года val_year, валидация на val_year, тест на val_year+1
    # Создаём окна: train до 2021, val 2022, test 2023; затем train до 2022, val 2023, test 2024; и т.д.
    windows = []
    for i in range(len(years)-2):
        val_year = years[i+1]
        test_year = years[i+2]
        train_end_date = pd.Timestamp(f"{val_year}-01-01")
        val_start_date = pd.Timestamp(f"{val_year}-01-01")
        val_end_date = pd.Timestamp(f"{test_year}-01-01")
        test_start_date = val_end_date
        test_end_date = pd.Timestamp(f"{test_year+1}-01-01") if test_year+1 <= max(years) else df_full.index.max()

        train_df = df_full[df_full.index < train_end_date].copy()
        val_df = df_full[(df_full.index >= val_start_date) & (df_full.index < val_end_date)].copy()
        test_df = df_full[(df_full.index >= test_start_date) & (df_full.index < test_end_date)].copy()

        if len(train_df) > 500 and len(val_df) > 200 and len(test_df) > 200:
            windows.append({
                'name': f"train_until_{val_year-1}_val_{val_year}_test_{test_year}",
                'train_df': train_df,
                'val_df': val_df,
                'test_df': test_df,
                'train_mean': None,
                'train_std': None,
                'test_start': test_start_date,
                'test_end': test_end_date
            })

    # Если windows пуст – создаём одно окно по заданным диапазонам (2018-2023 train, 2024-2025 test)
    if not windows:
        train_df = df_full[(df_full.index >= "2018-01-01") & (df_full.index < "2024-01-01")].copy()
        test_df = df_full[(df_full.index >= "2024-01-01") & (df_full.index <= "2025-12-31")].copy()
        windows = [{
            'name': 'train_2018_2023_test_2024_2025',
            'train_df': train_df,
            'val_df': test_df.sample(frac=0.2) if len(test_df) > 200 else test_df,  # fallback val
            'test_df': test_df,
            'test_start': "2024-01-01",
            'test_end': "2025-12-31"
        }]

    # Параметры обучения
    WIN = 60
    NUM_ENVS = 4
    TOTAL_TIMESTEPS = 1_000_000  # можно увеличить до 2M

    for win in windows:
        print(f"\n===== Обработка окна: {win['name']} =====")
        train_df = win['train_df']
        val_df = win['val_df']
        test_df = win['test_df']

        # Нормализация на train
        train_features = train_df[feature_cols].values.astype(np.float32)
        train_mean = np.mean(train_features, axis=0)
        train_std = np.std(train_features, axis=0)
        win['train_mean'] = train_mean
        win['train_std'] = train_std

        def make_train_env():
            return ForexTradingEnv(
                df=train_df,
                window_size=WIN,
                feature_columns=feature_cols,
                spread_pips=1.0,
                commission_pips=0.0,
                max_slippage_pips=1.0,
                random_start=True,
                min_episode_steps=300,
                episode_max_steps=800,
                feature_mean=train_mean,
                feature_std=train_std,
                risk_per_trade=0.005,
                base_sl_pips=40.0,
                base_tp_pips=80.0,
                k_sl=0.3,
                k_tp=0.6,
                open_penalty_pips=0.0,
                time_penalty_pips=0.0005,
                trailing_atr_mult=2.0,
                min_atr_pips=5.0,
                slope_div_reward_scale=0.002,
                open_bonus_pips=5.0,
                reward_scale=0.002,
                pip_value=0.01,
                lot_size=1.0,
                leverage=1.0
            )

        def make_eval_env(df, mean, std):
            return ForexTradingEnv(
                df=df,
                window_size=WIN,
                feature_columns=feature_cols,
                spread_pips=1.0,
                commission_pips=0.0,
                max_slippage_pips=1.0,
                random_start=False,
                episode_max_steps=None,
                feature_mean=mean,
                feature_std=std,
                risk_per_trade=0.005,
                base_sl_pips=40.0,
                base_tp_pips=80.0,
                k_sl=0.3,
                k_tp=0.6,
                open_penalty_pips=0.0,
                time_penalty_pips=0.0005,
                trailing_atr_mult=2.0,
                min_atr_pips=5.0,
                slope_div_reward_scale=0.002,
                open_bonus_pips=5.0,
                reward_scale=0.002,
                pip_value=0.01,
                lot_size=1.0,
                leverage=1.0
            )

        # Обучающая среда с нормализацией
        train_vec_env = SubprocVecEnv([lambda i=i: make_train_env() for i in range(NUM_ENVS)])
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

        def make_normalized_eval_env(df, mean, std, norm_path):
            env = DummyVecEnv([lambda: make_eval_env(df, mean, std)])
            if os.path.exists(norm_path):
                env = VecNormalize.load(norm_path, env)
                env.training = False
                env.norm_reward = False
            else:
                env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
            return env

        # Валидационная среда (пока без загрузки нормализации, она будет загружена после обучения)
        val_env = make_normalized_eval_env(val_df, train_mean, train_std, "dummy.pkl")

        policy_kwargs = dict(
            net_arch=dict(shared=[128], pi=[64], vf=[32]),
            lstm_hidden_size=48,
            n_lstm_layers=1,
            activation_fn=nn.Tanh,
        )

        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_vec_env,
            verbose=1,
            tensorboard_log=f"./tensorboard_log/{win['name']}",
            policy_kwargs=policy_kwargs,
            ent_coef=0.01,
            learning_rate=3e-5,
            n_steps=2048,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.3,
            max_grad_norm=0.7,
        )

        os.makedirs(f"./checkpoints/{win['name']}", exist_ok=True)
        checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./checkpoints/{win['name']}", name_prefix="RecurrentPPO_sber")
        eval_callback = EvalCallback(val_env, best_model_save_path=f"./best_model_{win['name']}",
                                     log_path=f"./eval_log_{win['name']}", eval_freq=10000,
                                     deterministic=True, render=False)

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback])
        model.save(f"model_sber_{win['name']}")
        train_vec_env.save(f"vec_normalize_{win['name']}.pkl")

        # Оценка на тестовом периоде
        test_env = make_normalized_eval_env(test_df, train_mean, train_std, f"vec_normalize_{win['name']}.pkl")
        test_curve, test_final = evaluate_model(model, test_env)
        test_sharpe, test_mdd = compute_metrics(test_curve, initial_equity=100000.0)

        print(f"[{win['name']}] Test final equity: {test_final:.2f} RUB, Sharpe: {test_sharpe:.3f}, MaxDD: {test_mdd*100:.2f}%")

        # Сохраняем метаданные окна
        metadata = {
            'name': win['name'],
            'train_mean': train_mean.tolist(),
            'train_std': train_std.tolist(),
            'feature_cols': feature_cols,
            'test_start': str(win['test_start']),
            'test_end': str(win['test_end']),
            'test_sharpe': test_sharpe,
            'test_max_dd': test_mdd,
            'test_final_equity': test_final
        }
        with open(f"metadata_{win['name']}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # График тестовой кривой
        plt.figure()
        plt.plot(test_curve, label=f"Test equity – {win['name']}")
        plt.title(f"Test Equity Curve – {win['name']}")
        plt.legend()
        plt.savefig(f"test_curve_{win['name']}.png")
        plt.close()

    print("\n===== АУТСЕМПЛИНГ ЗАВЕРШЁН =====")

if __name__ == "__main__":
    main()