import os
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

def compute_metrics(equity_curve, initial_equity=10000.0):
    full_equity = np.array([initial_equity] + equity_curve)
    returns = np.diff(full_equity) / full_equity[:-1]
    if len(returns) == 0:
        return 0.0, 0.0
    sharpe = np.sqrt(252 * 24) * returns.mean() / (returns.std() + 1e-8)
    peak = np.maximum.accumulate(full_equity)
    drawdown = (peak - full_equity) / peak
    max_dd = np.max(drawdown)
    return sharpe, max_dd

def evaluate_model(model, eval_env, deterministic=True):
    """
    Оценка модели на одном эпизоде с корректным сбросом LSTM состояний.
    """
    obs = eval_env.reset()
    lstm_states = None
    episode_starts = np.ones((eval_env.num_envs,), dtype=bool)
    equity_curve = []
    # Получаем начальную эквити из среды
    initial_equity = eval_env.get_attr("initial_equity_usd")[0]  # для DummyVecEnv
    # Если используется VecNormalize, нужно получить через envs[0]
    # Упростим: будем брать из info после первого шага? Но проще добавить метод в env.
    # Альтернатива: передавать initial_equity параметром. Для простоты пока захардкодим 10000.
    # Но лучше получить динамически:
    try:
        initial_equity = eval_env.get_attr("initial_equity_usd")[0]
    except:
        initial_equity = 10000.0
    equity_curve.append(initial_equity)

    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)
        # Обновляем episode_starts и сбрасываем LSTM для завершившихся окружений
        episode_starts = dones
        # Если окружение завершилось (в DummyVecEnv только одно), выходим
        if dones[0]:
            # Добавляем финальную эквити
            equity_curve.append(float(infos[0]["equity_usd"]))
            break
        equity_curve.append(float(infos[0]["equity_usd"]))
    return equity_curve, equity_curve[-1]

def main():
    file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    split_idx = int(len(df) * 0.8)
    train_val_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    val_size = int(0.1 * len(train_val_df))
    val_df = train_val_df.iloc[-val_size:].copy()
    train_df = train_val_df.iloc[:-val_size].copy()

    train_features = train_df[feature_cols].values.astype(np.float32)
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    WIN = 60
    NUM_ENVS = 4

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
            risk_per_trade=0.002,      # изменено
            base_sl_pips=30.0,         # изменено
            base_tp_pips=60.0,         # изменено
            k_sl=0.3,
            k_tp=0.6,
            open_penalty_pips=0.0,     # изменено
            time_penalty_pips=0.0005,  # изменено
            trailing_atr_mult=2.0,
            min_atr_pips=10.0,
            slope_div_reward_scale=0.01,
            open_bonus_pips=5.0,
            reward_scale=0.001,
        )

    def make_eval_env(df):
        return ForexTradingEnv(
            df=df,
            window_size=WIN,
            feature_columns=feature_cols,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=1.0,
            random_start=False,
            episode_max_steps=None,
            feature_mean=train_mean,
            feature_std=train_std,
            risk_per_trade=0.002,
            base_sl_pips=30.0,
            base_tp_pips=60.0,
            k_sl=0.3,
            k_tp=0.6,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0005,
            trailing_atr_mult=2.0,
            min_atr_pips=10.0,
            slope_div_reward_scale=0.01,
            open_bonus_pips=5.0,
            reward_scale=0.001,
        )

    # Обучающая среда с векторизацией и нормализацией
    train_vec_env = SubprocVecEnv([lambda i=i: make_train_env() for i in range(NUM_ENVS)])
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Функция для создания eval-среды с загрузкой статистики нормализации
    def make_normalized_eval_env(df, norm_path="vec_normalize.pkl"):
        env = DummyVecEnv([lambda: make_eval_env(df)])
        # Загружаем статистику, сохранённую после обучения
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False
        else:
            # Если файла нет (первый запуск), создаём пустую нормализацию
            env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
        return env

    # Создаём eval-среды (нормализация будет загружена после обучения)
    val_env = make_normalized_eval_env(val_df)
    test_eval_env = make_normalized_eval_env(test_df)
    train_eval_env = make_normalized_eval_env(train_df)

    policy_kwargs = dict(
        net_arch=dict(shared=[128], pi=[64], vf=[64]),
        lstm_hidden_size=48,
        n_lstm_layers=1
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.01,          # увеличено с 0.001
        learning_rate=3e-5,     # уменьшено с 1e-4
        n_steps=2048,           # уменьшено с 8192
        batch_size=128,         # уменьшено с 256
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,         # уменьшено с 0.3
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="RecurrentPPO_eurusd")
    eval_callback = EvalCallback(val_env, best_model_save_path="./best_model",
                                 log_path="./eval_log", eval_freq=10000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=2_000_000, callback=[checkpoint_callback, eval_callback])
    model.save("model_eurusd_best")
    train_vec_env.save("vec_normalize.pkl")   # сохраняем статистику нормализации

    # После сохранения можно пересоздать eval-среды с загрузкой (но они уже созданы без загрузки – пересоздадим)
    val_env = make_normalized_eval_env(val_df, "vec_normalize.pkl")
    test_eval_env = make_normalized_eval_env(test_df, "vec_normalize.pkl")
    train_eval_env = make_normalized_eval_env(train_df, "vec_normalize.pkl")

    # Оценка на train, val, test
    train_curve, train_final = evaluate_model(model, train_eval_env)
    test_curve, test_final = evaluate_model(model, test_eval_env)

    train_sharpe, train_mdd = compute_metrics(train_curve)
    test_sharpe, test_mdd = compute_metrics(test_curve)

    print(f"[Train] Final equity: {train_final:.2f}, Sharpe: {train_sharpe:.3f}, MaxDD: {train_mdd*100:.2f}%")
    print(f"[Test]  Final equity: {test_final:.2f}, Sharpe: {test_sharpe:.3f}, MaxDD: {test_mdd*100:.2f}%")

    plt.figure(figsize=(12,6))
    plt.plot(train_curve, label=f"Train equity: {train_final:.2f}$")
    plt.title("Train Equity Curve")
    plt.legend()
    plt.grid()
    plt.savefig("train_curve.png")

    plt.figure(figsize=(12,6))
    plt.plot(test_curve, label=f"Test equity: {test_final:.2f}$", color="orange")
    plt.title("Test Equity Curve")
    plt.legend()
    plt.grid()
    plt.savefig("test_curve.png")
    plt.show()

if __name__ == "__main__":
    main()