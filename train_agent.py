import os
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

def compute_metrics(equity_curve, initial_equity=10000.0):
    """Расчёт Sharpe и MaxDD с учётом начальной эквити."""
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
    obs = eval_env.reset()
    lstm_states = None
    episode_starts = np.ones((eval_env.num_envs,), dtype=bool)
    equity_curve = []
    # ИЗМЕНЕНО: запоминаем начальную эквити из среды
    initial_equity = eval_env.get_attr("initial_equity_usd")[0]
    equity_curve.append(initial_equity)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)
        done = bool(dones[0])
        episode_starts = dones
        info = infos[0]
        equity_curve.append(float(info["equity_usd"]))
        if done:
            break
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
            risk_per_trade=0.007,        # ИЗМЕНЕНО: увеличено
            base_sl_pips=35.0,           # ИЗМЕНЕНО
            base_tp_pips=70.0,           # ИЗМЕНЕНО
            k_sl=0.3,                    # ИЗМЕНЕНО
            k_tp=0.6,                    # ИЗМЕНЕНО
            open_penalty_pips=0.5,       # ИЗМЕНЕНО
            time_penalty_pips=0.001,     # ИЗМЕНЕНО
            trailing_trigger_pips=30.0,  # ИЗМЕНЕНО: добавлено
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
            risk_per_trade=0.007,
            base_sl_pips=35.0,
            base_tp_pips=70.0,
            k_sl=0.3,
            k_tp=0.6,
            open_penalty_pips=0.5,
            time_penalty_pips=0.001,
            trailing_trigger_pips=30.0,
        )

    train_vec_env = SubprocVecEnv([lambda i=i: make_train_env() for i in range(NUM_ENVS)])
    def make_monitored_eval_env(df):
        env = make_eval_env(df)
        env = Monitor(env)
        return env
    train_eval_env = DummyVecEnv([lambda: make_monitored_eval_env(train_df)])
    val_env = SubprocVecEnv([lambda: make_monitored_eval_env(val_df)])
    test_eval_env = DummyVecEnv([lambda: make_eval_env(test_df)])

    policy_kwargs = dict(
        net_arch=dict(shared=[128], pi=[64], vf=[64]),  # ИЗМЕНЕНО: увеличены размеры
        lstm_hidden_size=48,
        n_lstm_layers=1
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.01,                 # ИЗМЕНЕНО: увеличено
        learning_rate=3e-5,            # ИЗМЕНЕНО: увеличено
        n_steps=4096,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=1.0,                   # ИЗМЕНЕНО: увеличено для лучшей оценки value
        max_grad_norm=0.5,             # ИЗМЕНЕНО: добавлено
    )

    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="RecurrentPPO_eurusd")
    eval_callback = EvalCallback(val_env, best_model_save_path="./best_model",
                                 log_path="./eval_log", eval_freq=10000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=500_000, callback=[checkpoint_callback, eval_callback])  # увеличено до 2M
    model.save("model_eurusd_best")

    # Оценка
    train_curve, train_final = evaluate_model(model, train_eval_env)
    test_curve, test_final = evaluate_model(model, test_eval_env)

    train_sharpe, train_mdd = compute_metrics(train_curve)
    test_sharpe, test_mdd = compute_metrics(test_curve)

    print(f"[Train] Final equity: {train_final:.2f}, Sharpe: {train_sharpe:.3f}, MaxDD: {train_mdd*100:.2f}%")
    print(f"[Test]  Final equity: {test_final:.2f}, Sharpe: {test_sharpe:.3f}, MaxDD: {test_mdd*100:.2f}%")

    plt.figure(figsize=(12,6))
    plt.plot(train_curve, label=f"Train equity: {train_final:.2f}$")
    plt.title("Train Equity Curve")
    plt.legend(); plt.grid(); plt.savefig("train_curve.png")

    plt.figure(figsize=(12,6))
    plt.plot(test_curve, label=f"Test equity: {test_final:.2f}$", color="orange")
    plt.title("Test Equity Curve")
    plt.legend(); plt.grid(); plt.savefig("test_curve.png")

if __name__ == "__main__":
    main()