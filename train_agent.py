# train.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def evaluate_model_test(model, vec_env, deterministic: bool = True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    while True:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic
        )
      
        obs, rewards, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        episode_starts = dones
        info = infos[0]

        if not done:
            # Обычный шаг: пишем эквити и проверяем закрытые сделки
            if isinstance(info, dict) and "equity_usd" in info:
                equity_curve.append(float(info["equity_usd"]))
            
            if isinstance(info, dict) and info.get("last_trade_info") is not None:
                trade_info = info["last_trade_info"]
                if trade_info.get("event") == "CLOSE":
                    if not closed_trades or closed_trades[-1]["step"] != trade_info["step"]:
                        closed_trades.append(trade_info)
        else:
            # Терминальный шаг: среда АВТО-СБРОСИЛАСЬ. 
            # Достаем реальные финальные данные из пре-терминального состояния (terminal_info)
            if "terminal_info" in info and isinstance(info["terminal_info"], dict):
                term_info = info["terminal_info"]
            else:
                # В зависимости от версии SB3, данные могут лежать в корне info
                term_info = info

            # 1. Забираем точное финальное эквити
            if "equity_usd" in term_info:
                equity_curve.append(float(term_info["equity_usd"]))
            elif "terminal_observation" in info and isinstance(info["terminal_observation"], dict) and "equity_usd" in info["terminal_observation"]:
                equity_curve.append(float(info["terminal_observation"]["equity_usd"]))
            elif equity_curve:
                equity_curve.append(equity_curve[-1])
            
            # 2. Критически важно: забираем финальную сделку (например, закрытую по END_OF_DATA)
            if "last_trade_info" in term_info and term_info["last_trade_info"] is not None:
                trade_info = term_info["last_trade_info"]
                if trade_info.get("event") == "CLOSE":
                    if not closed_trades or closed_trades[-1]["step"] != trade_info["step"]:
                        closed_trades.append(trade_info)
    break
 return equity_curve, closed_trades



def main():
    file_path = "data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("Training bars:", len(train_df))
    print("Testing bars :", len(test_df))

    train_features_df = train_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    train_features = train_features_df.values.astype(np.float32)
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    SL_OPTS = [1.0, 1.5]
    TP_OPTS = [2.0, 3.5]
    WIN = 60
    NUM_ENVS = 4  
    
    def make_train_env():
        return ForexTradingEnv(
            df=train_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=0.5,
            random_start=True, min_episode_steps=300, episode_max_steps=800,
            feature_mean=train_mean, feature_std=train_std, feature_columns=feature_cols,
            open_penalty_pips=0.1, time_penalty_pips=0.005,
        )

    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=0.5,
            random_start=False, episode_max_steps=None,
            feature_mean=train_mean, feature_std=train_std, feature_columns=feature_cols,
            open_penalty_pips=0.1, time_penalty_pips=0.005,
        )

    def make_test_eval_env():
        return ForexTradingEnv(
            df=test_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=0.5,
            random_start=False, episode_max_steps=None,
            feature_std=train_std, feature_mean=train_mean, feature_columns=feature_cols,
            open_penalty_pips=0.1, time_penalty_pips=0.005,
        )

    # ФИКС: Убран лишний аргумент `i=i` из лямбды, так как фабрика не принимает аргументов
    train_vec_env = SubprocVecEnv([lambda: make_train_env() for _ in range(NUM_ENVS)])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    policy_kwargs = dict(
        net_arch=dict(shared=[64], pi=[32], vf=[32]),
        lstm_hidden_size=48,
        n_lstm_layers=1
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy", env=train_vec_env, verbose=1,
        tensorboard_log="./tensorboard_log/", policy_kwargs=policy_kwargs,
        ent_coef=0.02, learning_rate=1e-4, n_steps=1024, batch_size=256,
        n_epochs=4, gamma=0.99, gae_lambda=0.95, clip_range=0.1, vf_coef=0.5
    )

    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=ckpt_dir, name_prefix="RecurrentPPO_eurusd"
    )

    model.learn(total_timesteps=500000, callback=checkpoint_callback)
    model.save("model_eurusd_best")
    print("Model saved: model_eurusd_best")

    plot_model_train = RecurrentPPO.load("model_eurusd_best", env=train_eval_env)
    equity_curve_train, final_equity_train = evaluate_model(plot_model_train, train_eval_env)
    
    plot_model_test = RecurrentPPO.load("model_eurusd_best", env=test_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(plot_model_test, test_eval_env)

    print(f"[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    # ФИКС: Обязательный plt.close() во избежание утечки памяти RAM
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label=f"Train (in-sample) equity: {final_equity_train:.2f}$")
    plt.title("Train (In-Sample) Equity Curve - Best Model")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model1_train_clean.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_test, label=f"Test (out-of-sample) equity: {final_equity_test:.2f}$", color="orange")
    plt.title("Test (Out-of-Sample) Equity Curve - Best Model")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model1_test_clean.png")
    plt.close()


if __name__ == "__main__":
    main()