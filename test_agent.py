import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from train_agent import evaluate_model, calculate_metrics   # переиспользуем функции

def main():
    file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_features = train_df[feature_cols].values.astype(np.float32)
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    SL_OPTS = [1.0, 1.5]
    TP_OPTS = [2.0, 3.5]
    WIN = 60

    def make_test_env():
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.5,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            feature_mean=train_mean,
            feature_std=train_std,
            open_penalty_pips=0.1,
            time_penalty_pips=0.005,
            unrealized_reward_coef=0.1
        )

    test_env = DummyVecEnv([make_test_env])
    model = RecurrentPPO.load("model_eurusd_best", env=test_env)

    equity_curve, closed_trades = evaluate_model(model, test_env, deterministic=True)

    metrics = calculate_metrics(equity_curve, closed_trades)
    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    plt.figure(figsize=(10,6))
    plt.plot(equity_curve, label=f"Final Equity: {equity_curve[-1]:.2f}")
    plt.title("Test Equity Curve")
    plt.xlabel("Steps")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("my_plot.png")
    plt.show()

    if closed_trades:
        pd.DataFrame(closed_trades).to_csv("trade_history_output.csv", index=False)
        print("Trades saved.")

if __name__ == "__main__":
    main()