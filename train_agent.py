import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

# Вспомогательные функции для расчёта метрик
def calculate_metrics(equity_curve: list, trades: list, initial_capital: float = 10000.0):
    """Рассчитывает основные торговые метрики."""
    equity = np.array(equity_curve)
    if len(equity) < 2:
        return {}
    returns = np.diff(equity) / equity[:-1]
    sharpe = np.sqrt(252 * 24) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
    neg_returns = returns[returns < 0]
    sortino = np.sqrt(252 * 24) * np.mean(returns) / np.std(neg_returns) if len(neg_returns) > 0 and np.std(neg_returns) > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown)
    total_return = (equity[-1] - initial_capital) / initial_capital
    annual_return = (1 + total_return) ** (252 * 24 / len(equity)) - 1 if len(equity) > 0 else 0
    calmar = annual_return / max_dd if max_dd > 0 else np.inf
    if trades:
        trades_df = pd.DataFrame(trades)
        net_pips = trades_df["net_pips"].values
        profit_factor = trades_df[trades_df["net_pips"] > 0]["net_pips"].sum() / abs(trades_df[trades_df["net_pips"] < 0]["net_pips"].sum()) if (trades_df["net_pips"] < 0).sum() > 0 else np.inf
        win_rate = (trades_df["net_pips"] > 0).mean()
        avg_trade = np.mean(net_pips)
        total_trades = len(trades_df)
    else:
        profit_factor = 0.0
        win_rate = 0.0
        avg_trade = 0.0
        total_trades = 0
    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd,
        "calmar_ratio": calmar,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "avg_trade_pips": avg_trade,
        "total_trades": total_trades,
        "final_equity": equity[-1]
    }

def evaluate_model(model, eval_env, deterministic=True):
    """Запускает один эпизод и возвращает кривую эквити и список сделок."""
    obs = eval_env.reset()
    equity_curve = []
    closed_trades = []
    lstm_states = None
    episode_starts = np.ones((eval_env.num_envs,), dtype=bool)

    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
        obs, rewards, dones, infos = eval_env.step(action)
        done = bool(dones[0])
        episode_starts = dones
        info = infos[0]

        if done:
            if isinstance(info, dict) and "equity_usd" in info:
                equity_curve.append(float(info["equity_usd"]))
            elif equity_curve:
                equity_curve.append(equity_curve[-1])
            if isinstance(info, dict) and "last_trade_info" in info:
                trade_info = info["last_trade_info"]
                if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
                    closed_trades.append(trade_info)
            break
        else:
            equity_curve.append(eval_env.get_attr("equity_usd")[0])
            trade_info = eval_env.get_attr("last_trade_info")[0]
            if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
                if not closed_trades or closed_trades[-1] != trade_info:
                    closed_trades.append(trade_info)
    return equity_curve, closed_trades

def main():
    # Загрузка и предобработка данных
    file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # Разбиение (простое 80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Нормализация на train
    train_features = train_df[feature_cols].values.astype(np.float32)
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    # Параметры среды
    SL_OPTS = [1.0, 1.5]
    TP_OPTS = [2.0, 3.5]
    WIN = 60
    NUM_ENVS = 4

    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.5,
            random_start=True,
            min_episode_steps=300,
            episode_max_steps=800,
            feature_mean=train_mean,
            feature_std=train_std,
            feature_columns=feature_cols,
            open_penalty_pips=2.0,
            time_penalty_pips=0.1,
            unrealized_reward_coef=0.05,
            max_drawdown_pct=0.25,
            risk_adjusted_scale=1.0,
            trade_penalty_pips=2.0,
            reward_scale=0.01
        )

    def make_eval_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.5,
            random_start=False,
            episode_max_steps=None,
            feature_mean=train_mean,
            feature_std=train_std,
            feature_columns=feature_cols,
            open_penalty_pips=2.0,
            time_penalty_pips=0.1,
            unrealized_reward_coef=0.05,
            max_drawdown_pct=0.25,
            risk_adjusted_scale=1.0,
            trade_penalty_pips=2.0,
            reward_scale=0.01
        )

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
            feature_mean=train_mean,
            feature_std=train_std,
            feature_columns=feature_cols,
            open_penalty_pips=2.0,
            time_penalty_pips=0.1,
            unrealized_reward_coef=0.05,
            max_drawdown_pct=0.25,
            risk_adjusted_scale=1.0,
            trade_penalty_pips=2.0,
            reward_scale=0.01
        )

    train_vec_env = SubprocVecEnv([lambda i=i: make_train_env() for i in range(NUM_ENVS)])
    eval_env = SubprocVecEnv([make_eval_env])

    # Увеличенная LSTM сеть
    policy_kwargs = dict(
        net_arch=dict(shared=[128], pi=[64], vf=[64]),
        lstm_hidden_size=96,
        n_lstm_layers=2
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.01,          # уменьшен для снижения случайности
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.05,        # более консервативное обновление
        vf_coef=0.5
    )

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="RecurrentPPO_eurusd")
    eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model", log_path="./eval_log", eval_freq=25_000, deterministic=True, render=False)

    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    # Финальное тестирование на OOS
    best_model = RecurrentPPO.load("./best_model/best_model.zip")
    test_env = SubprocVecEnv([make_test_env])
    equity_test, trades_test = evaluate_model(best_model, test_env, deterministic=True)

    metrics = calculate_metrics(equity_test, trades_test)
    print("\n========== METRICS ON TEST SET ==========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    plt.figure(figsize=(12,6))
    plt.plot(equity_test, label=f"Test Equity Final: {equity_test[-1]:.2f}")
    plt.title("Out-of-Sample Equity Curve")
    plt.xlabel("Steps")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("test_equity_curve.png")
    plt.show()

    if trades_test:
        pd.DataFrame(trades_test).to_csv("test_trades.csv", index=False)
        print("Trades saved to test_trades.csv")

if __name__ == "__main__":
    main()