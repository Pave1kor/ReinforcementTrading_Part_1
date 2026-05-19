# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def run_one_episode(model, vec_env, deterministic=True):
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

        # ЗАЩИТА ОТ АВТО-RESET: Если это финал, берем данные из info
        if done:
            # Записываем последнее реальное значение баланса перед сбросом
            if isinstance(info, dict) and "equity_usd" in info:
                equity_curve.append(float(info["equity_usd"]))
            elif equity_curve:
                equity_curve.append(equity_curve[-1])
            
            # Пытаемся забрать финальную сделку (например, END_OF_DATA)
            if isinstance(info, dict) and "last_trade_info" in info:
                trade_info = info["last_trade_info"]
                if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
                    closed_trades.append(trade_info)
            break
        else:
            # Если эпизод продолжается, пишем данные из живой среды как обычно
            equity_curve.append(vec_env.get_attr("equity_usd")[0])
            
            trade_info = vec_env.get_attr("last_trade_info")[0]
            if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
                # Защита от дублирования одной и той же сделки на разных барах
                if not closed_trades or closed_trades[-1] != trade_info:
                    closed_trades.append(trade_info) 

    return equity_curve, closed_trades


def main():
    # Choose the dataset you want to evaluate on
    file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # Разделяем выборку на обучающую и тестовую для правильного расчета нормализации
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Извлекаем признаки для расчета параметров нормализации (БЕЗ учета таргетов/цен, если нужно)
    # Считаем среднее и дисперсию СТРОГО на обучающей выборке (Защита от Data Leakage)
    train_features_df = train_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    train_features = train_features_df.values.astype(np.float32)
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    # Must match training params
    SL_OPTS = [30, 90]
    TP_OPTS = [30, 90]
    WIN = 60

    test_env = ForexTradingEnv(
        df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            feature_mean=train_mean, 
            feature_std=train_std,  
            hold_reward_weight=0.0,
            open_penalty_pips=0.1,      # half a pip per open
            time_penalty_pips=0.0,     # 0.02 pips per bar in trade
            unrealized_delta_weight=1.0
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    # Load best model
    model = RecurrentPPO.load("model_eurusd_best", env=vec_test_env)

    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # Save trades
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history_output.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"Closed trade history saved to {out_csv}")
    else:
        print("No closed trades recorded.")

    # Plot equity
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label=f"Equity Final: {equity_curve[-1]:.2f}$")
    plt.title("Equity Curve - Test Evaluation (Clean)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('my_plot.png')
    print("Plot saved to my_plot.png")


if __name__ == "__main__":
    main()
