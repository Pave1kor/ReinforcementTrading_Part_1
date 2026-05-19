# test_agent.py
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

        # ФИКС: Все данные берем СТРОГО из info, так как живая среда авто-сбрасывается
        if done:
            # Ищем эквити в terminal_observation (куда SB3 бережно складывает состояние до reset)
            if "terminal_observation" in info and isinstance(info["terminal_observation"], dict):
                # Если твоя среда сама не прокинула эквити в корень info при завершении:
                # Нам надежнее забрать его из корня info, так как в trading_env.py 
                # ты явно пишешь: info.update({"equity_usd": ...}) прямо перед return.
                pass
            
            if isinstance(info, dict) and "equity_usd" in info:
                equity_curve.append(float(info["equity_usd"]))
            elif equity_curve:
                equity_curve.append(equity_curve[-1])
            
            # Забираем финальную сделку, если она была закрыта по END_OF_DATA
            if isinstance(info, dict) and "last_trade_info" in info:
                trade_info = info["last_trade_info"]
                if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
                    closed_trades.append(trade_info)
            break
        else:
            # Если эпизод продолжается, берем данные из info текущего шага
            if isinstance(info, dict) and "equity_usd" in info:
                equity_curve.append(float(info["equity_usd"]))
            
            # ФИКС СДЕЛОК: Проверяем, что событие CLOSE произошло именно НА ТЕКУЩЕМ шаге.
            # В trading_env.py в момент закрытия ты пишешь: "step": self.current_step.
            # Сверяем этот шаг с текущим шагом внутри info, чтобы избежать дублирования на "HOLD" барах.
            if isinstance(info, dict) and "last_trade_info" in info:
                trade_info = info["last_trade_info"]
                if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
                    # Проверяем, совпадает ли шаг закрытия сделки с шагом среды в info.
                    # Но в info у тебя нет явного текущего шага, зато мы можем проверить, 
                    # что мы еще не добавляли эту сделку (сверяем уникальный step сделки).
                    if not closed_trades or closed_trades[-1]["step"] != trade_info["step"]:
                        closed_trades.append(trade_info)

    return equity_curve, closed_trades


def main():
    # ФИКС из предыдущего шага: принудительно очищаем графики во избежание OOM
    plt.close('all')

    file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # Разделяем выборку на обучающую и тестовую
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Считаем параметры нормализации СТРОГО на трейне
    train_features_df = train_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    train_features = train_features_df.values.astype(np.float32)
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    SL_OPTS = [1.0, 1.5]
    TP_OPTS = [2.0, 3.5]
    WIN = 60

    # ФИКС: Сюда передается test_df. Внутри конструктора ForexTradingEnv 
    # теперь стоит принудительный .reset_index(drop=True), поэтому KeyError: 0 больше не возникнет.
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
        open_penalty_pips=0.1,      
        time_penalty_pips=0.005,     
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    # Load best model
    model = RecurrentPPO.load("model_eurusd_best", env=vec_test_env)

    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # Save trades
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history_output.csv"
        # Дропаем дубликаты по шагу закрытия на всякий случай (двойная защита)
        trades_df.drop_duplicates(subset=["step", "event"], inplace=True)
        trades_df.to_csv(out_csv, index=False)
        print(f"Closed trade history saved to {out_csv}. Total trades: {len(trades_df)}")
    else:
        print("No closed trades recorded.")

    # Plot equity
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label=f"Equity Final: {equity_curve[-1]:.2f}$")
    plt.title("Equity Curve - Test Evaluation (Clean & Fixed)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('my_plot.png')
    plt.close()
    print("Plot saved to my_plot.png")


if __name__ == "__main__":
    main()
