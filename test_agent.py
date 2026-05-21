# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

def compute_full_metrics(equity_curve, trades_df=None, initial_equity=10000.0):
    """Расчёт метрик (уже корректен, т.к. equity_curve включает плавающую прибыль)."""
    full_equity = np.array([initial_equity] + equity_curve)
    if len(full_equity) < 2:
        return {k: 0.0 for k in ["sharpe", "sortino", "max_drawdown_pct", "calmar", 
                                  "profit_factor", "win_rate", "avg_trade_usd", "turnover"]}

    returns = np.diff(full_equity) / full_equity[:-1]
    annual_factor = np.sqrt(252 * 24)
    sharpe = annual_factor * returns.mean() / (returns.std() + 1e-8)

    downside_returns = returns[returns < 0]
    sortino = annual_factor * returns.mean() / (downside_returns.std() + 1e-8) if len(downside_returns) > 0 else 0.0

    peak = np.maximum.accumulate(full_equity)
    drawdown = (peak - full_equity) / peak
    max_dd_pct = drawdown.max() * 100

    total_return = (full_equity[-1] - full_equity[0]) / full_equity[0]
    years = len(full_equity) / (252 * 24)
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    calmar = annual_return / (max_dd_pct / 100 + 1e-8)

    metrics = {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd_pct,
        "calmar_ratio": calmar,
        "total_return_pct": total_return * 100,
        "final_equity": full_equity[-1],
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "avg_trade_usd": 0.0,
        "turnover": 0.0
    }

    if trades_df is not None and len(trades_df) > 0:
        dollar_pnl = trades_df['net_pips'] * trades_df['lot_size'] * 0.0001
        gains = dollar_pnl[dollar_pnl > 0].sum()
        losses = abs(dollar_pnl[dollar_pnl < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf
        win_rate = (dollar_pnl > 0).mean() * 100
        avg_trade_usd = dollar_pnl.mean()
        n_days = len(full_equity) / 24
        turnover = len(trades_df) / n_days if n_days > 0 else 0.0

        metrics.update({
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "avg_trade_usd": avg_trade_usd,
            "turnover": turnover
        })
    return metrics

def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    initial_equity = vec_env.get_attr("initial_equity_usd")[0]
    equity_curve = [initial_equity]
    closed_trades = []
    
    while True:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
        )
        obs, rewards, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        episode_starts = dones
        info = infos[0]

        equity_curve.append(float(info["equity_usd"]))
        trade_info = info.get("last_trade_info")
        if trade_info and trade_info.get("event") == "CLOSE":
            if not closed_trades or closed_trades[-1] != trade_info:
                closed_trades.append(trade_info)
        if done:
            break
    return equity_curve, closed_trades

def main():
    DATA_PATH = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    MODEL_PATH = "model_eurusd_best"   # после переобучения путь будет другой
    WIN = 60
    # Используем новые параметры, согласованные с обучением
    BASE_SL_PIPS = 45.0
    BASE_TP_PIPS = 90.0
    K_SL = 0.3
    K_TP = 0.6

    df, feature_cols = load_and_preprocess_data(DATA_PATH)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_features = train_df[feature_cols].values.astype(np.float32)
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    test_env = ForexTradingEnv(
        df=test_df,
        window_size=WIN,
        feature_columns=feature_cols,
        spread_pips=1.0,
        commission_pips=0.0,
        max_slippage_pips=1.0,
        random_start=False,
        episode_max_steps=None,
        feature_mean=train_mean,
        feature_std=train_std,
        base_sl_pips=BASE_SL_PIPS,
        base_tp_pips=BASE_TP_PIPS,
        k_sl=K_SL,
        k_tp=K_TP,
        risk_per_trade=0.005,          # уменьшено
        open_penalty_pips=0.5,
        time_penalty_pips=0.001,
        trailing_atr_mult=2.0,         # динамический трейлинг
        min_atr_pips=10.0,
    )

    vec_test_env = DummyVecEnv([lambda: test_env])
    model = RecurrentPPO.load(MODEL_PATH, env=vec_test_env)

    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    trades_df = pd.DataFrame(closed_trades) if closed_trades else None
    metrics = compute_full_metrics(equity_curve, trades_df=trades_df, initial_equity=10000.0)

    print("========== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ==========")
    print(f"Финальная эквити: {metrics['final_equity']:.2f} USD")
    print(f"Общая доходность: {metrics['total_return_pct']:.2f}%")
    print(f"Коэффициент Шарпа (годовой): {metrics['sharpe_ratio']:.3f}")
    print(f"Коэффициент Сортино (годовой): {metrics['sortino_ratio']:.3f}")
    print(f"Максимальная просадка: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Коэффициент Калмара: {metrics['calmar_ratio']:.3f}")
    if metrics['profit_factor'] > 0:
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Средняя сделка (USD): {metrics['avg_trade_usd']:.2f}")
        print(f"Оборачиваемость (сделок/день): {metrics['turnover']:.2f}")
    print(f"Всего закрытых сделок: {len(closed_trades)}")

    if closed_trades:
        trades_df.to_csv("test_trade_history.csv", index=False)
        print("История сделок сохранена в test_trade_history.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label=f"Equity (final: {metrics['final_equity']:.2f} USD)", linewidth=1.5)
    plt.title("Test Episode Equity Curve (with unrealized PnL)")
    plt.xlabel("Bars")
    plt.ylabel("Equity (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    textstr = f"Sharpe: {metrics['sharpe_ratio']:.2f}\nSortino: {metrics['sortino_ratio']:.2f}\nMax DD: {metrics['max_drawdown_pct']:.1f}%\nCalmar: {metrics['calmar_ratio']:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig("test_equity_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()