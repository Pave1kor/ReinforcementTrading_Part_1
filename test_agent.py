import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

def compute_full_metrics(equity_curve, trades_df=None, initial_equity=100000.0):
    full_equity = np.array([initial_equity] + equity_curve)
    if len(full_equity) < 2:
        return {k: 0.0 for k in ["sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "calmar_ratio",
                                  "total_return_pct", "final_equity", "profit_factor", "win_rate",
                                  "avg_trade_usd", "turnover"]}

    returns = np.diff(full_equity) / full_equity[:-1]
    annual_factor = np.sqrt(252 * 39)
    sharpe = annual_factor * returns.mean() / (returns.std() + 1e-8)

    downside = returns[returns < 0]
    sortino = annual_factor * returns.mean() / (downside.std() + 1e-8) if len(downside) > 0 else 0.0

    peak = np.maximum.accumulate(full_equity)
    drawdown = (peak - full_equity) / peak
    max_dd_pct = drawdown.max() * 100

    total_return = (full_equity[-1] - full_equity[0]) / full_equity[0]
    years = len(full_equity) / (252 * 39)
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
        dollar_pnl = trades_df['net_pips'] * trades_df['lot_size'] * 0.01
        gains = dollar_pnl[dollar_pnl > 0].sum()
        losses = abs(dollar_pnl[dollar_pnl < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf
        win_rate = (dollar_pnl > 0).mean() * 100
        avg_trade_usd = dollar_pnl.mean()
        n_days = len(full_equity) / 39
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
    equity_curve = []
    closed_trades = []

    while True:
        action, lstm_states = model.predict(obs, state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=deterministic)
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
    TEST_DATA_PATH = "data/SBER_test_2023_daily.csv"
    if not os.path.exists(TEST_DATA_PATH):
        print("Тестовый файл не найден. Создайте data/SBER_test_2023_daily.csv")
        return
    if not os.path.exists("final_metadata.json"):
        print("final_metadata.json не найден. Запустите train_agent_walkforward.py сначала.")
        return

    with open("final_metadata.json", "r") as f:
        meta = json.load(f)
    feature_cols = meta['feature_cols']
    train_mean = np.array(meta['mean'])
    train_std = np.array(meta['std'])

    df_test, _ = load_and_preprocess_data(TEST_DATA_PATH)

    WIN = 60
    test_env = ForexTradingEnv(
        df=df_test, window_size=WIN, feature_columns=feature_cols,
        spread_pips=1.0, commission_pips=0.0, max_slippage_pips=1.0,
        random_start=False, episode_max_steps=None,
        feature_mean=train_mean, feature_std=train_std,
        risk_per_trade=0.005, base_sl_pips=40.0, base_tp_pips=80.0,
        k_sl=0.3, k_tp=0.6, open_penalty_pips=0.0, time_penalty_pips=0.0005,
        trailing_atr_mult=2.0, min_atr_pips=5.0, slope_div_reward_scale=0.002,
        open_bonus_pips=5.0, reward_scale=0.002, pip_value=0.01, lot_size=1.0, leverage=1.0
    )

    vec_env = DummyVecEnv([lambda: test_env])
    if os.path.exists("vec_normalize_final.pkl"):
        vec_env = VecNormalize.load("vec_normalize_final.pkl", vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)

    model = RecurrentPPO.load("model_sber_final", env=vec_env)

    equity_curve, closed_trades = run_one_episode(model, vec_env, deterministic=True)

    trades_df = pd.DataFrame(closed_trades) if closed_trades else None
    metrics = compute_full_metrics(equity_curve, trades_df=trades_df, initial_equity=100000.0)

    print("========== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ НА 2024-2025 ==========")
    print(f"Период: {df_test.index[0]} – {df_test.index[-1]}")
    print(f"Финальная эквити: {metrics['final_equity']:.2f} RUB")
    print(f"Общая доходность: {metrics['total_return_pct']:.2f}%")
    print(f"Коэффициент Шарпа (годовой): {metrics['sharpe_ratio']:.3f}")
    print(f"Коэффициент Сортино (годовой): {metrics['sortino_ratio']:.3f}")
    print(f"Максимальная просадка: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Коэффициент Калмара: {metrics['calmar_ratio']:.3f}")
    if metrics['profit_factor'] > 0:
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Средняя сделка (RUB): {metrics['avg_trade_usd']:.2f}")
        print(f"Оборачиваемость (сделок/день): {metrics['turnover']:.2f}")
    print(f"Всего закрытых сделок: {len(closed_trades)}")

    if closed_trades:
        trades_df.to_csv("test_trade_history_final.csv", index=False)
        print("История сделок сохранена в test_trade_history_final.csv")

    plt.figure(figsize=(12,6))
    plt.plot(equity_curve, label=f"Equity (final: {metrics['final_equity']:.2f} RUB)")
    plt.title("Test Equity Curve 2024-2025")
    plt.xlabel("10-min bars")
    plt.ylabel("Equity (RUB)")
    plt.grid(True)
    plt.legend()
    textstr = f"Sharpe: {metrics['sharpe_ratio']:.2f}\nSortino: {metrics['sortino_ratio']:.2f}\nMax DD: {metrics['max_drawdown_pct']:.1f}%\nCalmar: {metrics['calmar_ratio']:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig("test_equity_curve_final.png")
    plt.show()

if __name__ == "__main__":
    main()