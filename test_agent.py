import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from utils import compute_full_metrics

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def make_env(df, config, feature_cols, mean=None, std=None):
    return ForexTradingEnv(
        df=df,
        window_size=config['data']['window_size'],
        feature_columns=feature_cols,
        spread_pips=config['environment']['spread_pips'],
        max_slippage_pips=config['environment']['max_slippage_pips'],
        risk_per_trade=config['environment']['risk_per_trade'],
        leverage=config['environment']['leverage'],
        commission_percent=config['environment']['commission_percent'],
        use_atr_sl=config['environment']['use_atr_sl'],
        atr_multiplier_sl=config['environment']['atr_multiplier_sl'],
        atr_multiplier_tp=config['environment']['atr_multiplier_tp'],
        partial_close_enabled=config['environment']['partial_close_enabled'],
        partial_close_ratio=config['environment']['partial_close_ratio'],
        tp_levels=config['environment']['tp_levels'],
        fixed_sl_percent=config['environment']['fixed_sl_percent'],
        fixed_tp_percents=config['environment']['fixed_tp_percents'],
        open_penalty_pips=config['environment']['reward']['open_penalty_pips'],
        time_penalty_pips=config['environment']['reward']['time_penalty_pips'],
        slope_div_reward_scale=config['environment']['reward']['slope_div_reward_scale'],
        reward_scale=config['environment']['reward']['reward_scale'],
        open_bonus_pips=0.0,
        random_start=False,
        min_episode_steps=config['training']['min_episode_steps'],
        episode_max_steps=None,
        feature_mean=mean,
        feature_std=std,
        pip_value=config['data']['pip_value']
    )

def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    equity_curve = []
    closed_trades = []
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
        obs, _, dones, infos = vec_env.step(action)
        episode_starts = dones
        equity_curve.append(float(infos[0]["equity_usd"]))
        trade_info = infos[0].get("last_trade_info")
        if trade_info and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)
        if dones[0]:
            break
    return equity_curve, closed_trades

def main():
    config = load_config()
    
    # Загрузка тестовых данных
    if os.path.exists(config['data']['test_path']):
        test_df, feature_cols = load_and_preprocess_data(config['data']['test_path'])
    else:
        df_full, feature_cols = load_and_preprocess_data(config['data']['path'])
        test_df = df_full[df_full.index >= pd.Timestamp("2024-01-01")].copy()
    
    # Загрузка метаданных финальной модели
    with open("final_metadata.json", "r") as f:
        metadata = json.load(f)
    train_mean = np.array(metadata['mean'])
    train_std = np.array(metadata['std'])
    
    # Создание тестовой среды
    test_env = make_env(test_df, config, feature_cols, train_mean, train_std)
    vec_env = DummyVecEnv([lambda: test_env])
    vec_env = VecNormalize.load("vec_normalize_final.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    # Загрузка модели
    model = RecurrentPPO.load("model_sber_final", env=vec_env)
    
    # Запуск эпизода
    equity_curve, closed_trades = run_one_episode(model, vec_env, deterministic=config['testing']['deterministic'])
    
    # Формирование DataFrame для сделок
    trades_df = pd.DataFrame(closed_trades) if closed_trades else None
    # Добавляем pnl_usd, если его нет (для совместимости со старой версией)
    if trades_df is not None and 'pnl_usd' not in trades_df.columns:
        # Вычисляем PnL приблизительно
        trades_df['pnl_usd'] = trades_df.apply(
            lambda row: (row['exit_price'] - row['entry_price']) * row['lot_size'] * (1 if row['position'] == 1 else -1)
            - row.get('commission_rub', 0), axis=1
        )
    
    # Полный расчёт метрик
    metrics = compute_full_metrics(equity_curve, trades_df, initial_equity=100000.0)
    
    print("========== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ==========")
    print(f"Период: {test_df.index[0]} – {test_df.index[-1]}")
    print(f"Финальная эквити: {metrics['final_equity']:.2f} RUB")
    print(f"Общая доходность: {metrics['total_return_pct']:.2f}%")
    print(f"Коэффициент Шарпа (годовой): {metrics['sharpe']:.3f}")
    print(f"Коэффициент Сортино (годовой): {metrics['sortino']:.3f}")
    print(f"Максимальная просадка: {metrics['max_dd_pct']:.2f}%")
    print(f"Коэффициент Калмара: {metrics['calmar']:.3f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Средняя сделка (RUB): {metrics['avg_trade_usd']:.2f}")
    print(f"Оборачиваемость (сделок/день): {metrics['turnover']:.2f}")
    print(f"Всего закрытых сделок: {len(closed_trades)}")
    
    if config['testing']['save_trades'] and trades_df is not None:
        trades_df.to_csv("test_trade_history_final.csv", index=False)
        print("История сделок сохранена в test_trade_history_final.csv")
    
    # Построение графика эквити
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label=f"Equity (final: {metrics['final_equity']:.2f} RUB)")
    plt.title(f"Test Equity Curve {test_df.index[0].year}-{test_df.index[-1].year}")
    plt.xlabel("10-min bars")
    plt.ylabel("Equity (RUB)")
    plt.grid(True)
    plt.legend()
    textstr = f"Sharpe: {metrics['sharpe']:.2f}\nSortino: {metrics['sortino']:.2f}\nMax DD: {metrics['max_dd_pct']:.1f}%\nCalmar: {metrics['calmar']:.2f}\nProfit Factor: {metrics['profit_factor']:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig("test_equity_curve_final.png")
    plt.show()

if __name__ == "__main__":
    main()