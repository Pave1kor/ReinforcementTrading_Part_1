# test_script.py
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from features import (
    load_and_preprocess_data_from_raw,
    load_genetic_trees,
    add_genetic_features_vectorized,
)
from environment import ForexTradingEnv
from train_walkforward import compute_metrics

def evaluate_with_trades(worker_model, env, vec_normalize=None, deterministic=True):
    """
    Оценка с полным логированием сделок.
    Возвращает:
        - equity_curve (np.array)
        - trades_df (pd.DataFrame): записи по каждой завершённой сделке
        - trade_stats (dict): агрегированные метрики
    """
    env.reset(start_idx=0)
    vec_env = DummyVecEnv([lambda: env])
    if vec_normalize is not None:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        vec_env.obs_rms = vec_normalize.obs_rms
    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    equity_curve = [1.0]
    done = False

    trades = []
    current_trade = None

    while not done:
        action, lstm_states = worker_model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
        )
        obs, reward, done, info = vec_env.step(action)
        episode_starts = done

        pos = env.position
        entry_price = env.entry_price
        idx = env.idx - 1   # только что обработанный бар
        price = env.df.iloc[idx]['Close']

        # Дата текущего бара (если есть колонка "begin", иначе оставляем None)
        date = None
        if 'begin' in env.df.columns:
            date = env.df.iloc[idx]['begin']
        else:
            date = str(idx)

        if current_trade is None and pos != 0.0:
            current_trade = {
                'entry_date': date,
                'entry_price': entry_price,
                'direction': np.sign(pos),
                'size': abs(pos),
            }
        elif current_trade is not None:
            # Определяем, изменилась ли позиция
            prev_size = current_trade['size']
            if abs(pos) < abs(prev_size) - 1e-9:  # позиция уменьшилась
                close_size = prev_size - abs(pos)
                if current_trade['direction'] > 0:
                    pnl = close_size * (price / current_trade['entry_price'] - 1)
                else:
                    pnl = close_size * (1 - price / current_trade['entry_price'])
                trades.append({
                    'Entry Date': current_trade['entry_date'],
                    'Exit Date': date,
                    'Direction': current_trade['direction'],
                    'Entry Price': current_trade['entry_price'],
                    'Exit Price': price,
                    'Size': close_size,
                    'PnL': pnl,
                    'Type': 'partial'
                })
                current_trade['size'] = abs(pos)
                if abs(pos) < 1e-9:
                    current_trade = None
            elif abs(pos) < 1e-9:  # полный выход
                if current_trade['direction'] > 0:
                    pnl = current_trade['size'] * (price / current_trade['entry_price'] - 1)
                else:
                    pnl = current_trade['size'] * (1 - price / current_trade['entry_price'])
                trades.append({
                    'Entry Date': current_trade['entry_date'],
                    'Exit Date': date,
                    'Direction': current_trade['direction'],
                    'Entry Price': current_trade['entry_price'],
                    'Exit Price': price,
                    'Size': current_trade['size'],
                    'PnL': pnl,
                    'Type': 'full'
                })
                current_trade = None

        equity_curve.append(env.equity)

    # Принудительное закрытие, если позиция осталась
    if current_trade is not None:
        last_idx = env.idx
        price = env.df.iloc[last_idx]['Close']
        if current_trade['direction'] > 0:
            pnl = current_trade['size'] * (price / current_trade['entry_price'] - 1)
        else:
            pnl = current_trade['size'] * (1 - price / current_trade['entry_price'])
        trades.append({
            'Entry Date': current_trade['entry_date'],
            'Exit Date': date,
            'Direction': current_trade['direction'],
            'Entry Price': current_trade['entry_price'],
            'Exit Price': price,
            'Size': current_trade['size'],
            'PnL': pnl,
            'Type': 'forced_close'
        })

    vec_env.close()

    # DataFrame со сделками
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # Добавим кумулятивный PnL
        trades_df['Cumulative PnL'] = trades_df['PnL'].cumsum()

    # Агрегированные метрики
    if len(trades) > 0:
        pnls = [t['PnL'] for t in trades]
        win_trades = [p for p in pnls if p > 0]
        loss_trades = [p for p in pnls if p < 0]
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        win_rate = len(win_trades) / len(pnls)
        gross_profit = sum(win_trades)
        gross_loss = sum(loss_trades)
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.inf
        trade_stats = {
            'total_trades': len(pnls),
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': np.mean(win_trades) if win_trades else 0,
            'avg_loss': np.mean(loss_trades) if loss_trades else 0,
        }
    else:
        trade_stats = {'total_trades': 0}

    return np.array(equity_curve), trades_df, trade_stats


def main():
    # Настройте пути под свои файлы
    CONFIG_PATH = "config.yaml"
    DATA_PATH = "data/SBER_2018_2025.csv"
    GP_TREE_FILES = [f"best_tree_fold_20230101_{i}.txt" for i in range(4)]
    MANAGER_PATH = "manager_20230101.zip"
    WORKER_PATH = "worker_20230101.zip"
    VECNORM_PATH = "vecnorm_20230101.pkl"

    # Загрузка конфига
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Подготовка данных
    raw_df = pd.read_csv(DATA_PATH, parse_dates=["begin"], dayfirst=True)
    raw_df.columns = raw_df.columns.str.strip().str.lower()
    raw_df = raw_df.set_index("begin").sort_index()
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    raw_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    # Выберите тестовый период (пример: первое полугодие 2021)
    test_start = pd.Timestamp("2021-01-01")
    test_end = pd.Timestamp("2021-06-30")
    indicator_start = test_start - pd.DateOffset(months=2)
    test_raw = raw_df.loc[indicator_start:test_end].copy()
    test_full_df, _ = load_and_preprocess_data_from_raw(test_raw)
    test_df = test_full_df.loc[test_start:test_end]

    # Загрузка GP-деревьев и добавление признаков
    trees = load_genetic_trees(*GP_TREE_FILES)
    test_df = add_genetic_features_vectorized(test_df, *trees)
    genetic_cols = ['gen_feat0', 'gen_feat1', 'gen_feat2', 'gen_feat3']

    # Загрузка моделей
    manager_model = RecurrentPPO.load(MANAGER_PATH)
    worker_model = RecurrentPPO.load(WORKER_PATH)
    temp_env = ForexTradingEnv(test_df, config, manager_model, genetic_cols)
    temp_env.reset(start_idx=0)
    vec_env = DummyVecEnv([lambda: temp_env])
    vec_norm = VecNormalize.load(VECNORM_PATH, vec_env)
    vec_env.close()

    # Тестирование
    test_env = ForexTradingEnv(test_df, config, manager_model, genetic_cols)
    equity_curve, trades_df, trade_stats = evaluate_with_trades(worker_model, test_env, vec_normalize=vec_norm)

    # Вывод результатов
    print("\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
    print(f"Итоговая эквити:           {equity_curve[-1]:.4f}")
    metrics = compute_metrics(equity_curve, periods_per_year=8760)
    print(f"Годовой Sharpe Ratio:      {metrics['sharpe']:.3f}")
    print(f"Sortino Ratio:             {metrics['sortino']:.3f}")
    print(f"Максимальная просадка:     {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio:              {metrics['calmar']:.3f}")

    print("\n=== СТАТИСТИКА СДЕЛОК ===")
    print(f"Общее количество сделок:    {trade_stats['total_trades']}")
    if trade_stats['total_trades'] > 0:
        print(f"Средний PnL на сделку:     {trade_stats['avg_pnl']:.6f}")
        print(f"Суммарный PnL:             {trade_stats['total_pnl']:.6f}")
        print(f"Win Rate:                  {trade_stats['win_rate']:.2%}")
        print(f"Profit Factor:             {trade_stats['profit_factor']:.2f}")
        print(f"Средняя прибыль (win):     {trade_stats['avg_win']:.6f}")
        print(f"Средний убыток (loss):     {trade_stats['avg_loss']:.6f}")

    # Сохранение сделок в Excel
    if not trades_df.empty:
        output_file = "trade_log.xlsx"
        trades_df.to_excel(output_file, index=False, float_format="%.6f")
        print(f"\n✅ Сделки сохранены в файл: {os.path.abspath(output_file)}")
    else:
        print("\nСделок не было, файл не создан.")

    # График эквити
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Equity', color='blue')
    plt.axhline(1.0, color='gray', linestyle='--', label='Initial capital')
    plt.title('Тест иерархической модели (HRL + GP) на out‑of‑sample данных')
    plt.xlabel('Бары (часы)')
    plt.ylabel('Капитал')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()