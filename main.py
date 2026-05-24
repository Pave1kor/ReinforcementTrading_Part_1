# main.py
import os
import yaml
import numpy as np
import pandas as pd
from datetime import timedelta
from features import (
    load_and_preprocess_data,
    load_genetic_trees,
    add_genetic_features_vectorized,
    train_genetic_on_split,
)
from environment import ForexTradingEnv
from manager_env import ManagerEnv
from train_walkforward import (
    train_manager_sequential,
    train_worker_sequential,
    evaluate_full_period,
    compute_metrics,
)

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def walk_forward_semiannual(
    csv_path: str,
    config: dict,
    start_train_date="2018-01-01",
    test_start="2021-01-01",
    test_end="2025-12-31",
    train_window_years=3,
    step_months=6
):
    # Честный walk-forward с окнами по полугодиям.
    # Для каждого тестового полугодия индикаторы рассчитываются заново
    # без подглядывания в будущее.

    # Загружаем сырые данные один раз
    raw_df = pd.read_csv(csv_path, parse_dates=["begin"], dayfirst=True)
    raw_df.columns = raw_df.columns.str.strip().str.lower()
    raw_df = raw_df.set_index("begin").sort_index()
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    raw_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    full_start = raw_df.index.min()
    full_end = raw_df.index.max()

    # Генерируем список тестовых окон
    test_starts = pd.date_range(start=test_start, end=test_end, freq=f"{step_months}MS")
    results = []

    for test_start_date in test_starts:
        test_end_date = test_start_date + pd.DateOffset(months=step_months) - timedelta(days=1)
        if test_end_date > full_end:
            break
        train_end_date = test_start_date - timedelta(days=1)
        train_start_date = train_end_date - pd.DateOffset(years=train_window_years)
        if train_start_date < full_start:
            train_start_date = full_start
        # Дополнительная история для расчёта индикаторов (например, 2 месяца)
        extra_lookback = pd.DateOffset(months=2)
        indicator_start = max(full_start, train_start_date - extra_lookback)

        print(f"\n=== Фолд: тест {test_start_date.date()} – {test_end_date.date()} ===")
        print(f"Тренировка: {train_start_date.date()} – {train_end_date.date()}")

        # 1. Тренировочные данные (без тестового периода)
        train_raw = raw_df.loc[indicator_start:train_end_date].copy()
        train_df, _ = load_and_preprocess_data_from_raw(train_raw)
        train_df = train_df.loc[train_start_date:train_end_date]  # обрезаем лишнюю историю
        if len(train_df) < 500:
            print("Недостаточно тренировочных данных, пропускаем")
            continue

        # 2. Генетическая эволюция на train/val разбиении (последние 20% – валидация)
        split_idx = int(len(train_df) * 0.8)
        gp_train_df = train_df.iloc[:split_idx]
        gp_val_df = train_df.iloc[split_idx:]
        print(f"Запуск GP: train {len(gp_train_df)} баров, val {len(gp_val_df)} баров")
        trees = train_genetic_on_split(gp_train_df, gp_val_df, generations=15, pop_size=80)
        # Сохраняем деревья
        for i, t in enumerate(trees):
            with open(f"best_tree_fold_{test_start_date.strftime('%Y%m%d')}_{i}.txt", "w") as f:
                f.write(str(t))

        # Добавляем генетические признаки к тренировочным данным
        train_df = add_genetic_features_vectorized(train_df, *trees)
        genetic_cols = ['gen_feat0', 'gen_feat1', 'gen_feat2', 'gen_feat3']

        # 3. Обучение Manager и Worker
        print("Обучение Manager...")
        manager_model, _ = train_manager_sequential(
            ManagerEnv, train_df, config,
            total_timesteps=config['training'].get('manager_timesteps', 20000)
        )
        print("Обучение Worker...")
        worker_model, vec_norm = train_worker_sequential(
            ForexTradingEnv, train_df, config, manager_model, genetic_cols,
            total_timesteps=config['training']['total_timesteps']
        )
        # Сохранение моделей
        manager_model.save(f"manager_{test_start_date.strftime('%Y%m%d')}.zip")
        worker_model.save(f"worker_{test_start_date.strftime('%Y%m%d')}.zip")
        vec_norm.save(f"vecnorm_{test_start_date.strftime('%Y%m%d')}.pkl")

        # 4. Тестирование на независимом тестовом периоде
        test_raw = raw_df.loc[indicator_start:test_end_date].copy()
        test_full_df, _ = load_and_preprocess_data_from_raw(test_raw)
        test_df = test_full_df.loc[test_start_date:test_end_date]
        if len(test_df) < 100:
            print("Слишком короткий тестовый период, пропускаем")
            continue
        test_df = add_genetic_features_vectorized(test_df, *trees)
        test_env = ForexTradingEnv(test_df, config, manager_model, genetic_cols)
        equity_curve = evaluate_full_period(worker_model, test_env, test_df, vec_normalize=vec_norm)
        metrics = compute_metrics(equity_curve, periods_per_year=8760)

        print(f"Результаты: Final Equity {metrics['final_equity']:.3f}, "
              f"Sharpe {metrics['sharpe']:.3f}, Sortino {metrics['sortino']:.3f}, "
              f"Max DD {metrics['max_drawdown']:.3f}, Calmar {metrics['calmar']:.3f}")
        results.append((test_start_date, metrics))

    # Итоговый отчёт
    print("\n=== Сводка по всем фолдам ===")
    for date, m in results:
        print(f"{date.date()}: Equity={m['final_equity']:.3f}, Sharpe={m['sharpe']:.3f}, "
              f"MaxDD={m['max_drawdown']:.3f}, Calmar={m['calmar']:.3f}")

# Импорт функции расчёта индикаторов из features
from features import load_and_preprocess_data_from_raw

def main():
    config = load_config("config.yaml")
    DATA_PATH = config['data']['path']

    walk_forward_semiannual(
        csv_path=DATA_PATH,
        config=config,
        start_train_date="2018-01-01",
        test_start="2021-01-01",
        test_end="2025-12-31",
        train_window_years=3,
        step_months=6
    )

if __name__ == "__main__":
    main()