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
from train_walkforward import evaluate_full_period, compute_metrics

def main():
    # Пути к файлам
    CONFIG_PATH = "config.yaml"
    DATA_PATH = "data/SBER_2018_2025.csv"
    GP_TREE_FILES = [f"best_tree_fold_20210101_{i}.txt" for i in range(4)]  # пример
    MANAGER_PATH = "manager_20210101.zip"
    WORKER_PATH = "worker_20210101.zip"
    VECNORM_PATH = "vecnorm_20210101.pkl"

    # Загрузка конфига
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Загружаем сырые данные и рассчитываем индикаторы для всего периода
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

    # Для теста выберем определённый период (например, первое полугодие 2021)
    test_start = pd.Timestamp("2021-01-01")
    test_end = pd.Timestamp("2021-06-30")
    # Для индикаторов нужна история до test_start (например, 2 месяца)
    indicator_start = test_start - pd.DateOffset(months=2)
    test_raw = raw_df.loc[indicator_start:test_end].copy()
    test_full_df, _ = load_and_preprocess_data_from_raw(test_raw)
    test_df = test_full_df.loc[test_start:test_end]

    # Загружаем деревья GP и добавляем признаки
    trees = load_genetic_trees(*GP_TREE_FILES)
    test_df = add_genetic_features_vectorized(test_df, *trees)
    genetic_cols = ['gen_feat0', 'gen_feat1', 'gen_feat2', 'gen_feat3']

    # Загружаем модели
    manager_model = RecurrentPPO.load(MANAGER_PATH)
    worker_model = RecurrentPPO.load(WORKER_PATH)
    temp_env = ForexTradingEnv(test_df, config, manager_model, genetic_cols)
    temp_env.reset(start_idx=0)
    vec_env = DummyVecEnv([lambda: temp_env])
    vec_norm = VecNormalize.load(VECNORM_PATH, vec_env)
    vec_env.close()

    # Тестирование
    test_env = ForexTradingEnv(test_df, config, manager_model, genetic_cols)
    equity_curve = evaluate_full_period(worker_model, test_env, test_df, vec_normalize=vec_norm)
    metrics = compute_metrics(equity_curve, periods_per_year=8760)

    print("\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
    print(f"Итоговая эквити:           {metrics['final_equity']:.4f}")
    print(f"Годовой Sharpe Ratio:      {metrics['sharpe']:.3f}")
    print(f"Sortino Ratio:             {metrics['sortino']:.3f}")
    print(f"Максимальная просадка:     {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio:              {metrics['calmar']:.3f}")

    # График
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