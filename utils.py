# utils.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, List, Dict, Any

def check_stationarity(series: pd.Series, significance_level: float = 0.05) -> bool:

    # Проверка стационарности ряда с помощью ADF-теста.
    # Возвращает True если ряд стационарен.

    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value < significance_level

def make_stationary(df: pd.DataFrame, columns: List[str], threshold: float = 0.05) -> pd.DataFrame:
    
    # Применяет дифференцирование к нестационарным признакам.
    # Возвращает копию DataFrame с преобразованными колонками.

    df_out = df.copy()
    for col in columns:
        if not check_stationarity(df[col], threshold):
            # Первая разность
            df_out[col] = df[col].diff()
            # Если всё ещё нестационарна – вторая разность
            if not check_stationarity(df_out[col].dropna(), threshold):
                df_out[col] = df[col].diff().diff()
    # Удаляем NA, возникшие из-за дифференцирования
    df_out.dropna(inplace=True)
    return df_out

def compute_rolling_metrics(equity_curve: List[float], initial_equity: float = 100000.0,
                            window: int = 252*39) -> Dict[str, float]:
    
    # Расчёт скользящих метрик для мониторинга в обучении.

    full_equity = np.array([initial_equity] + equity_curve)
    if len(full_equity) < 2:
        return {"sharpe": 0.0, "max_dd": 0.0, "calmar": 0.0}
    
    # Ограничимся последними window шагами
    if len(full_equity) > window:
        full_equity = full_equity[-window:]
    
    returns = np.diff(full_equity) / full_equity[:-1]
    annual_factor = np.sqrt(252 * 39)  # 10-минутных баров в день ~39
    
    sharpe = annual_factor * returns.mean() / (returns.std() + 1e-8)
    
    peak = np.maximum.accumulate(full_equity)
    drawdown = (peak - full_equity) / peak
    max_dd = drawdown.max()
    
    # Calmar: годовая доходность / максимальная просадка
    total_return = (full_equity[-1] - full_equity[0]) / full_equity[0]
    years = len(full_equity) / (252 * 39)
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    calmar = annual_return / (max_dd + 1e-8)
    
    return {"sharpe": sharpe, "max_dd": max_dd, "calmar": calmar}

def compute_full_metrics(equity_curve: List[float], trades_df: pd.DataFrame = None,
                         initial_equity: float = 100000.0) -> Dict[str, float]:
    
    # Полный набор метрик для финального отчёта.
    
    full_equity = np.array([initial_equity] + equity_curve)
    if len(full_equity) < 2:
        return {k: 0.0 for k in ["sharpe", "sortino", "max_dd_pct", "calmar",
                                  "total_return_pct", "profit_factor", "win_rate",
                                  "avg_trade_usd", "turnover", "final_equity"]}
    
    returns = np.diff(full_equity) / full_equity[:-1]
    annual_factor = np.sqrt(252 * 39)
    sharpe = annual_factor * returns.mean() / (returns.std() + 1e-8)
    
    downside = returns[returns < 0]
    sortino = annual_factor * returns.mean() / (downside.std() + 1e-8) if len(downside) > 0 else 0.0
    
    peak = np.maximum.accumulate(full_equity)
    drawdown = (peak - full_equity) / peak
    max_dd_pct = drawdown.max() * 100
    
    total_return = (full_equity[-1] - full_equity[0]) / full_equity[0]
    total_return_pct = total_return * 100
    years = len(full_equity) / (252 * 39)
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    calmar = annual_return / (max_dd_pct / 100 + 1e-8)
    
    metrics = {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd_pct,
        "calmar": calmar,
        "total_return_pct": total_return_pct,
        "final_equity": full_equity[-1],
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "avg_trade_usd": 0.0,
        "turnover": 0.0
    }
    
    if trades_df is not None and len(trades_df) > 0:
        # Предполагаем, что trades_df содержит колонки: pnl_usd (чистая прибыль по сделке)
        if 'pnl_usd' in trades_df.columns:
            gains = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
            losses = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
            metrics['profit_factor'] = gains / losses if losses > 0 else np.inf
            metrics['win_rate'] = (trades_df['pnl_usd'] > 0).mean() * 100
            metrics['avg_trade_usd'] = trades_df['pnl_usd'].mean()
            n_days = len(full_equity) / 39
            metrics['turnover'] = len(trades_df) / n_days if n_days > 0 else 0.0
    
    return metrics

def purged_walk_forward_splits(df: pd.DataFrame, val_years: List[int], purge_days: int = 5):
    
    # Генерирует train/val/test индексы с очисткой (purge) между train и val.
    # Каждая итерация: train = до val_year - purge_days, val = val_year, test = val_year+1 (до следующего purge).
    # Возвращает список словарей с индексами.
    
    df['date'] = df.index
    splits = []
    for i, val_year in enumerate(val_years):
        # Границы
        train_end = pd.Timestamp(f"{val_year}-01-01") - pd.Timedelta(days=purge_days)
        val_start = pd.Timestamp(f"{val_year}-01-01")
        val_end = pd.Timestamp(f"{val_year+1}-01-01") - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = pd.Timestamp(f"{val_year+2}-01-01") - pd.Timedelta(days=purge_days) if val_year+2 <= max(val_years)+1 else df.index.max()
        
        train_idx = df[df.index < train_end].index
        val_idx = df[(df.index >= val_start) & (df.index <= val_end)].index
        test_idx = df[(df.index >= test_start) & (df.index <= test_end)].index
        
        if len(train_idx) > 500 and len(val_idx) > 200 and len(test_idx) > 200:
            splits.append({
                'name': f"train_{train_end.year}_val_{val_year}_test_{test_start.year}",
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx
            })
    return splits