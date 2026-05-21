import pandas as pd
import numpy as np
import pandas_ta as ta

def load_and_preprocess_data(csv_path: str):
    """
    Загрузка данных для акций Сбербанка (10-минутные бары).
    Ожидаемые колонки: begin, open, high, low, close, volume.
    """
    df = pd.read_csv(csv_path, parse_dates=["begin"], dayfirst=True)
    df.columns = df.columns.str.strip()
    df = df.set_index("begin").sort_index()
    # Переименуем колонки в стандартный формат
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Базовые производные (без look-ahead) ---
    df["pressure"] = (df["Close"] - df["Low"]) - (df["High"] - df["Close"])
    df["bar_range"] = df["High"] - df["Low"]
    df["norm_pressure"] = np.where(df["bar_range"] != 0, df["pressure"] / df["bar_range"], 0.0)
    df["delta"] = df["norm_pressure"] * df["Volume"]

    # ALMA для сглаживания
    df["cvd_avg"] = ta.alma(df["delta"], length=50, sigma=0.85, distribution_offset=4)
    df["price_avg"] = ta.alma(df["Close"], length=34, sigma=0.85, distribution_offset=4)

    # Наклоны (offset=1, 2 – без look-ahead)
    cvd_linreg_curr = ta.linreg(df["cvd_avg"], length=8, offset=1).squeeze()
    cvd_linreg_prev = ta.linreg(df["cvd_avg"], length=8, offset=2).squeeze()
    df["cvd_slope_raw"] = cvd_linreg_curr - cvd_linreg_prev

    price_linreg_curr = ta.linreg(df["price_avg"], length=8, offset=1).squeeze()
    price_linreg_prev = ta.linreg(df["price_avg"], length=8, offset=2).squeeze()
    df["price_slope_raw"] = price_linreg_curr - price_linreg_prev

    # Волатильность (ATR через ALMA)
    tr_series = ta.true_range(df["High"], df["Low"], df["Close"])
    df["alma_atr"] = ta.alma(tr_series, length=300, sigma=0.85, distribution_offset=4)
    df["alma_vol"] = ta.alma(df["Volume"], length=300, sigma=0.85, distribution_offset=4)

    # Нормированные наклоны
    df["price_slope"] = np.where(df["alma_atr"] != 0, df["price_slope_raw"] / df["alma_atr"], df["price_slope_raw"])
    df["cvd_slope"] = np.where(df["alma_vol"] != 0, df["cvd_slope_raw"] / df["alma_vol"], df["cvd_slope_raw"])

    # slope_div (со знаком) – используется только в среде
    df["slope_div"] = df["price_slope"] * df["cvd_slope"] * 10000.0

    # Дивергенции (бинарные)
    df["bull_div"] = ((df["price_slope"] < 0) & (df["cvd_slope"] > 0)).astype(float)
    df["bear_div"] = ((df["price_slope"] > 0) & (df["cvd_slope"] < 0)).astype(float)

    # Взвешенная сила дивергенции
    df["weighted_div"] = np.where(
        (df["bull_div"] == 1) | (df["bear_div"] == 1),
        df["price_slope"].abs() * df["cvd_slope"].abs(),
        0.0
    )

    # RSI
    df["rsi_norm"] = ta.rsi(df["Close"], length=14) / 100.0

    # Отклонение цены от средней
    df["price_dist_from_avg"] = np.where(df["alma_atr"] != 0, (df["Close"] - df["price_avg"]) / df["alma_atr"], 0.0)

    # ========== ПРИЗНАКИ ДЛЯ 10-МИНУТНЫХ ДАННЫХ ==========
    df["return_10"] = df["Close"].pct_change(10)
    df["norm_return_10"] = df["return_10"] / (df["alma_atr"] / df["Close"] + 1e-8)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd_hist"] = macd["MACDh_12_26_9"] / (df["alma_atr"] / df["Close"] + 1e-8)

    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["adx"] = adx_df["ADX_14"] / 100.0

    rolling_mean_vol = df["Volume"].rolling(100).mean()
    rolling_std_vol = df["Volume"].rolling(100).std()
    df["volume_zscore"] = (df["Volume"] - rolling_mean_vol) / (rolling_std_vol + 1e-8)
    df["volume_zscore"] = df["volume_zscore"].clip(-3, 3)

    df["volatility_regime"] = df["alma_atr"] / (df["alma_atr"].rolling(100).mean() + 1e-8)

    df["div_persistence"] = df["weighted_div"].rolling(5).mean()

    df.dropna(inplace=True)

    feature_cols = [
        "norm_return_10",
        "macd_hist",
        "adx",
        "rsi_norm",
        "price_dist_from_avg",
        "div_persistence",
        "volume_zscore",
        "volatility_regime",
        "cvd_slope",
    ]
    return df, feature_cols