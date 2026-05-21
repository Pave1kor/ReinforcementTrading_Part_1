import pandas as pd
import numpy as np
import pandas_ta as ta

def load_and_preprocess_data(csv_path: str):
    """
    Загрузка данных и расчёт признаков БЕЗ look-ahead.
    Добавлены EMA 50/200, их пересечение и логарифмическая доходность.
    """
    df = pd.read_csv(csv_path, parse_dates=["Time (EET)"], dayfirst=True)
    df.columns = df.columns.str.strip()
    df = df.set_index("Time (EET)").sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # pressure
    df["pressure"] = (df["Close"] - df["Low"]) - (df["High"] - df["Close"])
    df["bar_range"] = df["High"] - df["Low"]
    df["norm_pressure"] = np.where(df["bar_range"] != 0, df["pressure"] / df["bar_range"], 0.0)

    # delta
    df["delta"] = df["norm_pressure"] * df["Volume"]

    # скользящие средние (ALMA)
    df["cvd_avg"] = ta.alma(df["delta"], length=50, sigma=0.85, distribution_offset=4)
    df["price_avg"] = ta.alma(df["Close"], length=34, sigma=0.85, distribution_offset=4)

    # сырые наклоны
    cvd_linreg_curr = ta.linreg(df["cvd_avg"], length=8, offset=0).squeeze()
    cvd_linreg_prev = ta.linreg(df["cvd_avg"], length=8, offset=1).squeeze()
    df["cvd_slope_raw"] = cvd_linreg_curr - cvd_linreg_prev

    price_linreg_curr = ta.linreg(df["price_avg"], length=8, offset=0).squeeze()
    price_linreg_prev = ta.linreg(df["price_avg"], length=8, offset=1).squeeze()
    df["price_slope_raw"] = price_linreg_curr - price_linreg_prev

    # волатильность и объём (ALMA)
    tr_series = ta.true_range(df["High"], df["Low"], df["Close"])
    df["alma_atr"] = ta.alma(tr_series, length=300, sigma=0.85, distribution_offset=4)
    df["alma_vol"] = ta.alma(df["Volume"], length=300, sigma=0.85, distribution_offset=4)

    # нормирование
    df["price_slope"] = np.where(df["alma_atr"] != 0, df["price_slope_raw"] / df["alma_atr"], df["price_slope_raw"])
    df["cvd_slope"] = np.where(df["alma_vol"] != 0, df["cvd_slope_raw"] / df["alma_vol"], df["cvd_slope_raw"])

    df["slope_div"] = df["price_slope"] * df["cvd_slope"] * 10000.0
    df["abs_slope_div"] = df["slope_div"].abs()

    # дивергенции
    df["bull_div"] = ((df["price_slope"] < 0) & (df["cvd_slope"] > 0)).astype(float)
    df["bear_div"] = ((df["price_slope"] > 0) & (df["cvd_slope"] < 0)).astype(float)

    # RSI
    df["rsi_norm"] = ta.rsi(df["Close"], length=14) / 100.0

    # отклонение цены от средней
    df["price_dist_from_avg"] = np.where(df["alma_atr"] != 0, (df["Close"] - df["price_avg"]) / df["alma_atr"], 0.0)

    # относительный объём
    df["relative_volume"] = np.where(df["alma_vol"] != 0, df["Volume"] / df["alma_vol"], 1.0)

    # ========== НОВЫЕ ПРИЗНАКИ ==========
    # EMA 50 и 200
    df["ema_50"] = ta.ema(df["Close"], length=50)
    df["ema_200"] = ta.ema(df["Close"], length=200)
    # Пересечение EMA (сигнал 1 если быстрая выше медленной, иначе 0)
    df["ema_cross"] = (df["ema_50"] > df["ema_200"]).astype(float)
    # Логарифмическая доходность за 1 бар
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    # ATR ratio (ATR / Close)
    df["atr_ratio"] = df["alma_atr"] / df["Close"]

    # удаляем строки с NaN (первые 300+ баров)
    df.dropna(inplace=True)

    # Обновлённый список фичей
    feature_cols = [
        "alma_atr",
        "relative_volume",
        "price_dist_from_avg",
        "rsi_norm",
        "slope_div",
        "abs_slope_div",
        "cvd_slope",
        "price_slope",
        "bull_div",
        "bear_div",
        "ema_cross",      # новое
        "log_return",     # новое
        "atr_ratio",      # новое
    ]
    return df, feature_cols