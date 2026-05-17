import pandas as pd
import numpy as np
import pandas_ta as ta


def load_and_preprocess_data(csv_path: str):
    """
    Loads EURUSD data from CSV and preprocesses it by adding RELATIVE technical features.

    CSV expected columns: [Time (EET), Open, High, Low, Close, Volume]
    The returned DataFrame still contains OHLCV for env internals,
    but `feature_cols` lists only the RELATIVE columns to feed the agent.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=["Time (EET)"],
        dayfirst=True,
    )

    # Strip any trailing spaces in headers (e.g. "Volume ")
    df.columns = df.columns.str.strip()

    # Datetime index
    df = df.set_index("Time (EET)")
    df.sort_index(inplace=True)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- Indicators ----

# ----- Pressure -----
    df["pressure"] = (df["Close"] - df["Low"]) - (df["High"] - df["Close"])
    df["bar_range"] = df["High"] - df["Low"]
    df["norm_pressure"] = np.where(
        df["bar_range"] != 0, df["pressure"] / df["bar_range"], 0.0
        )

    # ----- Delta -----
    df["delta"] = df["norm_pressure"] * df["Volume"]

    # ----- Средние -----

    df["cvd_avg"] = ta.alma(
        df["delta"], length=50, sigma=0.85, distribution_offset=4
        )
    df["price_avg"] = ta.alma(
        df["Close"], length=34, sigma=0.85, distribution_offset=4
    )

    # ----- Наклоны (сырые) -----

    cvd_linreg_curr = ta.linreg(df["cvd_avg"], length=8, offset=0).squeeze()
    cvd_linreg_prev = ta.linreg(df["cvd_avg"], length=8, offset=1).squeeze()
    df["cvd_slope_raw"] = cvd_linreg_curr - cvd_linreg_prev

    price_linreg_curr = ta.linreg(df["price_avg"], length=8, offset=0).squeeze()
    price_linreg_prev = ta.linreg(df["price_avg"], length=8, offset=1).squeeze()
    df["price_slope_raw"] = price_linreg_curr - price_linreg_prev

    # ----- Нормализация -----
    tr_series = ta.true_range(
        df["High"], df["Low"], df["Close"]
        )
    alma_atr = ta.alma(
        tr_series, length=300, sigma=0.85, distribution_offset=4
        )
    alma_vol = ta.alma(
        df["Volume"], length=300, sigma=0.85, distribution_offset=4
        )

    # 3. Нормирование (обработка деления на ноль через np.where)
    df["price_slope"] = np.where(
        alma_atr != 0, 
        df["price_slope_raw"] / alma_atr, 
        df["price_slope_raw"]
    )

    df["cvd_slope"] = np.where(
        alma_vol != 0, 
        df["cvd_slope_raw"] / alma_vol, 
        df["cvd_slope_raw"]
    )

    df["slope_div"]=df["price_slope"]*df["cvd_slope"]

    df["bull_div"] = (df["price_slope"] < 0) & (df["cvd_slope"] > 0).astype(float)
    df["bear_div"] = (df["price_slope"] > 0) & (df["cvd_slope"] < 0).astype(float)

    # 1. RSI (уже ограничен от 0 до 100, делим на 100 для нормализации от 0.0 до 1.0)
    df["rsi_norm"] = ta.rsi(df["Close"], length=14) / 100.0
    
    # 2. Положение цены внутри диапазона волатильности (аналог Bollinger Bands %B, но через ATR)
    # Показывает, насколько цена отклонилась от своего среднего значения в масштабе текущей волатильности
    df["price_dist_from_avg"] = np.where(
        alma_atr != 0, 
        (df["Close"] - df["price_avg"]) / alma_atr, 
        0.0
    )
    
    # 3. Относительный объем (Текущий объем делим на средний исторический объем)
    df["relative_volume"] = np.where(
        alma_vol != 0,
        df["Volume"] / alma_vol,
        1.0
    )
    
    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see 
    feature_cols = [
        "relative_volume",
        "price_dist_from_avg",
        "rsi_norm",
        "slope_div",
        "cvd_slope",
        "price_slope",
        "bull_div",
        "bear_div",
    ]

    return df, feature_cols
