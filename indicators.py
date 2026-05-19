# indicators.py
import pandas as pd
import numpy as np
import pandas_ta as ta


def load_and_preprocess_data(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["Time (EET)"], dayfirst=True)
    df.columns = df.columns.str.strip()
    df = df.set_index("Time (EET)")
    df.sort_index(inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----- Pressure & Delta -----
    df["pressure"] = (df["Close"] - df["Low"]) - (df["High"] - df["Close"])
    df["bar_range"] = df["High"] - df["Low"]
    df["norm_pressure"] = np.where(df["bar_range"] != 0, df["pressure"] / df["bar_range"], 0.0)
    df["delta"] = df["norm_pressure"] * df["Volume"]

    # ----- Moving Averages -----
    df["cvd_avg"] = ta.alma(df["delta"], length=50, sigma=0.85, distribution_offset=4)
    df["price_avg"] = ta.alma(df["Close"], length=34, sigma=0.85, distribution_offset=4)

    # ----- Slopes -----
    cvd_linreg_curr = ta.linreg(df["cvd_avg"], length=8, offset=0).iloc[:, 0].to_numpy()
    cvd_linreg_prev = ta.linreg(df["cvd_avg"], length=8, offset=1).iloc[:, 0].to_numpy()
    df["cvd_slope_raw"] = cvd_linreg_curr - cvd_linreg_prev

    price_linreg_curr = ta.linreg(df["price_avg"], length=8, offset=0).iloc[:, 0].to_numpy()
    price_linreg_prev = ta.linreg(df["price_avg"], length=8, offset=1).iloc[:, 0].to_numpy()
    df["price_slope_raw"] = price_linreg_curr - price_linreg_prev

    # ----- Normalization Baselines -----
    tr_series = ta.true_range(df["High"], df["Low"], df["Close"])
    df["alma_atr"] = ta.alma(tr_series, length=300, sigma=0.85, distribution_offset=4)
    df["alma_vol"] = ta.alma(df["Volume"], length=300, sigma=0.85, distribution_offset=4)

    # Защита от Data Leakage: Только ffill(). Если в начале NaN, ставим базовое значение
    df["alma_atr"] = df["alma_atr"].ffill().fillna(0.0001)
    df["alma_vol"] = df["alma_vol"].ffill().fillna(1.0)

    # ----- Normalized Features -----
    df["price_slope"] = np.where(df["alma_atr"] != 0, df["price_slope_raw"] / df["alma_atr"], df["price_slope_raw"])
    df["cvd_slope"] = np.where(df["alma_vol"] != 0, df["cvd_slope_raw"] / df["alma_vol"], df["cvd_slope_raw"])
    df["slope_div"] = df["price_slope"] * df["cvd_slope"]

    df["bull_div"] = ((df["price_slope"] < 0) & (df["cvd_slope"] > 0)).astype(float)
    df["bear_div"] = ((df["price_slope"] > 0) & (df["cvd_slope"] < 0)).astype(float)

    df["rsi_norm"] = ta.rsi(df["Close"], length=14) / 100.0
    df["price_dist_from_avg"] = np.where(df["alma_atr"] != 0, (df["Close"] - df["price_avg"]) / df["alma_atr"], 0.0)
    df["relative_volume"] = np.where(df["alma_vol"] != 0, df["Volume"] / df["alma_vol"], 1.0)

    # Удаляем строки, где индикаторы еще не накопили историческое окно
    df.dropna(inplace=True)

    feature_cols = [
        "alma_atr", "relative_volume", "price_dist_from_avg", 
        "rsi_norm", "slope_div", "cvd_slope", "price_slope", 
        "bull_div", "bear_div"
    ]
    return df, feature_cols
