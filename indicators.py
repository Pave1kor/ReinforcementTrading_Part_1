import pandas as pd
import numpy as np
import ta


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
    df["norm_pressure"] = np.where(df["bar_range"] != 0, df["pressure"] / df["bar_range"], 0.0)

    # ----- Delta -----
    df["delta"] = df["norm_pressure"] * df["volume"]

    # ----- Средние -----
    df["cvd_avg"] = ta.alma(df["delta"], 100, 0.85, 6)
    df["price_avg"] = ta.alma(df["Close"], 65, 0.85, 6)

    # ----- Наклоны (сырые) -----
    df["cvd_slope_raw"] = ta.linreg(df["cvd_avg"], 8, 0) - ta.linreg(df["cvd_avg"], 8, 1)
    df["price_slope_raw"] = ta.linreg(df["price_avg"], 8, 0) - ta.linreg(df["price_avg"], 8, 1)

    # ----- Нормализация -----
    df["atr"] = ta.atr(300)
    df["avg_vol"] = ta.sma(df["volume"], 300)

    df["price_slope"] = np.where(df["atr"] != 0, df["price_slope_raw"] / df["atr"], df["price_slope_raw"])
    df["cvd_slope"] = np.where(df["avg_vol"] != 0, df["cvd_slope_raw"] / df["avg_vol"], df["cvd_slope_raw"])

    # ----- Дивергенция -----
    df["slope_sum"] = df["price_slope"].abs() + df["cvd_slope"].abs()
    base_eps = 1e-10
    alma_sum = ta.alma(df["slope_sum"], 50, 0.85, 6)
    df["eps"] = np.maximum(base_eps, alma_sum * 0.01)
    df["slope_div"] = 2 * (df["price_slope"] - df["cvd_slope"]).abs() / (df["slope_sum"] + df["eps"])

    df["bull_div"] = (df["price_slope"] < 0) & (df["cvd_slope"] > 0)
    df["bear_div"] = (df["price_slope"] > 0) & (df["cvd_slope"] < 0)
    
    # ----- Уровни силы -----
    df["p90"] = ta.percentile_nearest_rank(df["slope_div"], 300, 90)
    df["p80"] = ta.percentile_nearest_rank(df["slope_div"], 300, 80)
    df["p70"] = ta.percentile_nearest_rank(df["slope_div"], 300, 70)
    df["p60"] = ta.percentile_nearest_rank(df["slope_div"], 300, 60)
    df["p50"] = ta.percentile_nearest_rank(df["slope_div"], 300, 50)
    df["p40"] = ta.percentile_nearest_rank(df["slope_div"], 300, 40)
    df["p30"] = ta.percentile_nearest_rank(df["slope_div"], 300, 30)
    df["p20"] = ta.percentile_nearest_rank(df["slope_div"], 300, 20)

    conditions = [
    df["divStrength"] > df["p90"],
    df["divStrength"] > df["p80"],
    df["divStrength"] > df["p70"],
    df["divStrength"] > df["p60"],
    df["divStrength"] > df["p50"],
    df["divStrength"] > df["p40"],
    df["divStrength"] > df["p30"],
    df["divStrength"] > df["p20"],
    ]
    choices = [9, 8, 7, 6, 5, 4, 3, 2]

    df["strengthLevel"] = np.select(conditions, choices, default=1)

    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see 
    feature_cols = [
        "slope_div",
        "cvd_slope",
        "price_slope",
        "strengthLevel",
        "bull_div",
        "bear_div",
    ]

    return df, feature_cols
