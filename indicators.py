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
    df["norm_pressure"] = np.where(df["bar_range"] != 0, df["pressure"] / df["bar_range"], 0.0)

    # ----- Delta -----
    df["delta"] = df["norm_pressure"] * df["Volume"]

    # ----- Средние -----

    df["cvd_avg"] = ta.alma(df["delta"], 100, 0.85, 6)
    df["price_avg"] = ta.alma(df["Close"], 65, 0.85, 6)

    # ----- Наклоны (сырые) -----
    df["cvd_slope_raw"] = ta.linreg(df["cvd_avg"], 8, 0) - ta.linreg(df["cvd_avg"], 8, 1)
    df["price_slope_raw"] = ta.linreg(df["price_avg"], 8, 0) - ta.linreg(df["price_avg"], 8, 1)

    # ----- Нормализация -----
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=300)
    df["avg_vol"] = ta.sma(df["Volume"], length=300)

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
    # Вычисляем скользящие перцентили через встроенный метод pandas
    for p in range(20, 100, 10):
        df[f"p{p}"] = (
            df["slope_div"]
            .rolling(window=300)
            .quantile(p / 100, interpolation="nearest")
        )

    # Удаляем NaNs СРАЗУ после оконных функций, чтобы np.select отработал корректно
    df.dropna(inplace=True)

    # Формируем списки условий и значений для np.select
    conditions = [
        df["slope_div"] > df["p90"],
        df["slope_div"] > df["p80"],
        df["slope_div"] > df["p70"],
        df["slope_div"] > df["p60"],
        df["slope_div"] > df["p50"],
        df["slope_div"] > df["p40"],
        df["slope_div"] > df["p30"],
        df["slope_div"] > df["p20"],
    ]
    choices = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    # Присваиваем уровни силы (если ни одно условие не выполнено, ставится 1)
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
