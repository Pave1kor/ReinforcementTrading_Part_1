import pandas as pd
import numpy as np


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

# ---- Technicals ----

# --- Давление (pressure)
    df["pressure"] = (df["Сlose"] - df["Low"]) - (df["High"] - df["Close"])
    df["barRange"] = df["High"] - df["Low"]
    # normPressure = barRange != 0 ? pressure / barRange : 0
    df["normPressure"] = np.where(df["barRange"] != 0, df["pressure"] / df["barRange"], 0)
    
    # --- Дельта
    df["delta"] = df["normPressure"] * df["volume"]
    
    # --- Среднее значение (EMA)
    # ta.ema(delta, lencvd) -> EMA с периодом lencvd
    df["cvdAverage"] = df["delta"].ewm(span=50, adjust=False).mean()
    df["priceAverage"] = df["Close"].ewm(span=34, adjust=False).mean()
    
    # --- Наклон (ROC - Rate of Change)
    # ta.roc(series, 5) = (series - series[-5]) / series[-5] * 100
    df["cvdSlope"] = df["cvdAverage"].pct_change(periods=5) * 100
    df["priceSlope"] = df["priceAverage"].pct_change(periods=5) * 100
    
    # slopeSum = |priceSlope| + |cvdSlope|
    df["slopeSum"] = df["priceSlope"].abs() + df["cvdSlope"].abs()
    
    baseEps = 1e-10
    # eps = max(baseEps, ta.ema(slopeSum, 50) * 0.01)
    df["ema_slopeSum_50"] = df["slopeSum"].ewm(span=50, adjust=False).mean()
    df["eps"] = np.maximum(baseEps, df["ema_slopeSum_50"] * 0.01)
    
    # slopeDiv = 2 * |priceSlope - cvdSlope| / (slopeSum + eps)
    df["divStrength"] = 2 * (df["priceSlope"] - df["cvdSlope"]).abs() / (df["slopeSum"] + df["eps"])
    
    # --- Дивергенции
    df["bullDiv"] = (df["priceSlope"] < 0) & (df["cvdSlope"] > 0)
    df["bearDiv"] = (df["priceSlope"] > 0) & (df["cvdSlope"] < 0)
    
    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see 
    feature_cols = [
        "divStrength",
        "bullDiv",
        "bearDiv"
    ]

    return df, feature_cols
