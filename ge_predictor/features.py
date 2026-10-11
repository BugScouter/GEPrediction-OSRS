from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame, horizons: List[int] = [1, 3, 6]) -> pd.DataFrame:
    """
    Computes stationary feature engineering & time features for OSRS price timeseries.
    """
    data = df.copy()

    # Handle timestamp
    if "timestamp" in data.columns:
        if pd.api.types.is_numeric_dtype(data["timestamp"]):
            data["datetime"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)
        else:
            data["datetime"] = pd.to_datetime(data["timestamp"], utc=True)
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must contain 'timestamp' column or have a DatetimeIndex.")
    else:
        data["datetime"] = data.index

    # Sort by datetime
    data = data.sort_values("datetime").reset_index(drop=True)

    # Standardize column names if needed
    high_col = "avgHighPrice" if "avgHighPrice" in data.columns else "high"
    low_col = "avgLowPrice" if "avgLowPrice" in data.columns else "low"
    high_vol_col = "highPriceVolume" if "highPriceVolume" in data.columns else "high_vol"
    low_vol_col = "lowPriceVolume" if "lowPriceVolume" in data.columns else "low_vol"

    # Fill missing or 0 prices if any
    data[high_col] = data[high_col].ffill().bfill()
    data[low_col] = data[low_col].ffill().bfill()

    # Calculate mid price
    data["mid_price"] = (data[high_col] + data[low_col]) / 2.0

    # Total volume
    high_v = data[high_vol_col].fillna(0) if high_vol_col in data.columns else 0
    low_v = data[low_vol_col].fillna(0) if low_vol_col in data.columns else 0
    data["total_volume"] = high_v + low_v

    # 1. Log returns
    data["log_return"] = np.log(data["mid_price"] / data["mid_price"].shift(1))

    # 2. EMA Ratios
    ema5 = data["mid_price"].ewm(span=5, adjust=False).mean()
    ema20 = data["mid_price"].ewm(span=20, adjust=False).mean()
    data["ema_ratio_5_20"] = ema5 / ema20

    # 3. Rolling Volatility
    data["volatility_5"] = data["log_return"].rolling(window=5).std()
    data["volatility_20"] = data["log_return"].rolling(window=20).std()

    # 4. Relative High-Low Spread
    data["hl_spread_pct"] = (data[high_col] - data[low_col]) / data["mid_price"].replace(0, np.nan)

    # 5. Volume Momentum
    vol_sma5 = data["total_volume"].rolling(window=5).mean()
    vol_sma20 = data["total_volume"].rolling(window=20).mean()
    data["vol_momentum"] = np.log((vol_sma5 + 1) / (vol_sma20 + 1))

    # 6. Cyclical Time Features
    hours = data["datetime"].dt.hour
    day_of_week = data["datetime"].dt.dayofweek

    data["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    data["day_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    data["day_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    # Forward targets
    for h in horizons:
        data[f"target_forward_return_{h}"] = np.log(data["mid_price"].shift(-h) / data["mid_price"])

    return data

FEATURE_COLUMNS = [
    "log_return",
    "ema_ratio_5_20",
    "volatility_5",
    "volatility_20",
    "hl_spread_pct",
    "vol_momentum",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]
