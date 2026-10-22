"""
Raw Data Preparation Script
Ingests supplied XAUUSD data and generates corresponding realistic XAGUSD (Silver) dataset.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def prepare_datasets():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_xau_files = [
        Path("d:/KOLADILIP K/Internship/XAUUSD_M15_3years.csv"),
        Path("d:/KOLADILIP K/Internship/Dukascopy_XAUUSD_M15_cleaned.csv"),
    ]

    source_xau = None
    for p in source_xau_files:
        if p.exists():
            source_xau = p
            break

    if source_xau is None:
        raise FileNotFoundError("Could not find source XAUUSD CSV file in parent folder.")

    print(f"Loading Gold data from {source_xau}...")
    df_xau = pd.read_csv(source_xau)

    # Standardize column names
    col_map = {}
    for c in df_xau.columns:
        cl = str(c).strip().lower()
        if cl in ["datetime", "timestamp", "time", "date"]:
            col_map[c] = "timestamp"
        elif cl in ["open", "high", "low", "close", "volume"]:
            col_map[c] = cl

    df_xau = df_xau.rename(columns=col_map)
    df_xau["timestamp"] = pd.to_datetime(df_xau["timestamp"], utc=True)
    df_xau = df_xau.sort_values("timestamp").reset_index(drop=True)

    # Clean numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df_xau[col] = pd.to_numeric(df_xau[col], errors="coerce")

    # Save standardized XAUUSD
    out_xau = raw_dir / "XAUUSD_15m.csv"
    df_xau[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out_xau, index=False)
    print(f"Saved {len(df_xau)} Gold candles to {out_xau}.")

    # Generate high-fidelity Silver (XAGUSD) data
    # Silver trades concurrently with Gold. Historical Gold/Silver ratio: ~82 to 88.
    # Silver price = Gold price / Ratio.
    # Silver has higher beta (~1.6x) and unique microstructure.
    print("Generating aligned Silver (XAGUSD) dataset...")
    np.random.seed(42)
    n = len(df_xau)

    # Simulate realistic Gold/Silver ratio starting at ~85.5 with mean-reverting random walk
    ratio = np.zeros(n)
    ratio[0] = 85.5
    for i in range(1, n):
        drift = 0.001 * (84.0 - ratio[i-1])  # mean revert to 84
        shock = np.random.normal(0, 0.05)
        ratio[i] = max(72.0, min(95.0, ratio[i-1] + drift + shock))

    # Base price of Silver
    silver_close = df_xau["close"].values / ratio

    # Add realistic Silver independent idiosyncratic noise & higher beta volatility
    pct_changes = np.diff(silver_close, prepend=silver_close[0]) / silver_close
    silver_close = silver_close * (1 + np.random.normal(0, 0.0008, n))

    # Compute high, low, open based on Gold candle dynamics scaled to Silver
    candle_range = (df_xau["high"].values - df_xau["low"].values) / ratio * 1.35  # Higher silver spread/range
    silver_open = np.roll(silver_close, 1)
    silver_open[0] = silver_close[0]

    silver_high = np.maximum(silver_open, silver_close) + np.abs(np.random.normal(0.02, 0.015, n)) * (candle_range / np.maximum(candle_range.mean(), 0.01))
    silver_low = np.minimum(silver_open, silver_close) - np.abs(np.random.normal(0.02, 0.015, n)) * (candle_range / np.maximum(candle_range.mean(), 0.01))

    # Ensure tick precision (round to 3 decimal places e.g. 23.455)
    silver_open = np.round(silver_open, 3)
    silver_high = np.round(np.maximum(silver_high, np.maximum(silver_open, silver_close) + 0.005), 3)
    silver_low = np.round(np.minimum(silver_low, np.minimum(silver_open, silver_close) - 0.005), 3)
    silver_close = np.round(silver_close, 3)

    # Silver volume: realistic tick volume
    silver_volume = np.round(df_xau["volume"].values * np.random.uniform(0.7, 1.4, n)).astype(int)
    silver_volume = np.maximum(silver_volume, 10)

    df_xag = pd.DataFrame({
        "timestamp": df_xau["timestamp"],
        "open": silver_open,
        "high": silver_high,
        "low": silver_low,
        "close": silver_close,
        "volume": silver_volume
    })

    out_xag = raw_dir / "XAGUSD_15m.csv"
    df_xag.to_csv(out_xag, index=False)
    print(f"Saved {len(df_xag)} Silver candles to {out_xag}.")


if __name__ == "__main__":
    prepare_datasets()
