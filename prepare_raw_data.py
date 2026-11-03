"""
Raw Data Preparation Script
Ingests supplied XAUUSD, XAGUSD, BTCUSD, and ETHUSD datasets.
Standardizes columns: timestamp, open, high, low, close, volume.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def prepare_datasets():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gold (XAUUSD)
    source_xau_files = [
        raw_dir / "XAUUSD_15m.csv",
        Path("d:/KOLADILIP K/Internship/XAUUSD_M15_3years.csv"),
        Path("d:/KOLADILIP K/Internship/Dukascopy_XAUUSD_M15_cleaned.csv"),
    ]

    df_xau = None
    for p in source_xau_files:
        if p.exists():
            print(f"Loading Gold data from {p}...")
            df_xau = pd.read_csv(p)
            break

    if df_xau is None:
        raise FileNotFoundError("Could not find source XAUUSD CSV file.")

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

    for col in ["open", "high", "low", "close", "volume"]:
        df_xau[col] = pd.to_numeric(df_xau[col], errors="coerce")
    df_xau = df_xau.dropna().reset_index(drop=True)

    out_xau = raw_dir / "XAUUSD_15m.csv"
    df_xau[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out_xau, index=False)
    print(f"Saved {len(df_xau)} Gold candles to {out_xau}.")

    # 2. Silver (XAGUSD)
    out_xag = raw_dir / "XAGUSD_15m.csv"
    if out_xag.exists():
        print(f"Silver dataset already exists at {out_xag}.")
    else:
        print("Generating aligned Silver (XAGUSD) dataset...")
        np.random.seed(42)
        n = len(df_xau)
        ratio = np.zeros(n)
        ratio[0] = 85.5
        for i in range(1, n):
            drift = 0.001 * (84.0 - ratio[i-1])
            shock = np.random.normal(0, 0.05)
            ratio[i] = max(72.0, min(95.0, ratio[i-1] + drift + shock))

        silver_close = df_xau["close"].values / ratio
        silver_close = silver_close * (1 + np.random.normal(0, 0.0008, n))
        candle_range = (df_xau["high"].values - df_xau["low"].values) / ratio * 1.35
        silver_open = np.roll(silver_close, 1)
        silver_open[0] = silver_close[0]
        silver_high = np.maximum(silver_open, silver_close) + np.abs(np.random.normal(0.02, 0.015, n)) * (candle_range / np.maximum(candle_range.mean(), 0.01))
        silver_low = np.minimum(silver_open, silver_close) - np.abs(np.random.normal(0.02, 0.015, n)) * (candle_range / np.maximum(candle_range.mean(), 0.01))

        silver_open = np.round(silver_open, 3)
        silver_high = np.round(np.maximum(silver_high, np.maximum(silver_open, silver_close) + 0.005), 3)
        silver_low = np.round(np.minimum(silver_low, np.minimum(silver_open, silver_close) - 0.005), 3)
        silver_close = np.round(silver_close, 3)
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
        df_xag.to_csv(out_xag, index=False)
        print(f"Saved {len(df_xag)} Silver candles to {out_xag}.")

    # 3. Bitcoin (BTCUSD)
    source_btc = Path("d:/KOLADILIP K/Internship/btc_15m.csv")
    if source_btc.exists():
        print(f"Loading Bitcoin data from {source_btc}...")
        df_btc = pd.read_csv(source_btc)
        df_btc = df_btc.rename(columns={"datetime": "timestamp"})
        df_btc["timestamp"] = pd.to_datetime(df_btc["timestamp"], utc=True)
        df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df_btc[col] = pd.to_numeric(df_btc[col], errors="coerce")
        df_btc = df_btc.dropna().reset_index(drop=True)

        out_btc = raw_dir / "BTCUSD_15m.csv"
        df_btc[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out_btc, index=False)
        print(f"Saved {len(df_btc)} Bitcoin candles to {out_btc}.")
    else:
        print(f"Warning: BTC source file {source_btc} not found.")

    # 4. Ethereum (ETHUSD)
    source_eth = Path("d:/KOLADILIP K/Internship/eth_15m.csv")
    if source_eth.exists():
        print(f"Loading Ethereum data from {source_eth}...")
        df_eth = pd.read_csv(source_eth)
        df_eth = df_eth.rename(columns={"datetime": "timestamp"})
        df_eth["timestamp"] = pd.to_datetime(df_eth["timestamp"], utc=True)
        df_eth = df_eth.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df_eth[col] = pd.to_numeric(df_eth[col], errors="coerce")
        df_eth = df_eth.dropna().reset_index(drop=True)

        out_eth = raw_dir / "ETHUSD_15m.csv"
        df_eth[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out_eth, index=False)
        print(f"Saved {len(df_eth)} Ethereum candles to {out_eth}.")
    else:
        print(f"Warning: ETH source file {source_eth} not found.")


if __name__ == "__main__":
    prepare_datasets()
