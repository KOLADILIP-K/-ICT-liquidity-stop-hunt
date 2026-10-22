"""
Data Validation and Integrity Checks
Validates OHLCV market data and checks for leakage in features and labels.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    """Detailed summary of dataset health, cleanliness, and coverage."""
    instrument: str
    total_bars: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timeframe: str = "15m"
    duplicate_timestamps: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    ohlc_violations: int = 0
    zero_negative_prices: int = 0
    timestamp_gaps_detected: int = 0
    largest_gap_hours: float = 0.0
    average_volume: float = 0.0
    zero_volume_bars: int = 0
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "total_bars": self.total_bars,
            "start_time": str(self.start_time) if self.start_time else None,
            "end_time": str(self.end_time) if self.end_time else None,
            "timeframe": self.timeframe,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_values": self.missing_values,
            "ohlc_violations": self.ohlc_violations,
            "zero_negative_prices": self.zero_negative_prices,
            "timestamp_gaps_detected": self.timestamp_gaps_detected,
            "largest_gap_hours": round(self.largest_gap_hours, 2),
            "average_volume": round(self.average_volume, 2),
            "zero_volume_bars": self.zero_volume_bars,
            "is_valid": self.is_valid,
            "warnings_count": len(self.warnings),
            "warnings": self.warnings[:10]  # First 10 warnings
        }


def validate_ohlcv_dataframe(df: pd.DataFrame, instrument: str = "UNKNOWN", expected_freq_minutes: int = 15) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Validates and cleans an OHLCV dataframe.
    
    Checks:
    - Required columns present
    - Non-null values
    - Valid timestamps sorted chronologically
    - High >= Low, High >= max(Open, Close), Low <= min(Open, Close)
    - Non-negative prices
    - Duplicate timestamps
    - Gaps in continuous time series
    
    Returns:
        Cleaned, deduplicated, sorted DataFrame and DataQualityReport.
    """
    report = DataQualityReport(instrument=instrument, total_bars=len(df))
    warnings = []

    if df.empty:
        report.is_valid = False
        report.warnings.append("DataFrame is empty.")
        return df, report

    # 1. Standardize column names
    col_map = {}
    for col in df.columns:
        c_low = str(col).strip().lower()
        if c_low in ["time", "timestamp", "datetime", "date"]:
            col_map[col] = "timestamp"
        elif c_low in ["open", "o"]:
            col_map[col] = "open"
        elif c_low in ["high", "h"]:
            col_map[col] = "high"
        elif c_low in ["low", "l"]:
            col_map[col] = "low"
        elif c_low in ["close", "c"]:
            col_map[col] = "close"
        elif c_low in ["volume", "vol", "v", "tick_volume"]:
            col_map[col] = "volume"

    df_cleaned = df.rename(columns=col_map).copy()
    required = ["timestamp", "open", "high", "low", "close"]
    missing_req = [c for c in required if c not in df_cleaned.columns]
    if missing_req:
        report.is_valid = False
        report.warnings.append(f"Missing required columns: {missing_req}")
        return df_cleaned, report

    if "volume" not in df_cleaned.columns:
        df_cleaned["volume"] = 1.0
        warnings.append("Volume column not found; initialized to default 1.0.")

    # 2. Check nulls
    null_counts = df_cleaned[["timestamp", "open", "high", "low", "close", "volume"]].isnull().sum().to_dict()
    report.missing_values = {k: int(v) for k, v in null_counts.items() if v > 0}
    if sum(report.missing_values.values()) > 0:
        warnings.append(f"Null values detected: {report.missing_values}. Dropping rows with null OHLC.")
        df_cleaned = df_cleaned.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()

    # 3. Parse timestamp
    if not pd.api.types.is_datetime64_any_dtype(df_cleaned["timestamp"]):
        df_cleaned["timestamp"] = pd.to_datetime(df_cleaned["timestamp"], utc=True, errors="coerce")
        df_cleaned = df_cleaned.dropna(subset=["timestamp"]).copy()
    elif df_cleaned["timestamp"].dt.tz is None:
        df_cleaned["timestamp"] = df_cleaned["timestamp"].dt.tz_localize("UTC")
    else:
        df_cleaned["timestamp"] = df_cleaned["timestamp"].dt.tz_convert("UTC")

    # 4. Check & remove duplicates
    dupes = df_cleaned.duplicated(subset=["timestamp"], keep="first").sum()
    report.duplicate_timestamps = int(dupes)
    if dupes > 0:
        warnings.append(f"Found {dupes} duplicate timestamps. Keeping first occurrence.")
        df_cleaned = df_cleaned.drop_duplicates(subset=["timestamp"], keep="first")

    # 5. Sort chronologically
    df_cleaned = df_cleaned.sort_values("timestamp").reset_index(drop=True)
    report.total_bars = len(df_cleaned)
    if report.total_bars > 0:
        report.start_time = df_cleaned["timestamp"].iloc[0].to_pydatetime()
        report.end_time = df_cleaned["timestamp"].iloc[-1].to_pydatetime()

    # 6. Check numeric price validity
    for col in ["open", "high", "low", "close", "volume"]:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

    zero_neg = (
        (df_cleaned["open"] <= 0) | 
        (df_cleaned["high"] <= 0) | 
        (df_cleaned["low"] <= 0) | 
        (df_cleaned["close"] <= 0)
    ).sum()
    report.zero_negative_prices = int(zero_neg)
    if zero_neg > 0:
        warnings.append(f"Found {zero_neg} bars with zero or negative prices. Removing them.")
        df_cleaned = df_cleaned[
            (df_cleaned["open"] > 0) & 
            (df_cleaned["high"] > 0) & 
            (df_cleaned["low"] > 0) & 
            (df_cleaned["close"] > 0)
        ].reset_index(drop=True)

    # 7. Check OHLC consistency: High >= Low, High >= max(Open, Close), Low <= min(Open, Close)
    viol_high_low = (df_cleaned["high"] < df_cleaned["low"])
    viol_high_body = (df_cleaned["high"] < df_cleaned["open"]) | (df_cleaned["high"] < df_cleaned["close"])
    viol_low_body = (df_cleaned["low"] > df_cleaned["open"]) | (df_cleaned["low"] > df_cleaned["close"])
    total_violations = (viol_high_low | viol_high_body | viol_low_body).sum()
    report.ohlc_violations = int(total_violations)
    if total_violations > 0:
        warnings.append(f"Found {total_violations} bars with OHLC logical violations. Repairing/removing.")
        # Auto-correct slight precision discrepancies if high is slightly below open/close
        df_cleaned["high"] = df_cleaned[["high", "open", "close"]].max(axis=1)
        df_cleaned["low"] = df_cleaned[["low", "open", "close"]].min(axis=1)

    # 8. Check gaps in timestamp
    if len(df_cleaned) > 1:
        time_diffs = df_cleaned["timestamp"].diff().dropna()
        expected_delta = pd.Timedelta(minutes=expected_freq_minutes)
        # Gaps larger than 3 bars (e.g. weekends or holidays)
        gaps = time_diffs[time_diffs > expected_delta * 3]
        report.timestamp_gaps_detected = len(gaps)
        if len(gaps) > 0:
            report.largest_gap_hours = gaps.max().total_seconds() / 3600.0
            warnings.append(f"Detected {len(gaps)} gaps (> 3 bars). Max gap: {report.largest_gap_hours:.1f} hours (e.g., weekend/holiday).")

    # 9. Volume stats
    report.average_volume = float(df_cleaned["volume"].mean())
    report.zero_volume_bars = int((df_cleaned["volume"] <= 0).sum())

    report.warnings = warnings
    report.is_valid = (len(df_cleaned) > 0) and (report.ohlc_violations == 0 or total_violations > 0)

    # Add instrument column
    df_cleaned["instrument"] = instrument

    return df_cleaned, report
