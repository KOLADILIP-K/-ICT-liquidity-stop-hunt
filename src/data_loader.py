"""
Data Loader and Multi-Instrument Preprocessor
Loads, cleans, aligns, and chronologically splits market data for XAUUSD and XAGUSD.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from src.validation import DataQualityReport, validate_ohlcv_dataframe


class DataLoader:
    """Production data loader with multi-format parsing, timezone handling, and quality assurance."""

    def __init__(self, raw_data_dir: str = "data/raw", config_path: str = "config.yaml"):
        self.raw_data_dir = Path(raw_data_dir)
        self.config = self._load_config(config_path)
        self.quality_reports: Dict[str, DataQualityReport] = {}

    @staticmethod
    def _load_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def load_csv(
        self,
        filepath: Union[str, Path],
        instrument: str,
        provider: str = "generic",
        expected_freq_minutes: int = 15
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Loads and validates a single CSV file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Handle different separators or encodings if needed
        try:
            df = pd.read_csv(filepath)
        except Exception:
            df = pd.read_csv(filepath, sep=None, engine="python")

        df_clean, report = validate_ohlcv_dataframe(
            df=df,
            instrument=instrument,
            expected_freq_minutes=expected_freq_minutes
        )

        df_clean["provider"] = provider
        self.quality_reports[instrument] = report
        return df_clean, report

    def load_instrument(
        self,
        instrument: str,
        filename_pattern: Optional[str] = None
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Finds and loads data for a specified instrument from raw_data_dir.
        """
        if not self.raw_data_dir.exists():
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Search for matching files
        patterns = [
            f"{instrument}_15m.csv",
            f"{instrument}.csv",
            f"*{instrument}*.csv",
            f"*{instrument.lower()}*.csv"
        ] if filename_pattern is None else [filename_pattern]

        found_file = None
        for pat in patterns:
            matches = list(self.raw_data_dir.glob(pat))
            if matches:
                found_file = matches[0]
                break

        if found_file is None:
            raise FileNotFoundError(
                f"No data file found for instrument {instrument} in {self.raw_data_dir}."
            )

        provider = "dukascopy" if "dukascopy" in found_file.name.lower() else "broker"
        return self.load_csv(found_file, instrument=instrument, provider=provider)

    def load_all_instruments(
        self,
        instruments: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Loads all specified instruments (defaults to config: XAUUSD, XAGUSD).
        """
        if instruments is None:
            instruments = self.config.get("general", {}).get("instruments", ["XAUUSD", "XAGUSD"])

        data_dict = {}
        for inst in instruments:
            try:
                df, _ = self.load_instrument(inst)
                data_dict[inst] = df
            except FileNotFoundError as e:
                print(f"Warning: {e}")

        return data_dict

    @staticmethod
    def align_multi_instrument(
        data_dict: Dict[str, pd.DataFrame],
        how: str = "inner"
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Calculates timestamp overlap across multiple instruments.
        If how='inner', filters each instrument to the exact shared timestamps.
        If how='outer' or 'preserve', retains all individual bars while providing the union alignment matrix.
        
        Returns:
            aligned_dict: Dictionary of instruments with aligned indices.
            alignment_summary: DataFrame with timestamp coverage statistics.
        """
        if not data_dict:
            return {}, pd.DataFrame()

        ts_sets = {inst: set(df["timestamp"]) for inst, df in data_dict.items()}
        common_ts = set.intersection(*ts_sets.values()) if ts_sets else set()
        union_ts = set.union(*ts_sets.values()) if ts_sets else set()

        stats = []
        for inst, ts_set in ts_sets.items():
            stats.append({
                "instrument": inst,
                "total_bars": len(ts_set),
                "common_bars": len(common_ts),
                "unmatched_bars": len(ts_set - common_ts),
                "coverage_pct": (len(common_ts) / len(ts_set) * 100.0) if len(ts_set) > 0 else 0.0
            })

        summary_df = pd.DataFrame(stats)

        aligned_dict = {}
        if how == "inner":
            sorted_common = sorted(list(common_ts))
            for inst, df in data_dict.items():
                aligned = df[df["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
                aligned_dict[inst] = aligned
        else:
            # Preserve individual data
            aligned_dict = data_dict

        return aligned_dict, summary_df

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        embargo_bars: int = 16,
        timestamp_col: str = "timestamp"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Strict chronological train / validation / test split with embargo gap.
        
        Embargo prevents look-ahead horizon leakage between adjacent sets:
        Train -> [embargo gap of N bars] -> Val -> [embargo gap of N bars] -> Test
        
        Returns:
            train_df, val_df, test_df
        """
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df_sorted)

        if n == 0:
            return df_sorted, df_sorted, df_sorted

        train_end_idx = int(n * train_ratio)
        val_start_idx = train_end_idx + embargo_bars
        val_end_idx = int(n * (train_ratio + val_ratio))
        test_start_idx = val_end_idx + embargo_bars

        train_df = df_sorted.iloc[:train_end_idx].copy().reset_index(drop=True)
        val_df = df_sorted.iloc[val_start_idx:val_end_idx].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[test_start_idx:].copy().reset_index(drop=True)

        return train_df, val_df, test_df
