"""
Swing Point Detection Module with Strict No-Look-Ahead Confirmation
Detects confirmed swing highs and lows.
A swing occurring at bar i is strictly confirmed and accessible only at bar i + k.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    """Represents a confirmed swing high or swing low."""
    bar_index: int                       # Index where the physical swing occurred
    confirmation_bar_index: int          # Index where the swing became confirmed (bar_index + k)
    timestamp: datetime                  # Timestamp when the swing occurred
    confirmation_timestamp: datetime     # Timestamp when the swing was confirmed
    price: float                         # Price of the swing high/low
    swing_type: str                      # 'high' or 'low'
    atr_at_swing: float                  # ATR at the swing bar
    atr_at_confirmation: float           # ATR at the confirmation bar


class SwingDetector:
    """
    Detects swing highs and swing lows with configurable lookback k.
    
    Zero-look-ahead guarantee:
    A swing at bar i requires:
      low[i] <= low[i - j] for j in 1..k
      low[i] < low[i + j] for j in 1..k
    This condition can only be evaluated at bar i + k!
    """

    def __init__(self, k: int = 3):
        self.k = k

    def detect_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Scans OHLCV DataFrame and returns chronologically ordered confirmed swing points.
        """
        swings: List[SwingPoint] = []
        n = len(df)
        k = self.k

        if n < 2 * k + 1:
            return swings

        highs = df["high"].values
        lows = df["low"].values
        timestamps = df["timestamp"].values
        atrs = df["atr"].values if "atr" in df.columns else np.full(n, 1.0)

        for i in range(k, n - k):
            # Check swing low
            is_swing_low = True
            for j in range(1, k + 1):
                if lows[i] > lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                conf_idx = i + k
                swings.append(SwingPoint(
                    bar_index=i,
                    confirmation_bar_index=conf_idx,
                    timestamp=pd.Timestamp(timestamps[i]).to_pydatetime(),
                    confirmation_timestamp=pd.Timestamp(timestamps[conf_idx]).to_pydatetime(),
                    price=float(lows[i]),
                    swing_type="low",
                    atr_at_swing=float(atrs[i]),
                    atr_at_confirmation=float(atrs[conf_idx])
                ))

            # Check swing high
            is_swing_high = True
            for j in range(1, k + 1):
                if highs[i] < highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                conf_idx = i + k
                swings.append(SwingPoint(
                    bar_index=i,
                    confirmation_bar_index=conf_idx,
                    timestamp=pd.Timestamp(timestamps[i]).to_pydatetime(),
                    confirmation_timestamp=pd.Timestamp(timestamps[conf_idx]).to_pydatetime(),
                    price=float(highs[i]),
                    swing_type="high",
                    atr_at_swing=float(atrs[i]),
                    atr_at_confirmation=float(atrs[conf_idx])
                ))

        # Sort by confirmation_bar_index so events become visible chronologically
        swings.sort(key=lambda x: (x.confirmation_bar_index, x.bar_index))
        return swings

    @staticmethod
    def get_swings_visible_at_bar(swings: List[SwingPoint], current_bar_index: int, swing_type: Optional[str] = None) -> List[SwingPoint]:
        """
        Returns only swings confirmed on or before current_bar_index.
        """
        res = [s for s in swings if s.confirmation_bar_index <= current_bar_index]
        if swing_type is not None:
            res = [s for s in res if s.swing_type == swing_type]
        return res
