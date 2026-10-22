"""
Liquidity Patterns Detection Module
Implements Double Bottoms, Triple Bottoms, and Ascending Trendlines with strict no-look-ahead rules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.swing_detection import SwingPoint


@dataclass
class LiquidityPattern:
    """Represents a detected structural liquidity zone or trendline."""
    pattern_id: str
    pattern_type: str                  # 'double_bottom', 'triple_bottom', 'trendline'
    direction: str                     # 'bullish' (stops below) or 'bearish' (stops above)
    instrument: str
    creation_bar_index: int            # Bar when pattern was first formed (e.g. L2 or L3 bar index)
    confirmation_bar_index: int        # Bar when latest swing anchor was confirmed (NO LOOK-AHEAD)
    creation_timestamp: datetime
    confirmation_timestamp: datetime
    
    # Levels and bounds
    zone_low: float
    zone_high: float
    neckline_price: float
    touch_indices: List[int] = field(default_factory=list)
    touch_timestamps: List[datetime] = field(default_factory=list)
    touch_prices: List[float] = field(default_factory=list)
    
    # Metrics
    equality_score: float = 0.0        # |L1 - L2| / ATR
    pattern_height_atr: float = 0.0    # Neckline height / ATR
    bar_separation: int = 0            # Distance between touches
    total_duration_bars: int = 0       # Total span in bars
    touch_count: int = 2
    zone_width_atr: float = 0.0        # (max_low - min_low) / ATR
    
    # Trendline specific
    slope_atr: float = 0.0             # Slope in ATR per bar
    intercept: float = 0.0
    line_quality_r2: float = 1.0       # Goodness of fit
    
    # State tracking
    is_active: bool = True
    invalidation_bar_index: Optional[int] = None
    invalidation_timestamp: Optional[datetime] = None
    
    def get_line_value_at_bar(self, bar_idx: int) -> float:
        """Projects trendline price at a given bar index."""
        if self.pattern_type == "trendline":
            return self.slope_atr * (bar_idx - self.touch_indices[0]) + self.intercept
        return self.zone_low


class LiquidityPatternDetector:
    """Detects double bottoms, triple bottoms, and trendlines."""

    def __init__(
        self,
        tol_atr: float = 0.25,
        min_separation: int = 5,
        max_separation_db: int = 60,
        max_duration_tb: int = 120,
        min_neckline_atr: float = 1.5,
        min_trendline_touches: int = 3,
        min_slope_atr: float = 0.005,
        max_slope_atr: float = 1.5,
        min_line_age: int = 10
    ):
        self.tol_atr = tol_atr
        self.min_separation = min_separation
        self.max_separation_db = max_separation_db
        self.max_duration_tb = max_duration_tb
        self.min_neckline_atr = min_neckline_atr
        self.min_trendline_touches = min_trendline_touches
        self.min_slope_atr = min_slope_atr
        self.max_slope_atr = max_slope_atr
        self.min_line_age = min_line_age

    def detect_all_patterns(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        instrument: str = "XAUUSD"
    ) -> List[LiquidityPattern]:
        """
        Scans OHLCV series and confirmed swings to extract all valid patterns.
        Reclassifies double bottoms to triple bottoms without duplication.
        """
        low_swings = [s for s in swings if s.swing_type == "low"]
        high_swings = [s for s in swings if s.swing_type == "high"]
        
        patterns: List[LiquidityPattern] = []
        double_bottoms: Dict[Tuple[int, int], LiquidityPattern] = {}
        closes = df["close"].values
        atrs = df["atr"].values if "atr" in df.columns else np.full(len(df), 1.0)
        
        # 1. Detect Double Bottoms
        for i in range(len(low_swings)):
            s1 = low_swings[i]
            for j in range(i + 1, len(low_swings)):
                s2 = low_swings[j]
                
                bar_sep = s2.bar_index - s1.bar_index
                if bar_sep < self.min_separation:
                    continue
                if bar_sep > self.max_separation_db:
                    break  # Lows are ordered chronologically; exceeding max separation stops inner loop
                
                # Check equal low tolerance: |L1 - L2| <= tol * ATR
                avg_atr = max(0.5 * (s1.atr_at_confirmation + s2.atr_at_confirmation), 1e-4)
                price_diff = abs(s1.price - s2.price)
                if price_diff > self.tol_atr * avg_atr:
                    continue
                
                # Check intervening neckline high
                neckline_candidates = [
                    h for h in high_swings 
                    if s1.bar_index < h.bar_index < s2.bar_index
                ]
                if not neckline_candidates:
                    continue
                neckline_point = max(neckline_candidates, key=lambda h: h.price)
                min_low = min(s1.price, s2.price)
                pattern_height = (neckline_point.price - min_low) / avg_atr
                if pattern_height < self.min_neckline_atr:
                    continue
                
                # Check invalidation: No close below min(L1, L2) between s1 and s2
                intervening_closes = closes[s1.bar_index + 1 : s2.bar_index]
                if len(intervening_closes) > 0 and np.any(intervening_closes < min_low):
                    continue
                
                pattern_id = f"DB_{s1.bar_index}_{s2.bar_index}"
                pattern = LiquidityPattern(
                    pattern_id=pattern_id,
                    pattern_type="double_bottom",
                    direction="bullish",
                    instrument=instrument,
                    creation_bar_index=s2.bar_index,
                    confirmation_bar_index=s2.confirmation_bar_index,
                    creation_timestamp=s2.timestamp,
                    confirmation_timestamp=s2.confirmation_timestamp,
                    zone_low=min_low,
                    zone_high=max(s1.price, s2.price),
                    neckline_price=neckline_point.price,
                    touch_indices=[s1.bar_index, s2.bar_index],
                    touch_timestamps=[s1.timestamp, s2.timestamp],
                    touch_prices=[s1.price, s2.price],
                    equality_score=price_diff / avg_atr,
                    pattern_height_atr=pattern_height,
                    bar_separation=bar_sep,
                    total_duration_bars=bar_sep,
                    touch_count=2,
                    zone_width_atr=price_diff / avg_atr
                )
                double_bottoms[(s1.bar_index, s2.bar_index)] = pattern

        # 2. Detect Triple Bottoms and Re-label
        triple_promoted_db_keys = set()
        triple_bottoms: List[LiquidityPattern] = []

        for (idx1, idx2), db in double_bottoms.items():
            # Find low swings occurring after idx2
            s1_candidates = [s for s in low_swings if s.bar_index == idx1]
            s2_candidates = [s for s in low_swings if s.bar_index == idx2]
            if not s1_candidates or not s2_candidates:
                continue
            s1 = s1_candidates[0]
            s2 = s2_candidates[0]

            s3_candidates = [s for s in low_swings if s.bar_index > idx2]
            for s3 in s3_candidates:
                sep_23 = s3.bar_index - s2.bar_index
                tot_duration = s3.bar_index - s1.bar_index

                if sep_23 < self.min_separation:
                    continue
                if tot_duration > self.max_duration_tb:
                    break

                # Tolerance for all 3 lows: max(L) - min(L) <= tol * ATR
                all_lows = [s1.price, s2.price, s3.price]
                min_low = min(all_lows)
                max_low = max(all_lows)
                avg_atr = max((s1.atr_at_confirmation + s2.atr_at_confirmation + s3.atr_at_confirmation) / 3.0, 1e-4)

                if (max_low - min_low) > self.tol_atr * avg_atr:
                    continue

                # Intervening high between s2 and s3
                neckline2_candidates = [
                    h for h in high_swings 
                    if s2.bar_index < h.bar_index < s3.bar_index
                ]
                if not neckline2_candidates:
                    continue
                neckline2 = max(neckline2_candidates, key=lambda h: h.price)
                if (neckline2.price - min(s2.price, s3.price)) / avg_atr < self.min_neckline_atr:
                    continue

                # Invalidation check: No close below min_low between s2 and s3
                interv_closes = closes[s2.bar_index + 1 : s3.bar_index]
                if len(interv_closes) > 0 and np.any(interv_closes < min_low):
                    continue

                # Valid Triple Bottom! Re-label and record
                triple_promoted_db_keys.add((idx1, idx2))
                tb_pattern = LiquidityPattern(
                    pattern_id=f"TB_{s1.bar_index}_{s2.bar_index}_{s3.bar_index}",
                    pattern_type="triple_bottom",
                    direction="bullish",
                    instrument=instrument,
                    creation_bar_index=s3.bar_index,
                    confirmation_bar_index=s3.confirmation_bar_index,
                    creation_timestamp=s3.timestamp,
                    confirmation_timestamp=s3.confirmation_timestamp,
                    zone_low=min_low,
                    zone_high=max_low,
                    neckline_price=max(db.neckline_price, neckline2.price),
                    touch_indices=[s1.bar_index, s2.bar_index, s3.bar_index],
                    touch_timestamps=[s1.timestamp, s2.timestamp, s3.timestamp],
                    touch_prices=[s1.price, s2.price, s3.price],
                    equality_score=(max_low - min_low) / avg_atr,
                    pattern_height_atr=(max(db.neckline_price, neckline2.price) - min_low) / avg_atr,
                    bar_separation=sep_23,
                    total_duration_bars=tot_duration,
                    touch_count=3,
                    zone_width_atr=(max_low - min_low) / avg_atr
                )
                triple_bottoms.append(tb_pattern)

        # Retain only double bottoms that were not promoted to triple bottoms
        for k, db in double_bottoms.items():
            if k not in triple_promoted_db_keys:
                patterns.append(db)
        patterns.extend(triple_bottoms)

        # 3. Detect Ascending Trendlines
        trendlines = self._detect_trendlines(df, low_swings, instrument)
        patterns.extend(trendlines)

        # Sort all patterns by confirmation_bar_index for strict causal ordering
        patterns.sort(key=lambda p: (p.confirmation_bar_index, p.creation_bar_index))
        return patterns

    def _detect_trendlines(
        self,
        df: pd.DataFrame,
        low_swings: List[SwingPoint],
        instrument: str
    ) -> List[LiquidityPattern]:
        """
        Detects ascending trendlines connecting >= 3 confirmed swing lows.
        First and last are anchors; middle swing lows are within tol of fitted line.
        """
        trendlines: List[LiquidityPattern] = []
        n_swings = len(low_swings)
        if n_swings < self.min_trendline_touches:
            return trendlines

        closes = df["close"].values
        atrs = df["atr"].values if "atr" in df.columns else np.full(len(df), 1.0)

        for i in range(n_swings - 2):
            s_start = low_swings[i]
            for j in range(i + self.min_trendline_touches - 1, min(i + 15, n_swings)):
                s_end = low_swings[j]
                
                dx = s_end.bar_index - s_start.bar_index
                dy = s_end.price - s_start.price

                if dx < self.min_line_age or dy <= 0:
                    continue  # Must be strictly ascending and meet minimum age

                slope = dy / dx
                avg_atr = max(0.5 * (s_start.atr_at_confirmation + s_end.atr_at_confirmation), 1e-4)
                slope_atr = slope / avg_atr

                # Slope filter
                if slope_atr < self.min_slope_atr or slope_atr > self.max_slope_atr:
                    continue

                # Test intermediate swing lows
                intercept = s_start.price
                matching_swings = [s_start]
                residuals = [0.0]

                for k in range(i + 1, j):
                    s_mid = low_swings[k]
                    fitted_price = intercept + slope * (s_mid.bar_index - s_start.bar_index)
                    dist = abs(s_mid.price - fitted_price)
                    if dist <= self.tol_atr * avg_atr:
                        matching_swings.append(s_mid)
                        residuals.append(dist)

                matching_swings.append(s_end)
                residuals.append(0.0)

                if len(matching_swings) < self.min_trendline_touches:
                    continue

                # Invalidation check: No close below the line between anchors
                is_invalid = False
                for b_idx in range(s_start.bar_index + 1, s_end.bar_index):
                    line_val = intercept + slope * (b_idx - s_start.bar_index)
                    if closes[b_idx] < line_val - (0.05 * avg_atr):
                        is_invalid = True
                        break

                if is_invalid:
                    continue

                r2 = max(0.0, 1.0 - (float(np.mean(residuals)) / (avg_atr + 1e-6)))

                pattern_id = f"TL_{s_start.bar_index}_{s_end.bar_index}"
                tl = LiquidityPattern(
                    pattern_id=pattern_id,
                    pattern_type="trendline",
                    direction="bullish",
                    instrument=instrument,
                    creation_bar_index=s_end.bar_index,
                    confirmation_bar_index=s_end.confirmation_bar_index,
                    creation_timestamp=s_end.timestamp,
                    confirmation_timestamp=s_end.confirmation_timestamp,
                    zone_low=s_end.price,
                    zone_high=s_end.price + self.tol_atr * avg_atr,
                    neckline_price=s_end.price + 1.5 * avg_atr,
                    touch_indices=[s.bar_index for s in matching_swings],
                    touch_timestamps=[s.timestamp for s in matching_swings],
                    touch_prices=[s.price for s in matching_swings],
                    equality_score=float(np.mean(residuals)) / avg_atr,
                    pattern_height_atr=1.5,
                    bar_separation=dx // len(matching_swings),
                    total_duration_bars=dx,
                    touch_count=len(matching_swings),
                    zone_width_atr=self.tol_atr,
                    slope_atr=slope,
                    intercept=intercept,
                    line_quality_r2=r2
                )
                trendlines.append(tl)

        return trendlines
