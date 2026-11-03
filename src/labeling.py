"""
Sweep, Reclaim, and Stop-Hunt Labeling Engine
Labels historical candidate events into qualifying hunts (1), non-hunts (0), breakdowns, or unresolved.
Strictly separates pre-event decision features from post-event outcome labels.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.liquidity_patterns import LiquidityPattern


@dataclass
class SweepEvent:
    """Represents a candidate sweep and its historical outcome."""
    event_id: str
    instrument: str
    pattern_id: str
    pattern_type: str                  # 'double_bottom', 'triple_bottom', 'trendline'
    direction: str                     # 'bullish' (sweep below) or 'bearish' (sweep above)
    sweep_bar_index: int
    sweep_timestamp: datetime
    level_price: float
    sweep_price_extreme: float
    sweep_depth_atr: float
    relative_volume_sweep: float
    
    # Reclaim outcome
    reclaimed: bool
    reclaim_bar_index: Optional[int]
    reclaim_timestamp: Optional[datetime]
    reclaim_speed_bars: Optional[int]  # OUTCOME VARIABLE ONLY
    
    # Target follow-through outcome
    max_favorable_atr: float           # OUTCOME VARIABLE ONLY
    max_adverse_atr: float             # OUTCOME VARIABLE ONLY
    label: str                         # '1' (hunt), '0' (non-hunt), 'Breakdown', 'Unresolved'
    target_reached: bool = False
    invalidation_reached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "instrument": self.instrument,
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "direction": self.direction,
            "sweep_bar_index": self.sweep_bar_index,
            "sweep_timestamp": str(self.sweep_timestamp),
            "level_price": round(self.level_price, 4),
            "sweep_price_extreme": round(self.sweep_price_extreme, 4),
            "sweep_depth_atr": round(self.sweep_depth_atr, 4),
            "relative_volume_sweep": round(self.relative_volume_sweep, 2),
            "reclaimed": self.reclaimed,
            "reclaim_bar_index": self.reclaim_bar_index,
            "reclaim_speed_bars": self.reclaim_speed_bars,
            "max_favorable_atr": round(self.max_favorable_atr, 4),
            "max_adverse_atr": round(self.max_adverse_atr, 4),
            "label": self.label
        }


class LabelingEngine:
    """
    Evaluates sweep and reclaim outcomes on historical bars.
    
    Target:
      Price sweeps beyond level by >= sweep_threshold * ATR.
      Price reclaims back across level within max_reclaim_bars.
      If no reclaim within max_reclaim_bars -> 'Breakdown'.
      If reclaimed:
        Does price reach +target_atr_x before hitting -invalidation_atr within horizon_bars_m?
        Yes -> 1 (Stop Hunt)
        No -> 0 (Failed Reversal / Non-Hunt)
        If horizon exceeds available data -> 'Unresolved' (NOT converted to 0!)
    """

    def __init__(
        self,
        sweep_threshold_atr: float = 0.1,
        max_reclaim_bars: int = 3,
        target_atr_x: float = 1.5,
        invalidation_atr: float = 1.0,
        horizon_bars_m: int = 16
    ):
        self.sweep_threshold_atr = sweep_threshold_atr
        self.max_reclaim_bars = max_reclaim_bars
        self.target_atr_x = target_atr_x
        self.invalidation_atr = invalidation_atr
        self.horizon_bars_m = horizon_bars_m

    def generate_events_and_labels(
        self,
        df: pd.DataFrame,
        patterns: List[LiquidityPattern]
    ) -> List[SweepEvent]:
        """
        Scans OHLCV series for sweeps of confirmed patterns and evaluates historical outcomes.
        """
        events: List[SweepEvent] = []
        n = len(df)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values
        atrs = df["atr"].values if "atr" in df.columns else np.full(n, 1.0)
        rvols = df["relative_volume"].values if "relative_volume" in df.columns else np.ones(n)

        # Map patterns by confirmation index so we only check bars AFTER confirmation
        for pattern in patterns:
            # We evaluate bars from pattern.confirmation_bar_index onward
            # Limit search to 100 bars post-confirmation to maintain zone relevance
            start_bar = pattern.confirmation_bar_index
            end_bar = min(n, start_bar + 100)

            for b in range(start_bar, end_bar):
                atr_b = max(atrs[b], 1e-4)
                level_price = pattern.get_line_value_at_bar(b)

                # Long side: check if bar b low penetrates below level_price by >= sweep_threshold * ATR
                penetration = level_price - lows[b]
                if penetration >= self.sweep_threshold_atr * atr_b:
                    # Potential sweep event at bar b!
                    sweep_depth = penetration / atr_b

                    # Check reclaim within max_reclaim_bars:
                    # Does any close in b .. min(n-1, b + max_reclaim_bars) close > level_price?
                    reclaimed = False
                    reclaim_bar = None
                    reclaim_speed = None

                    for r in range(b, min(n, b + self.max_reclaim_bars + 1)):
                        if closes[r] > level_price:
                            reclaimed = True
                            reclaim_bar = r
                            reclaim_speed = r - b
                            break

                    event_id = f"EV_{pattern.pattern_id}_{b}"

                    if not reclaimed:
                        # True Breakdown
                        events.append(SweepEvent(
                            event_id=event_id,
                            instrument=pattern.instrument,
                            pattern_id=pattern.pattern_id,
                            pattern_type=pattern.pattern_type,
                            direction=pattern.direction,
                            sweep_bar_index=b,
                            sweep_timestamp=pd.Timestamp(timestamps[b]).to_pydatetime(),
                            level_price=level_price,
                            sweep_price_extreme=float(lows[b]),
                            sweep_depth_atr=sweep_depth,
                            relative_volume_sweep=float(rvols[b]),
                            reclaimed=False,
                            reclaim_bar_index=None,
                            reclaim_timestamp=None,
                            reclaim_speed_bars=None,
                            max_favorable_atr=0.0,
                            max_adverse_atr=0.0,
                            label="Breakdown"
                        ))
                        # After a breakdown, this pattern is broken
                        break

                    else:
                        # Candidate reclaimed! Evaluate outcome over horizon_bars_m starting from reclaim_bar
                        eval_start = reclaim_bar
                        eval_end = eval_start + self.horizon_bars_m

                        if eval_end >= n:
                            # Insufficient future bars to observe full horizon
                            events.append(SweepEvent(
                                event_id=event_id,
                                instrument=pattern.instrument,
                                pattern_id=pattern.pattern_id,
                                pattern_type=pattern.pattern_type,
                                direction=pattern.direction,
                                sweep_bar_index=b,
                                sweep_timestamp=pd.Timestamp(timestamps[b]).to_pydatetime(),
                                level_price=level_price,
                                sweep_price_extreme=float(lows[b]),
                                sweep_depth_atr=sweep_depth,
                                relative_volume_sweep=float(rvols[b]),
                                reclaimed=True,
                                reclaim_bar_index=reclaim_bar,
                                reclaim_timestamp=pd.Timestamp(timestamps[reclaim_bar]).to_pydatetime(),
                                reclaim_speed_bars=reclaim_speed,
                                max_favorable_atr=0.0,
                                max_adverse_atr=0.0,
                                label="Unresolved"
                            ))
                            break

                        entry_price = closes[reclaim_bar]
                        target_dist = self.target_atr_x * atr_b
                        invalidation_dist = self.invalidation_atr * atr_b
                        target_price = entry_price + target_dist
                        invalidation_price = min(level_price, lows[b]) - invalidation_dist

                        max_favorable = 0.0
                        max_adverse = 0.0
                        label = "0"
                        target_hit = False
                        inval_hit = False

                        for h_idx in range(eval_start + 1, eval_end + 1):
                            # Upward move
                            favorable = (highs[h_idx] - entry_price) / atr_b
                            adverse = (entry_price - lows[h_idx]) / atr_b
                            max_favorable = max(max_favorable, favorable)
                            max_adverse = max(max_adverse, adverse)

                            # Invalidation check first (conservative risk evaluation)
                            if lows[h_idx] <= invalidation_price:
                                inval_hit = True
                                label = "0"
                                break

                            if highs[h_idx] >= target_price:
                                target_hit = True
                                label = "1"
                                break

                        events.append(SweepEvent(
                            event_id=event_id,
                            instrument=pattern.instrument,
                            pattern_id=pattern.pattern_id,
                            pattern_type=pattern.pattern_type,
                            direction=pattern.direction,
                            sweep_bar_index=b,
                            sweep_timestamp=pd.Timestamp(timestamps[b]).to_pydatetime(),
                            level_price=level_price,
                            sweep_price_extreme=float(lows[b]),
                            sweep_depth_atr=sweep_depth,
                            relative_volume_sweep=float(rvols[b]),
                            reclaimed=True,
                            reclaim_bar_index=reclaim_bar,
                            reclaim_timestamp=pd.Timestamp(timestamps[reclaim_bar]).to_pydatetime(),
                            reclaim_speed_bars=reclaim_speed,
                            max_favorable_atr=max_favorable,
                            max_adverse_atr=max_adverse,
                            label=label,
                            target_reached=target_hit,
                            invalidation_reached=inval_hit
                        ))
                        # Once swept and evaluated, do not evaluate duplicate sweep on immediate next bar
                        break

        # 2. Evaluate HTF Liquidity Points (PDH, PDL, PWH, PWL, PMH, PML) sweeps
        htf_events = self.generate_htf_sweep_events(df, instrument=patterns[0].instrument if patterns else "XAUUSD")
        events.extend(htf_events)

        # Sort events chronologically
        events.sort(key=lambda e: (e.sweep_bar_index, e.event_id))
        return events

    def generate_htf_sweep_events(
        self,
        df: pd.DataFrame,
        instrument: str = "XAUUSD"
    ) -> List[SweepEvent]:
        """
        Detects sweeps of designated Daily, Weekly, and Monthly Highs and Lows (PDF Sections 2, 3, 5).
        - Sell-side sweeps below Lows (PDL, PWL, PML): Bullish reversal candidates
        - Buy-side sweeps above Highs (PDH, PWH, PMH): Bearish reversal candidates
        """
        htf_events: List[SweepEvent] = []
        n = len(df)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values
        atrs = df["atr"].values if "atr" in df.columns else np.full(n, 1.0)
        rvols = df["relative_volume"].values if "relative_volume" in df.columns else np.ones(n)

        # Designated HTF definitions: (code, column_name, pattern_type, direction)
        htf_specs = [
            ("PDL", "pdl", "htf_daily", "bullish"),
            ("PWL", "pwl", "htf_weekly", "bullish"),
            ("PML", "pml", "htf_monthly", "bullish"),
            ("PDH", "pdh", "htf_daily", "bearish"),
            ("PWH", "pwh", "htf_weekly", "bearish"),
            ("PMH", "pmh", "htf_monthly", "bearish"),
        ]

        # Warmup period: skip first 100 bars for HTF stability
        start_b = min(100, n - 1)
        cooldown_bars = 12

        for code, col_name, p_type, direction in htf_specs:
            if col_name not in df.columns:
                continue

            level_vals = df[col_name].values
            last_sweep_bar = -999

            for b in range(start_b, n - self.horizon_bars_m):
                if b - last_sweep_bar < cooldown_bars:
                    continue

                level_price = level_vals[b]
                if np.isnan(level_price):
                    continue

                atr_b = max(atrs[b], 1e-4)

                # Check sweep penetration
                if direction == "bullish":
                    # Sell-side sweep below low
                    penetration = level_price - lows[b]
                    is_swept = (penetration >= self.sweep_threshold_atr * atr_b)
                    extreme_p = float(lows[b])
                else:
                    # Buy-side sweep above high
                    penetration = highs[b] - level_price
                    is_swept = (penetration >= self.sweep_threshold_atr * atr_b)
                    extreme_p = float(highs[b])

                if not is_swept:
                    continue

                sweep_depth = penetration / atr_b
                event_id = f"HTF_{code}_{b}"

                # Check reclaim within max_reclaim_bars
                reclaimed = False
                reclaim_bar = None
                reclaim_speed = None

                for r in range(b, min(n, b + self.max_reclaim_bars + 1)):
                    if direction == "bullish" and closes[r] > level_price:
                        reclaimed = True
                        reclaim_bar = r
                        reclaim_speed = r - b
                        break
                    elif direction == "bearish" and closes[r] < level_price:
                        reclaimed = True
                        reclaim_bar = r
                        reclaim_speed = r - b
                        break

                if not reclaimed:
                    htf_events.append(SweepEvent(
                        event_id=event_id,
                        instrument=instrument,
                        pattern_id=f"{code}_{b}",
                        pattern_type=p_type,
                        direction=direction,
                        sweep_bar_index=b,
                        sweep_timestamp=pd.Timestamp(timestamps[b]).to_pydatetime(),
                        level_price=level_price,
                        sweep_price_extreme=extreme_p,
                        sweep_depth_atr=sweep_depth,
                        relative_volume_sweep=float(rvols[b]),
                        reclaimed=False,
                        reclaim_bar_index=None,
                        reclaim_timestamp=None,
                        reclaim_speed_bars=None,
                        max_favorable_atr=0.0,
                        max_adverse_atr=0.0,
                        label="Breakdown"
                    ))
                    last_sweep_bar = b
                    continue

                # Reclaimed! Evaluate outcome over horizon
                eval_start = reclaim_bar
                eval_end = eval_start + self.horizon_bars_m
                if eval_end >= n:
                    htf_events.append(SweepEvent(
                        event_id=event_id,
                        instrument=instrument,
                        pattern_id=f"{code}_{b}",
                        pattern_type=p_type,
                        direction=direction,
                        sweep_bar_index=b,
                        sweep_timestamp=pd.Timestamp(timestamps[b]).to_pydatetime(),
                        level_price=level_price,
                        sweep_price_extreme=extreme_p,
                        sweep_depth_atr=sweep_depth,
                        relative_volume_sweep=float(rvols[b]),
                        reclaimed=True,
                        reclaim_bar_index=reclaim_bar,
                        reclaim_timestamp=pd.Timestamp(timestamps[reclaim_bar]).to_pydatetime(),
                        reclaim_speed_bars=reclaim_speed,
                        max_favorable_atr=0.0,
                        max_adverse_atr=0.0,
                        label="Unresolved"
                    ))
                    last_sweep_bar = b
                    continue

                entry_price = closes[reclaim_bar]
                target_dist = self.target_atr_x * atr_b
                invalidation_dist = self.invalidation_atr * atr_b

                if direction == "bullish":
                    target_price = entry_price + target_dist
                    invalidation_price = min(level_price, lows[b]) - invalidation_dist
                else:
                    target_price = entry_price - target_dist
                    invalidation_price = max(level_price, highs[b]) + invalidation_dist

                max_favorable = 0.0
                max_adverse = 0.0
                label = "0"
                target_hit = False
                inval_hit = False

                for h_idx in range(eval_start + 1, eval_end + 1):
                    if direction == "bullish":
                        favorable = (highs[h_idx] - entry_price) / atr_b
                        adverse = (entry_price - lows[h_idx]) / atr_b
                        max_favorable = max(max_favorable, favorable)
                        max_adverse = max(max_adverse, adverse)

                        if lows[h_idx] <= invalidation_price:
                            inval_hit = True
                            label = "0"
                            break
                        if highs[h_idx] >= target_price:
                            target_hit = True
                            label = "1"
                            break
                    else:
                        favorable = (entry_price - lows[h_idx]) / atr_b
                        adverse = (highs[h_idx] - entry_price) / atr_b
                        max_favorable = max(max_favorable, favorable)
                        max_adverse = max(max_adverse, adverse)

                        if highs[h_idx] >= invalidation_price:
                            inval_hit = True
                            label = "0"
                            break
                        if lows[h_idx] <= target_price:
                            target_hit = True
                            label = "1"
                            break

                htf_events.append(SweepEvent(
                    event_id=event_id,
                    instrument=instrument,
                    pattern_id=f"{code}_{b}",
                    pattern_type=p_type,
                    direction=direction,
                    sweep_bar_index=b,
                    sweep_timestamp=pd.Timestamp(timestamps[b]).to_pydatetime(),
                    level_price=level_price,
                    sweep_price_extreme=extreme_p,
                    sweep_depth_atr=sweep_depth,
                    relative_volume_sweep=float(rvols[b]),
                    reclaimed=True,
                    reclaim_bar_index=reclaim_bar,
                    reclaim_timestamp=pd.Timestamp(timestamps[reclaim_bar]).to_pydatetime(),
                    reclaim_speed_bars=reclaim_speed,
                    max_favorable_atr=max_favorable,
                    max_adverse_atr=max_adverse,
                    label=label,
                    target_reached=target_hit,
                    invalidation_reached=inval_hit
                ))
                last_sweep_bar = b

        return htf_events
