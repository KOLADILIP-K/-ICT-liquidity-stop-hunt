"""
Feature Engineering Pipeline with Strict Leakage Safeguards
Constructs tabular ML features strictly available at the prediction timestamp.
Excludes all future variables and post-event outcome measures.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.liquidity_patterns import LiquidityPattern
from src.labeling import SweepEvent
from src.session_liquidity import SessionLiquidityManager


class FeatureEngineer:
    """
    Builds event-based ML feature datasets.
    
    Guarantees that every feature is derived from data available on or before the event timestamp.
    Reclaim speed and follow-through moves are strictly excluded.
    """

    FEATURE_COLUMNS = [
        "instrument_code",              # 1 for XAUUSD, 0 for XAGUSD
        "pattern_type_code",            # 0: DB, 1: TB, 2: TL
        "equality_score",               # |L1 - L2| / ATR
        "pattern_height_atr",           # Neckline height / ATR
        "touch_count",                  # Number of confirmed touches
        "zone_width_atr",               # (max_low - min_low) / ATR
        "bar_separation",               # Separation between touches
        "slope_atr",                    # Trendline slope (or 0 for horizontal)
        "line_age",                     # Bars from pattern start to sweep
        "sweep_depth_atr",              # Penetration depth / ATR
        "relative_volume_sweep",        # Volume / SMA(volume, 20) on sweep bar
        "candle_lower_wick_ratio",      # Lower rejection wick / range
        "candle_body_ratio",            # Candle body / range
        "candle_close_location",        # (Close - Low) / Range on sweep bar
        "dist_to_pdl_atr",              # Distance to Prior Day Low in ATR
        "dist_to_pdh_atr",              # Distance to Prior Day High in ATR
        "confluence_score",             # Count of confluent session / daily levels
        "session_code",                 # 0: Asia, 1: London, 2: NYAM, 3: NYPM, 4: Other
        "rolling_volatility",           # 20-period return volatility
        "return_5",                     # 5-bar return prior to sweep
        "return_20"                     # 20-bar return prior to sweep
    ]

    def __init__(self, session_manager: Optional[SessionLiquidityManager] = None):
        self.session_manager = session_manager or SessionLiquidityManager()

    def build_feature_row(
        self,
        event: SweepEvent,
        pattern: LiquidityPattern,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Builds a single feature dictionary for an event using bar information up to event.sweep_bar_index.
        """
        b = event.sweep_bar_index
        row_b = df.iloc[b]
        atr = max(row_b.get("atr", 1.0), 1e-4)

        # Instrument encoding
        inst_code = 1 if "XAU" in str(event.instrument).upper() else 0

        # Pattern type encoding
        p_type_map = {"double_bottom": 0, "triple_bottom": 1, "trendline": 2}
        p_type_code = p_type_map.get(event.pattern_type, 0)

        # Confluence
        confluence_count, _ = self.session_manager.calculate_confluence(
            zone_price=event.level_price,
            current_bar_row=row_b,
            atr=atr
        )

        # Session code
        sess_code = 4  # Other
        if row_b.get("Asia_in_session", False):
            sess_code = 0
        elif row_b.get("London_in_session", False):
            sess_code = 1
        elif row_b.get("NYAM_in_session", False):
            sess_code = 2
        elif row_b.get("NYPM_in_session", False):
            sess_code = 3

        # Line age: bars from pattern creation touch 0 to sweep bar
        t0 = pattern.touch_indices[0] if pattern.touch_indices else pattern.creation_bar_index
        line_age = max(0, b - t0)

        feature_dict = {
            # Identifiers & Metadata (not fed directly to classifier)
            "event_id": event.event_id,
            "instrument": event.instrument,
            "timestamp": event.sweep_timestamp,
            "pattern_id": pattern.pattern_id,
            "pattern_type": event.pattern_type,
            "direction": event.direction,
            "level_price": event.level_price,
            "sweep_bar_index": event.sweep_bar_index,
            
            # Pre-event features
            "instrument_code": inst_code,
            "pattern_type_code": p_type_code,
            "equality_score": float(pattern.equality_score),
            "pattern_height_atr": float(pattern.pattern_height_atr),
            "touch_count": int(pattern.touch_count),
            "zone_width_atr": float(pattern.zone_width_atr),
            "bar_separation": int(pattern.bar_separation),
            "slope_atr": float(pattern.slope_atr),
            "line_age": int(line_age),
            "sweep_depth_atr": float(event.sweep_depth_atr),
            "relative_volume_sweep": float(event.relative_volume_sweep),
            "candle_lower_wick_ratio": float(row_b.get("lower_wick_ratio", 0.0)),
            "candle_body_ratio": float(row_b.get("body_ratio", 0.0)),
            "candle_close_location": float(row_b.get("close_location", 0.5)),
            "dist_to_pdl_atr": float(row_b.get("dist_to_pdl_atr", 0.0)),
            "dist_to_pdh_atr": float(row_b.get("dist_to_pdh_atr", 0.0)),
            "confluence_score": int(confluence_count),
            "session_code": int(sess_code),
            "rolling_volatility": float(row_b.get("rolling_volatility", 0.0)),
            "return_5": float(row_b.get("return_5", 0.0)),
            "return_20": float(row_b.get("return_20", 0.0)),
            
            # Outcome labels (ONLY for training / evaluation, never for prediction)
            "label": event.label,
            "reclaimed": event.reclaimed,
            "reclaim_speed_bars": event.reclaim_speed_bars if event.reclaim_speed_bars is not None else -1,
            "max_favorable_atr": event.max_favorable_atr,
            "max_adverse_atr": event.max_adverse_atr
        }

        return feature_dict

    def create_feature_dataset(
        self,
        df: pd.DataFrame,
        patterns: List[LiquidityPattern],
        events: List[SweepEvent]
    ) -> pd.DataFrame:
        """
        Builds the complete tabular DataFrame of features and labels for all events.
        """
        pattern_dict = {p.pattern_id: p for p in patterns}
        rows = []

        for ev in events:
            pat = pattern_dict.get(ev.pattern_id)
            if pat is None:
                continue
            f_row = self.build_feature_row(ev, pat, df)
            rows.append(f_row)

        if not rows:
            return pd.DataFrame(columns=self.FEATURE_COLUMNS + ["label", "event_id"])

        feature_df = pd.DataFrame(rows)
        return feature_df
