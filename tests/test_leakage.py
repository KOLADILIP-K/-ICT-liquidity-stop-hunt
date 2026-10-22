"""
Data Leakage Verification Tests
Proves that mutating future bars does not alter pre-event features or confirmation status.
"""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from src.swing_detection import SwingDetector, SwingPoint
from src.liquidity_patterns import LiquidityPattern
from src.labeling import SweepEvent
from src.feature_engineering import FeatureEngineer
from src.session_liquidity import SessionLiquidityManager


class TestDataLeakage(unittest.TestCase):

    def test_feature_invariance_to_future_data_mutation(self):
        # 1. Build synthetic base DataFrame
        n = 60
        timestamps = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n)]
        
        np.random.seed(42)
        base_prices = 2000.0 + np.cumsum(np.random.randn(n) * 1.5)
        highs = base_prices + 2.0
        lows = base_prices - 2.0
        opens = base_prices - 0.5
        closes = base_prices + 0.5

        df1 = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 1000.0),
            "atr": np.full(n, 3.5),
            "lower_wick_ratio": np.full(n, 0.4),
            "body_ratio": np.full(n, 0.3),
            "close_location": np.full(n, 0.7),
            "dist_to_pdl_atr": np.full(n, 0.2),
            "dist_to_pdh_atr": np.full(n, 2.0),
            "rolling_volatility": np.full(n, 0.003),
            "return_5": np.full(n, -0.001),
            "return_20": np.full(n, -0.002),
            "Asia_in_session": np.full(n, False),
            "London_in_session": np.full(n, True),
            "NYAM_in_session": np.full(n, False),
            "NYPM_in_session": np.full(n, False)
        })

        # Pattern confirmed at bar 20
        pattern = LiquidityPattern(
            pattern_id="P_LEAK_TEST",
            pattern_type="double_bottom",
            direction="bullish",
            instrument="XAUUSD",
            creation_bar_index=15,
            confirmation_bar_index=20,
            creation_timestamp=timestamps[15],
            confirmation_timestamp=timestamps[20],
            zone_low=1995.0,
            zone_high=1995.5,
            neckline_price=2005.0,
            touch_indices=[10, 15],
            equality_score=0.14,
            pattern_height_atr=2.8,
            touch_count=2,
            zone_width_atr=0.14,
            bar_separation=5,
            slope_atr=0.0
        )

        event = SweepEvent(
            event_id="EV_LEAK_TEST",
            instrument="XAUUSD",
            pattern_id=pattern.pattern_id,
            pattern_type=pattern.pattern_type,
            direction="bullish",
            sweep_bar_index=25,  # Decision timestamp is bar 25
            sweep_timestamp=timestamps[25],
            level_price=1995.0,
            sweep_price_extreme=1994.0,
            sweep_depth_atr=0.28,
            relative_volume_sweep=1.35,
            reclaimed=True,
            reclaim_bar_index=26,
            reclaim_timestamp=timestamps[26],
            reclaim_speed_bars=1,
            max_favorable_atr=1.8,
            max_adverse_atr=0.4,
            label="1"
        )

        fe = FeatureEngineer()
        f_original = fe.build_feature_row(event, pattern, df1)

        # 2. Mutate ALL future data (bars 26 to 59) with wild random changes
        df2 = df1.copy()
        df2.loc[26:, "high"] = df2.loc[26:, "high"] * 2.5 + 500.0
        df2.loc[26:, "low"] = df2.loc[26:, "low"] * 0.5 - 200.0
        df2.loc[26:, "close"] = df2.loc[26:, "close"] * 1.8
        df2.loc[26:, "volume"] = 999999.0
        df2.loc[26:, "return_5"] = 0.50

        f_mutated = fe.build_feature_row(event, pattern, df2)

        # 3. Assert every single pre-event feature is mathematically identical!
        for col in FeatureEngineer.FEATURE_COLUMNS:
            val_orig = f_original[col]
            val_mut = f_mutated[col]
            self.assertEqual(
                val_orig, val_mut,
                f"Leakage detected in feature '{col}': original={val_orig}, mutated_future={val_mut}"
            )


if __name__ == "__main__":
    unittest.main()
