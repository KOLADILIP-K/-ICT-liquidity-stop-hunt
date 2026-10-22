"""
Tests for Sweep, Reclaim, Breakdown, and Follow-Through Labeling.
"""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from src.liquidity_patterns import LiquidityPattern
from src.labeling import LabelingEngine


class TestLabeling(unittest.TestCase):

    def setUp(self):
        self.engine = LabelingEngine(
            sweep_threshold_atr=0.1,
            max_reclaim_bars=3,
            target_atr_x=1.5,
            invalidation_atr=1.0,
            horizon_bars_m=10
        )

    def test_breakdown_label(self):
        # Construct pattern confirmed at bar 5 with zone_low = 100.0, ATR = 1.0
        # Price penetrates to 99.5 (sweep depth 0.5) and stays below 100.0 for next 5 bars (never reclaims)
        n = 20
        timestamps = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n)]
        
        lows = np.full(n, 102.0)
        lows[6] = 99.5   # Sweeps below 100 by 0.5 ATR
        closes = np.full(n, 102.0)
        closes[6] = 99.6
        closes[7] = 99.4
        closes[8] = 99.2
        closes[9] = 99.0 # Never closes > 100

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": closes + 1.0,
            "low": lows,
            "close": closes,
            "atr": np.full(n, 1.0),
            "relative_volume": np.ones(n)
        })

        pattern = LiquidityPattern(
            pattern_id="TEST_P1",
            pattern_type="double_bottom",
            direction="bullish",
            instrument="XAUUSD",
            creation_bar_index=5,
            confirmation_bar_index=5,
            creation_timestamp=timestamps[5],
            confirmation_timestamp=timestamps[5],
            zone_low=100.0,
            zone_high=100.2,
            neckline_price=103.0
        )

        events = self.engine.generate_events_and_labels(df, [pattern])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "Breakdown")
        self.assertFalse(events[0].reclaimed)

    def test_successful_hunt_label(self):
        # Sweeps to 99.5, reclaims on bar 7 to 100.5, then reaches 100.5 + 1.5 = 102.0 before invalidation
        n = 25
        timestamps = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n)]
        
        lows = np.full(n, 102.0)
        lows[6] = 99.5
        highs = np.full(n, 102.0)
        closes = np.full(n, 102.0)
        closes[6] = 99.6
        closes[7] = 100.5 # Reclaims!
        highs[10] = 103.0 # Reaches target!

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": np.full(n, 1.0),
            "relative_volume": np.ones(n)
        })

        pattern = LiquidityPattern(
            pattern_id="TEST_P2",
            pattern_type="double_bottom",
            direction="bullish",
            instrument="XAUUSD",
            creation_bar_index=5,
            confirmation_bar_index=5,
            creation_timestamp=timestamps[5],
            confirmation_timestamp=timestamps[5],
            zone_low=100.0,
            zone_high=100.2,
            neckline_price=103.0
        )

        events = self.engine.generate_events_and_labels(df, [pattern])
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].reclaimed)
        self.assertEqual(events[0].label, "1")


if __name__ == "__main__":
    unittest.main()
