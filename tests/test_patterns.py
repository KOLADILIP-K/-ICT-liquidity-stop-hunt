"""
Tests for Structural Liquidity Patterns (Double Bottoms, Triple Bottoms, and Trendlines).
"""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from src.swing_detection import SwingPoint
from src.liquidity_patterns import LiquidityPatternDetector


class TestLiquidityPatterns(unittest.TestCase):

    def setUp(self):
        self.detector = LiquidityPatternDetector(
            tol_atr=0.25,
            min_separation=5,
            max_separation_db=60,
            max_duration_tb=120,
            min_neckline_atr=1.5
        )

    def test_double_bottom_detection(self):
        # Construct synthetic dataframe with 2 equal lows at index 10 and 25
        n = 50
        timestamps = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n)]
        lows = np.full(n, 105.0)
        lows[10] = 100.0   # L1
        lows[25] = 100.1   # L2 (diff 0.1 <= 0.25 * ATR)
        
        highs = np.full(n, 107.0)
        highs[17] = 104.0  # Neckline between L1 and L2 (height 4.0 >= 1.5 * ATR)

        closes = np.full(n, 105.0)
        closes[10] = 101.0
        closes[25] = 101.0

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": np.full(n, 2.0)
        })

        swings = [
            SwingPoint(10, 13, timestamps[10], timestamps[13], 100.0, "low", 2.0, 2.0),
            SwingPoint(17, 20, timestamps[17], timestamps[20], 104.0, "high", 2.0, 2.0),
            SwingPoint(25, 28, timestamps[25], timestamps[28], 100.1, "low", 2.0, 2.0),
        ]

        patterns = self.detector.detect_all_patterns(df, swings, instrument="XAUUSD")
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].pattern_type, "double_bottom")
        self.assertEqual(patterns[0].touch_count, 2)
        self.assertAlmostEqual(patterns[0].zone_low, 100.0)

    def test_double_bottom_invalidation_by_close(self):
        # Same setup but close at bar 20 violates support (< 100.0)
        n = 50
        timestamps = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n)]
        lows = np.full(n, 105.0)
        lows[10] = 100.0
        lows[25] = 100.1
        highs = np.full(n, 107.0)
        highs[17] = 104.0
        closes = np.full(n, 105.0)
        closes[20] = 98.5  # Closes BELOW support!

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": np.full(n, 2.0)
        })

        swings = [
            SwingPoint(10, 13, timestamps[10], timestamps[13], 100.0, "low", 2.0, 2.0),
            SwingPoint(17, 20, timestamps[17], timestamps[20], 104.0, "high", 2.0, 2.0),
            SwingPoint(25, 28, timestamps[25], timestamps[28], 100.1, "low", 2.0, 2.0),
        ]

        patterns = self.detector.detect_all_patterns(df, swings, instrument="XAUUSD")
        # Must be rejected due to invalidating close
        self.assertEqual(len(patterns), 0)


if __name__ == "__main__":
    unittest.main()
