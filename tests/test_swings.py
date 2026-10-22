"""
Tests for Swing Point Detection and No-Look-Ahead Confirmation.
"""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from src.swing_detection import SwingDetector, SwingPoint


class TestSwingDetection(unittest.TestCase):

    def setUp(self):
        # Create synthetic V-shaped swing low at bar 10 with k=3
        n = 30
        timestamps = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n)]
        
        # Prices descending to bar 10, then ascending
        lows = [100.0 - (i if i <= 10 else (20 - i)) for i in range(n)]
        highs = [l + 2.0 for l in lows]
        opens = [l + 1.0 for l in lows]
        closes = [l + 1.0 for l in lows]

        self.df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": np.full(n, 1.5)
        })

    def test_swing_low_confirmation_lag(self):
        detector = SwingDetector(k=3)
        swings = detector.detect_swings(self.df)

        low_swings = [s for s in swings if s.swing_type == "low"]
        self.assertTrue(len(low_swings) >= 1)

        target_swing = low_swings[0]
        self.assertEqual(target_swing.bar_index, 10)
        # MUST be confirmed at bar 10 + k = 13
        self.assertEqual(target_swing.confirmation_bar_index, 13)

    def test_visibility_at_bar_guarantee(self):
        detector = SwingDetector(k=3)
        swings = detector.detect_swings(self.df)

        # At bar 12 (before confirmation), swing 10 must NOT be visible
        visible_at_12 = SwingDetector.get_swings_visible_at_bar(swings, current_bar_index=12)
        self.assertEqual(len(visible_at_12), 0)

        # At bar 13 (on confirmation), swing 10 MUST be visible
        visible_at_13 = SwingDetector.get_swings_visible_at_bar(swings, current_bar_index=13)
        self.assertEqual(len(visible_at_13), 1)
        self.assertEqual(visible_at_13[0].bar_index, 10)


if __name__ == "__main__":
    unittest.main()
