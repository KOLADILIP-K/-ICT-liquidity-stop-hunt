"""
Tests for Core Indicators and Feature Engineering Ratios.
"""

import unittest
import numpy as np
import pandas as pd
from src.indicators import compute_atr, compute_candle_features, compute_relative_volume


class TestFeatures(unittest.TestCase):

    def test_atr_calculation(self):
        # 3 bars with constant range 2.0
        high = pd.Series([102.0, 102.0, 102.0, 102.0, 102.0])
        low = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
        close = pd.Series([101.0, 101.0, 101.0, 101.0, 101.0])

        atr = compute_atr(high, low, close, period=3)
        self.assertAlmostEqual(atr.iloc[-1], 2.0, places=2)

    def test_candle_anatomy_ratios_bounded(self):
        df = pd.DataFrame({
            "open": [100.0, 105.0],
            "high": [110.0, 110.0],
            "low": [95.0, 95.0],
            "close": [108.0, 97.0]
        })
        cf = compute_candle_features(df)

        for col in ["upper_wick_ratio", "lower_wick_ratio", "body_ratio", "close_location"]:
            self.assertTrue((cf[col] >= 0.0).all())
            self.assertTrue((cf[col] <= 1.0).all())

    def test_relative_volume(self):
        vol = pd.Series([100.0, 100.0, 100.0, 100.0, 200.0])
        rvol = compute_relative_volume(vol, window=4)
        # On bar 4, previous mean is 100, current is 200 -> rvol should be around 1.6-2.0
        self.assertGreater(rvol.iloc[-1], 1.2)


if __name__ == "__main__":
    unittest.main()
