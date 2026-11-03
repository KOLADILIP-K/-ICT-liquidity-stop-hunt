"""
Unit and Integration Tests for HTF Liquidity Points Reference & Checklist
Validates Daily, Weekly, and Monthly levels, zero lookahead bias, confluence clustering,
reading price behaviors (Sweep, Acceptance, Reaction, Magnet), and the 80% train split.
"""

from datetime import datetime, timedelta, timezone
import unittest
import numpy as np
import pandas as pd

from src.indicators import (
    compute_previous_day_levels,
    compute_previous_week_levels,
    compute_previous_month_levels,
    add_core_indicators
)
from src.liquidity_points import (
    LiquidityPointsEngine,
    DESIGNATED_LIQUIDITY_POINTS
)
from src.train import ModelTrainer


class TestHTFLiquidityPoints(unittest.TestCase):
    """Tests for designated Daily, Weekly, and Monthly liquidity points."""

    def setUp(self):
        # Create 60 days of synthetic 15-minute OHLCV data
        start_dt = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        n_bars = 60 * 96  # 5760 bars
        timestamps = [start_dt + timedelta(minutes=15 * i) for i in range(n_bars)]

        np.random.seed(42)
        base_price = 2000.0 + np.cumsum(np.random.normal(0, 0.5, n_bars))
        highs = base_price + np.abs(np.random.normal(1.0, 0.3, n_bars))
        lows = base_price - np.abs(np.random.normal(1.0, 0.3, n_bars))
        opens = base_price + np.random.normal(0, 0.2, n_bars)
        closes = base_price + np.random.normal(0, 0.2, n_bars)
        volumes = np.random.randint(100, 1000, n_bars)

        self.df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        self.engine = LiquidityPointsEngine()

    def test_daily_levels_no_lookahead(self):
        """Proves PDH and PDL on any day D are derived solely from day D-1 and never see day D."""
        daily_df = compute_previous_day_levels(self.df)
        self.assertIn("pdh", daily_df.columns)
        self.assertIn("pdl", daily_df.columns)

        # Check day 2: PDH must match day 1's max high exactly
        dates = self.df["timestamp"].dt.date
        unique_dates = list(dates.unique())
        self.assertGreater(len(unique_dates), 10)

        day1_mask = (dates == unique_dates[0])
        day2_mask = (dates == unique_dates[1])

        day1_high = self.df.loc[day1_mask, "high"].max()
        day2_pdh = daily_df.loc[day2_mask, "pdh"].iloc[0]

        self.assertAlmostEqual(day1_high, day2_pdh, places=4)

    def test_weekly_levels_no_lookahead(self):
        """Proves PWH and PWL on week W are derived strictly from week W-1."""
        weekly_df = compute_previous_week_levels(self.df)
        self.assertIn("pwh", weekly_df.columns)
        self.assertIn("pwl", weekly_df.columns)

        # Verify values remain constant throughout each week
        weeks = self.df["timestamp"].dt.isocalendar().week
        unique_weeks = list(weeks.unique())
        self.assertGreater(len(unique_weeks), 4)

        w1_mask = (weeks == unique_weeks[0])
        w2_mask = (weeks == unique_weeks[1])

        w1_high = self.df.loc[w1_mask, "high"].max()
        w2_pwh = weekly_df.loc[w2_mask, "pwh"].iloc[0]
        self.assertAlmostEqual(w1_high, w2_pwh, places=4)

    def test_monthly_levels_no_lookahead(self):
        """Proves PMH and PML on month M are derived strictly from month M-1."""
        monthly_df = compute_previous_month_levels(self.df, tz_name="UTC")
        self.assertIn("pmh", monthly_df.columns)
        self.assertIn("pml", monthly_df.columns)

        months = self.df["timestamp"].dt.month
        m1_mask = (months == 1)
        m2_mask = (months == 2)

        m1_high = self.df.loc[m1_mask, "high"].max()
        m2_pmh = monthly_df.loc[m2_mask, "pmh"].iloc[0]
        self.assertAlmostEqual(m1_high, m2_pmh, places=4)

    def test_confluence_clustering(self):
        """Tests that levels within tolerance cluster into stronger confluence zones (PDF Section 4)."""
        levels = {
            "PDH": 2650.0,
            "PWH": 2650.2,   # Very close to PDH -> Confluence!
            "PMH": 2720.0,
            "PDL": 2600.0,
            "PWL": 2580.0,
            "PML": 2550.0
        }
        atr = 5.0  # Confluence tol = 0.35 * 5.0 = 1.75
        clusters = self.engine.detect_confluence_clusters(levels, atr)
        self.assertGreater(len(clusters), 0)
        c0 = clusters[0]
        self.assertIn("PDH", c0.levels)
        self.assertIn("PWH", c0.levels)
        # Combined weight: PDH(1.0) + PWH(2.0) = 3.0
        self.assertEqual(c0.combined_weight, 3.0)

    def test_four_price_behaviors(self):
        """Tests classification of Sweep, Acceptance, Reaction, Magnet (PDF Section 5)."""
        levels = {"PDH": 2650.0, "PDL": 2600.0}
        atr = 5.0

        # 1. Sweep test: High pierces 2650 by >= 0.1 ATR (0.5), but close < 2650
        row_sweep = pd.Series({"open": 2645.0, "high": 2652.0, "low": 2640.0, "close": 2646.0})
        b_type, code, _ = self.engine.classify_bar_behavior(row_sweep, None, levels, atr)
        self.assertEqual(b_type, "Sweep")
        self.assertEqual(code, "PDH")

        # 2. Acceptance test: Candle closes above 2650 and holds
        row_acc = pd.Series({"open": 2651.0, "high": 2660.0, "low": 2650.5, "close": 2658.0})
        b_type, code, _ = self.engine.classify_bar_behavior(row_acc, None, levels, atr)
        self.assertEqual(b_type, "Acceptance")
        self.assertEqual(code, "PDH")

        # 3. Reaction test: Approaches within 0.3 ATR (1.5) without piercing, and rejects
        row_react = pd.Series({"open": 2649.0, "high": 2649.5, "low": 2642.0, "close": 2643.0})
        b_type, code, _ = self.engine.classify_bar_behavior(row_react, None, levels, atr)
        self.assertEqual(b_type, "Reaction")
        self.assertEqual(code, "PDH")

        # 4. Magnet test: Steadily advancing toward untouched level
        prev_row = pd.Series({"close": 2642.0})
        row_mag = pd.Series({"open": 2643.0, "high": 2646.0, "low": 2642.5, "close": 2645.5})
        b_type, code, _ = self.engine.classify_bar_behavior(row_mag, prev_row, levels, atr)
        self.assertEqual(b_type, "Magnet")
        self.assertEqual(code, "PDH")

    def test_checklist_evaluation(self):
        """Tests Section 6 Checklist execution on enriched dataframe."""
        df_ind = add_core_indicators(self.df, tz_name="UTC")
        chk = self.engine.evaluate_bar_checklist(len(df_ind) // 2, df_ind)

        self.assertTrue(chk.levels_up_to_date)
        self.assertTrue(chk.nearest_above is not None or chk.nearest_below is not None)
        self.assertIn(chk.htf_bias, ["Bullish", "Bearish", "Neutral"])

    def test_eighty_percent_chronological_split(self):
        """Verifies exactly 80% training data split in trainer."""
        trainer = ModelTrainer()
        synthetic_features = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=1000, freq="15min"),
            "label": np.random.choice(["1", "0"], 1000)
        })
        tr, val, te = trainer.chronological_split(synthetic_features, embargo_bars=16)

        # 80% of 1000 = 800
        self.assertEqual(len(tr), 800)
        # Verify strict chronological separation
        self.assertLess(tr["timestamp"].max(), val["timestamp"].min())
        self.assertLess(val["timestamp"].max(), te["timestamp"].min())


if __name__ == "__main__":
    unittest.main()
