"""
Tests for Backtesting Engine Mechanics and Cost Accounting.
"""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from src.backtest import BacktestEngine


class TestBacktest(unittest.TestCase):

    def test_backtest_cost_deduction_and_tp_exit(self):
        n = 10
        timestamps = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n)]
        
        # Bar 2 triggers entry at 100.0, Bar 4 reaches TP at 103.0
        highs = [101.0, 101.0, 101.0, 101.5, 104.0, 101.0, 101.0, 101.0, 101.0, 101.0]
        lows = [99.0, 99.0, 99.0, 99.5, 100.0, 99.0, 99.0, 99.0, 99.0, 99.0]
        opens = [100.0] * n
        closes = [100.0] * n

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": np.full(n, 2.0),
            "instrument": ["XAUUSD"] * n
        })

        # Signal at bar 2
        signals_df = pd.DataFrame([{
            "sweep_bar_index": 2,
            "reclaim_bar_index": 2,
            "level_price": 100.0,
            "sweep_price_extreme": 99.0,
            "hunt_probability": 0.70
        }])

        bt = BacktestEngine(
            initial_capital=10000.0,
            risk_per_trade_pct=1.0,
            spread_gold=0.20,
            slippage_atr=0.0,
            commission_per_order=0.50,
            target_atr_x=1.5  # TP = Entry + 3.0
        )

        trades_df, equity_df, metrics = bt.run_backtest(df, signals_df, probability_threshold=0.60)

        self.assertEqual(len(trades_df), 1)
        trade = trades_df.iloc[0]
        self.assertEqual(trade["exit_reason"], "TAKE_PROFIT")
        self.assertGreater(trade["pnl_net"], 0.0)
        # Cost must include spread and commissions
        self.assertGreater(trade["costs_total"], 0.0)
        self.assertAlmostEqual(trade["pnl_net"], trade["pnl_gross"] - trade["costs_total"], places=2)


if __name__ == "__main__":
    unittest.main()
