"""
Backtesting and Simulation Engine
Simulates trading strategy based on stop-hunt probability signals with realistic execution mechanics:
spread, slippage, commission, concurrent position caps, cooldown, and risk-adjusted analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """Represents a simulated executed trade."""
    trade_id: str
    instrument: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = "BUY"
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    costs_total: float = 0.0
    exit_reason: str = "OPEN"           # 'TAKE_PROFIT', 'STOP_LOSS', 'TIMEOUT', 'END_OF_DATA'
    duration_bars: int = 0
    hunt_probability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time) if self.exit_time else None,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "position_size": round(self.position_size, 3),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "pnl_gross": round(self.pnl_gross, 2),
            "pnl_net": round(self.pnl_net, 2),
            "costs_total": round(self.costs_total, 2),
            "exit_reason": self.exit_reason,
            "duration_bars": self.duration_bars,
            "hunt_probability": round(self.hunt_probability, 4)
        }


class BacktestEngine:
    """
    Executes backtest simulation on historical OHLCV data using model signals.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade_pct: float = 1.0,
        max_open_positions: int = 3,
        cooldown_bars: int = 5,
        spread_gold: float = 0.25,
        spread_silver: float = 0.02,
        slippage_atr: float = 0.05,
        commission_per_order: float = 0.50,
        target_atr_x: float = 1.5,
        sl_buffer_atr: float = 0.2,
        max_holding_bars: int = 32
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_open_positions = max_open_positions
        self.cooldown_bars = cooldown_bars
        self.spread_gold = spread_gold
        self.spread_silver = spread_silver
        self.slippage_atr = slippage_atr
        self.commission_per_order = commission_per_order
        self.target_atr_x = target_atr_x
        self.sl_buffer_atr = sl_buffer_atr
        self.max_holding_bars = max_holding_bars

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        probability_threshold: float = 0.60
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes sequential bar-by-bar simulation.
        
        Args:
            df: OHLCV DataFrame with 'atr'.
            signals_df: DataFrame with 'sweep_bar_index', 'reclaim_bar_index', 'hunt_probability', 'level_price'.
            probability_threshold: Minimum probability to trigger entry.
        
        Returns:
            trades_df: DataFrame of completed trades.
            equity_curve: DataFrame of equity and drawdown over time.
            summary_metrics: Performance dictionary.
        """
        if signals_df.empty or df.empty:
            return pd.DataFrame(), pd.DataFrame(), {"error": "Empty data"}

        # Filter signals by threshold
        valid_signals = signals_df[signals_df["hunt_probability"] >= probability_threshold].copy()
        valid_signals = valid_signals.sort_values("sweep_bar_index")

        signal_map = {}
        for _, s in valid_signals.iterrows():
            entry_idx = int(s.get("reclaim_bar_index", s["sweep_bar_index"]))
            if entry_idx not in signal_map:
                signal_map[entry_idx] = []
            signal_map[entry_idx].append(s)

        current_capital = self.initial_capital
        peak_equity = self.initial_capital
        active_trades: List[TradeRecord] = []
        closed_trades: List[TradeRecord] = []
        equity_records = []

        last_entry_bar = -999
        trade_counter = 0

        n = len(df)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        timestamps = df["timestamp"].values
        atrs = df["atr"].values if "atr" in df.columns else np.full(n, 1.0)
        instruments = df["instrument"].values if "instrument" in df.columns else ["XAUUSD"] * n

        for i in range(n):
            dt = pd.Timestamp(timestamps[i]).to_pydatetime()
            bar_atr = max(atrs[i], 1e-4)
            inst = instruments[i]
            spread = self.spread_gold if "XAU" in str(inst).upper() else self.spread_silver
            slip = self.slippage_atr * bar_atr

            # 1. Manage active open trades
            remaining_trades = []
            for t in active_trades:
                t.duration_bars += 1
                hit_tp = False
                hit_sl = False
                exit_price = 0.0

                # Check Stop Loss first (conservative evaluation)
                if lows[i] <= t.stop_loss:
                    hit_sl = True
                    exit_price = min(t.stop_loss, opens[i]) - slip
                    t.exit_reason = "STOP_LOSS"
                elif highs[i] >= t.take_profit:
                    hit_tp = True
                    exit_price = max(t.take_profit, opens[i]) - slip
                    t.exit_reason = "TAKE_PROFIT"
                elif t.duration_bars >= self.max_holding_bars:
                    # Timeout exit
                    exit_price = closes[i] - slip
                    t.exit_reason = "TIMEOUT"

                if hit_sl or hit_tp or t.duration_bars >= self.max_holding_bars:
                    t.exit_price = exit_price
                    t.exit_time = dt
                    # PnL
                    price_diff = t.exit_price - t.entry_price
                    t.pnl_gross = price_diff * t.position_size
                    # Costs: 2 orders (entry + exit), half spread on each side
                    cost = (spread * t.position_size) + (2 * self.commission_per_order)
                    t.costs_total = cost
                    t.pnl_net = t.pnl_gross - cost
                    current_capital += t.pnl_net
                    closed_trades.append(t)
                else:
                    remaining_trades.append(t)

            active_trades = remaining_trades

            # 2. Check for new trade signals at bar i
            if i in signal_map and (i - last_entry_bar >= self.cooldown_bars):
                if len(active_trades) < self.max_open_positions:
                    for sig in signal_map[i]:
                        if len(active_trades) >= self.max_open_positions:
                            break

                        prob = float(sig["hunt_probability"])
                        entry_price = closes[i] + (spread / 2.0) + slip
                        level = float(sig.get("level_price", closes[i]))
                        sweep_ext = float(sig.get("sweep_price_extreme", level - 0.2 * bar_atr))
                        
                        # Stop loss placed below sweep extreme with buffer
                        sl_price = sweep_ext - (self.sl_buffer_atr * bar_atr)
                        tp_price = entry_price + (self.target_atr_x * bar_atr)

                        risk_per_unit = max(entry_price - sl_price, 0.5 * bar_atr)
                        risk_capital = current_capital * (self.risk_per_trade_pct / 100.0)
                        pos_size = risk_capital / risk_per_unit

                        trade_counter += 1
                        trade = TradeRecord(
                            trade_id=f"T_{trade_counter}_{i}",
                            instrument=inst,
                            entry_time=dt,
                            direction="BUY",
                            entry_price=entry_price,
                            stop_loss=sl_price,
                            take_profit=tp_price,
                            position_size=pos_size,
                            hunt_probability=prob
                        )
                        active_trades.append(trade)
                        last_entry_bar = i

            # 3. Track daily/bar equity
            unrealized_pnl = sum((closes[i] - t.entry_price) * t.position_size for t in active_trades)
            total_equity = current_capital + unrealized_pnl
            peak_equity = max(peak_equity, total_equity)
            drawdown = max(0.0, peak_equity - total_equity)
            drawdown_pct = (drawdown / peak_equity * 100.0) if peak_equity > 0 else 0.0

            equity_records.append({
                "timestamp": dt,
                "capital": round(current_capital, 2),
                "total_equity": round(total_equity, 2),
                "peak_equity": round(peak_equity, 2),
                "drawdown": round(drawdown, 2),
                "drawdown_pct": round(drawdown_pct, 2),
                "open_positions": len(active_trades)
            })

        # Close any remaining open trades at final close
        for t in active_trades:
            t.exit_price = closes[-1]
            t.exit_time = pd.Timestamp(timestamps[-1]).to_pydatetime()
            t.exit_reason = "END_OF_DATA"
            price_diff = t.exit_price - t.entry_price
            t.pnl_gross = price_diff * t.position_size
            cost = (spread * t.position_size) + (2 * self.commission_per_order)
            t.costs_total = cost
            t.pnl_net = t.pnl_gross - cost
            current_capital += t.pnl_net
            closed_trades.append(t)

        trades_df = pd.DataFrame([t.to_dict() for t in closed_trades])
        equity_df = pd.DataFrame(equity_records)

        # Performance summary
        metrics = self._calculate_metrics(trades_df, equity_df)
        return trades_df, equity_df, metrics

    def _calculate_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        if trades_df.empty:
            return {
                "total_trades": 0,
                "net_profit": 0.0,
                "net_return_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0
            }

        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl_net"] > 0]
        losses = trades_df[trades_df["pnl_net"] <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = float(wins["pnl_net"].sum()) if win_count > 0 else 0.0
        gross_loss = abs(float(losses["pnl_net"].sum())) if loss_count > 0 else 0.0
        profit_factor = round(gross_profit / max(gross_loss, 1e-4), 2)

        net_profit = float(trades_df["pnl_net"].sum())
        net_return_pct = round((net_profit / self.initial_capital) * 100.0, 2)
        avg_win = float(wins["pnl_net"].mean()) if win_count > 0 else 0.0
        avg_loss = float(losses["pnl_net"].mean()) if loss_count > 0 else 0.0
        expectancy = (win_rate / 100.0 * avg_win) + ((1.0 - win_rate / 100.0) * avg_loss)

        max_dd_pct = float(equity_df["drawdown_pct"].max()) if not equity_df.empty else 0.0

        # Sharpe ratio
        eq_returns = equity_df["total_equity"].pct_change().dropna()
        sharpe = float(np.sqrt(252 * 24 * 4) * eq_returns.mean() / (eq_returns.std() + 1e-8)) if len(eq_returns) > 1 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(self.initial_capital + net_profit, 2),
            "net_profit": round(net_profit, 2),
            "net_return_pct": net_return_pct,
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": round(win_rate, 2),
            "profit_factor": profit_factor,
            "expectancy": round(expectancy, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2)
        }
