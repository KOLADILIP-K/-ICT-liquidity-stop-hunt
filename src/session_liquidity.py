"""
Session Liquidity and Confluence Detection Module
Computes ICT session high/low levels (Option B overlapping) and measures confluence with structural zones.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SessionLevel:
    """Represents a frozen session high or low reference level."""
    session_name: str
    level_type: str                   # 'high' or 'low'
    price: float
    session_start: datetime
    session_end: datetime
    is_active: bool = True
    swept: bool = False
    sweep_timestamp: Optional[datetime] = None


class SessionLiquidityManager:
    """
    Manages Option B timezone-aware session ranges and detects confluence with pattern zones.
    """

    def __init__(
        self,
        tz_name: str = "America/New_York",
        session_defs: Optional[Dict[str, str]] = None,
        confluence_distance_atr: float = 0.5
    ):
        self.tz_name = tz_name
        self.confluence_distance_atr = confluence_distance_atr
        # Default ICT Killzones matching Pine Script
        self.session_defs = session_defs or {
            "Asia": "20:00-00:00",
            "London": "02:00-05:00",
            "NYAM": "09:30-11:00",
            "NYPM": "13:30-16:00"
        }

    def _parse_session_time(self, time_str: str) -> Tuple[time, time]:
        start_str, end_str = time_str.split("-")
        s_h, s_m = map(int, start_str.split(":"))
        e_h, e_m = map(int, end_str.split(":"))
        return time(s_h, s_m), time(e_h, e_m)

    def is_in_session(self, dt: datetime, session_name: str) -> bool:
        """Determines if a timestamp in local timezone falls inside the given session window."""
        if session_name not in self.session_defs:
            return False
        start_t, end_t = self._parse_session_time(self.session_defs[session_name])
        t = dt.time()

        if start_t <= end_t:
            return start_t <= t < end_t
        else:
            # Crosses midnight (e.g. 20:00 - 00:00)
            return t >= start_t or t < end_t

    def compute_session_levels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[SessionLevel]]:
        """
        Enriches DataFrame with active session highs and lows without look-ahead bias.
        
        At each bar t:
        - If inside session S, developing high and low are updated with max(high) and min(low) so far.
        - Once session S ends, its final high and low become frozen reference levels.
        - Previous session levels are carried forward until that session occurs again.
        """
        df_out = df.copy()
        ts = df_out["timestamp"]
        if ts.dt.tz is None:
            ts_local = ts.dt.tz_localize("UTC").dt.tz_convert(self.tz_name)
        else:
            ts_local = ts.dt.tz_convert(self.tz_name)

        n = len(df_out)
        session_levels: List[SessionLevel] = []

        # Initialize session tracking columns
        for sess in self.session_defs.keys():
            df_out[f"{sess}_in_session"] = False
            df_out[f"{sess}_high"] = np.nan
            df_out[f"{sess}_low"] = np.nan

        # Process bars sequentially to ensure zero look-ahead
        active_ranges: Dict[str, Dict[str, Any]] = {
            s: {"high": np.nan, "low": np.nan, "start": None, "in_sess": False}
            for s in self.session_defs.keys()
        }

        sess_highs = {s: np.full(n, np.nan) for s in self.session_defs.keys()}
        sess_lows = {s: np.full(n, np.nan) for s in self.session_defs.keys()}
        in_sess_flags = {s: np.full(n, False) for s in self.session_defs.keys()}

        highs = df_out["high"].values
        lows = df_out["low"].values
        ts_loc_list = ts_local.dt.to_pydatetime()

        for i in range(n):
            dt = ts_loc_list[i]
            for sess in self.session_defs.keys():
                in_now = self.is_in_session(dt, sess)
                in_sess_flags[sess][i] = in_now
                state = active_ranges[sess]

                if in_now:
                    if not state["in_sess"]:
                        # Session started
                        state["in_sess"] = True
                        state["high"] = highs[i]
                        state["low"] = lows[i]
                        state["start"] = dt
                    else:
                        # Updating current session
                        state["high"] = max(state["high"], highs[i])
                        state["low"] = min(state["low"], lows[i])
                else:
                    if state["in_sess"]:
                        # Session just finished: freeze completed levels
                        state["in_sess"] = False
                        session_levels.append(SessionLevel(
                            session_name=sess,
                            level_type="high",
                            price=float(state["high"]),
                            session_start=state["start"],
                            session_end=dt
                        ))
                        session_levels.append(SessionLevel(
                            session_name=sess,
                            level_type="low",
                            price=float(state["low"]),
                            session_start=state["start"],
                            session_end=dt
                        ))

                # Record current frozen or developing level
                sess_highs[sess][i] = state["high"]
                sess_lows[sess][i] = state["low"]

        for sess in self.session_defs.keys():
            df_out[f"{sess}_in_session"] = in_sess_flags[sess]
            df_out[f"{sess}_high"] = pd.Series(sess_highs[sess], index=df_out.index).ffill()
            df_out[f"{sess}_low"] = pd.Series(sess_lows[sess], index=df_out.index).ffill()

        return df_out, session_levels

    def calculate_confluence(
        self,
        zone_price: float,
        current_bar_row: pd.Series,
        atr: float
    ) -> Tuple[int, List[str]]:
        """
        Calculates confluence score between a structural liquidity price and session/daily levels.
        Confluence exists when an active session low or prior day low is within confluence_distance_atr.
        """
        confluent_levels: List[str] = []
        atr_tol = max(self.confluence_distance_atr * atr, 1e-4)

        # Check Prior Day Low (PDL)
        pdl = current_bar_row.get("pdl", np.nan)
        if not np.isnan(pdl) and abs(zone_price - pdl) <= atr_tol:
            confluent_levels.append("PDL")

        # Check Prior Day High (PDH)
        pdh = current_bar_row.get("pdh", np.nan)
        if not np.isnan(pdh) and abs(zone_price - pdh) <= atr_tol:
            confluent_levels.append("PDH")

        # Check active session lows
        for sess in self.session_defs.keys():
            s_low = current_bar_row.get(f"{sess}_low", np.nan)
            if not np.isnan(s_low) and abs(zone_price - s_low) <= atr_tol:
                confluent_levels.append(f"{sess}_Low")

        return len(confluent_levels), confluent_levels
