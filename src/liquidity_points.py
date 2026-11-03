"""
Liquidity Points Reference Module
Implements designated Daily, Weekly, and Monthly Highs and Lows (PDH, PDL, PWH, PWL, PMH, PML),
zone confluence clustering, price behavior detection (Sweep, Acceptance, Reaction, Magnet),
and the complete Section 6 checklist engine directly derived from the Liquidity Points Reference.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class LiquidityPointMeta:
    code: str
    name: str
    liquidity_type: str        # 'Buy-side' (above highs) or 'Sell-side' (below lows)
    resting_orders: str
    weight_name: str           # 'Low', 'Medium', 'High'
    weight_val: float          # 1.0, 2.0, 3.0
    color: str
    line_style: str            # 'solid', 'dashed', 'dotted'


# Designated reference specification from PDF Section 2
DESIGNATED_LIQUIDITY_POINTS: Dict[str, LiquidityPointMeta] = {
    "PDH": LiquidityPointMeta(
        code="PDH",
        name="Previous Day High",
        liquidity_type="Buy-side",
        resting_orders="Stops of shorts, breakout buys above",
        weight_name="Low",
        weight_val=1.0,
        color="#EF5350",
        line_style="dotted"
    ),
    "PDL": LiquidityPointMeta(
        code="PDL",
        name="Previous Day Low",
        liquidity_type="Sell-side",
        resting_orders="Stops of longs, breakout sells below",
        weight_name="Low",
        weight_val=1.0,
        color="#26A69A",
        line_style="dotted"
    ),
    "PWH": LiquidityPointMeta(
        code="PWH",
        name="Previous Week High",
        liquidity_type="Buy-side",
        resting_orders="Same, from swing positions",
        weight_name="Medium",
        weight_val=2.0,
        color="#AB47BC",
        line_style="dashed"
    ),
    "PWL": LiquidityPointMeta(
        code="PWL",
        name="Previous Week Low",
        liquidity_type="Sell-side",
        resting_orders="Same, from swing positions",
        weight_name="Medium",
        weight_val=2.0,
        color="#42A5F5",
        line_style="dashed"
    ),
    "PMH": LiquidityPointMeta(
        code="PMH",
        name="Previous Month High",
        liquidity_type="Buy-side",
        resting_orders="Same, from position traders",
        weight_name="High",
        weight_val=3.0,
        color="#E91E63",
        line_style="solid"
    ),
    "PML": LiquidityPointMeta(
        code="PML",
        name="Previous Month Low",
        liquidity_type="Sell-side",
        resting_orders="Same, from position traders",
        weight_name="High",
        weight_val=3.0,
        color="#00B0FF",
        line_style="solid"
    ),
}


@dataclass
class LiquidityCluster:
    """Represents a confluence zone where multiple HTF levels sit within a few ticks/ATR."""
    zone_low: float
    zone_high: float
    center_price: float
    levels: List[str]
    combined_weight: float
    zone_type: str             # 'Buy-side', 'Sell-side', 'Confluent Zone'
    width_atr: float


@dataclass
class ChecklistResult:
    """Snapshot of the 6-point checklist at a given bar."""
    bar_index: int
    timestamp: datetime
    current_price: float
    atr: float
    levels_up_to_date: bool
    all_levels: Dict[str, float]
    nearest_above: Optional[Tuple[str, float, float]]   # (code, price, dist_atr)
    nearest_below: Optional[Tuple[str, float, float]]   # (code, price, dist_atr)
    swept_this_session: List[str]
    swept_this_week: List[str]
    confluence_clusters: List[LiquidityCluster]
    active_behavior: str                               # 'Sweep', 'Acceptance', 'Reaction', 'Magnet', 'None'
    interacting_level: Optional[str]
    htf_bias: str                                      # 'Bullish', 'Bearish', 'Neutral'
    behavior_fits_bias: bool


class LiquidityPointsEngine:
    """
    Core engine managing designated HTF liquidity points, reading price behaviors,
    detecting confluence zones, and executing the PDF Section 6 checklist.
    """

    def __init__(
        self,
        confluence_atr_tol: float = 0.35,
        sweep_threshold_atr: float = 0.10,
        reaction_proximity_atr: float = 0.30,
        magnet_range_atr: float = 1.50
    ):
        self.confluence_atr_tol = confluence_atr_tol
        self.sweep_threshold_atr = sweep_threshold_atr
        self.reaction_proximity_atr = reaction_proximity_atr
        self.magnet_range_atr = magnet_range_atr
        self.point_defs = DESIGNATED_LIQUIDITY_POINTS

    def detect_confluence_clusters(self, levels_dict: Dict[str, float], atr: float) -> List[LiquidityCluster]:
        """
        Groups levels within confluence_atr_tol * ATR into stronger confluence zones.
        PDF Section 4: "When two levels sit within a few ticks or pips, treat them as one stronger zone (confluence)."
        """
        valid_items = [(k, v) for k, v in levels_dict.items() if not np.isnan(v)]
        if len(valid_items) < 2:
            return []

        # Sort by price
        sorted_items = sorted(valid_items, key=lambda x: x[1])
        clusters: List[LiquidityCluster] = []
        tol = max(self.confluence_atr_tol * atr, 1e-4)

        current_cluster = [sorted_items[0]]

        for i in range(1, len(sorted_items)):
            code, price = sorted_items[i]
            prev_price = current_cluster[-1][1]

            if abs(price - prev_price) <= tol:
                current_cluster.append((code, price))
            else:
                if len(current_cluster) >= 2:
                    clusters.append(self._build_cluster(current_cluster, atr))
                current_cluster = [(code, price)]

        if len(current_cluster) >= 2:
            clusters.append(self._build_cluster(current_cluster, atr))

        return clusters

    def _build_cluster(self, items: List[Tuple[str, float]], atr: float) -> LiquidityCluster:
        codes = [item[0] for item in items]
        prices = [item[1] for item in items]
        z_low = min(prices)
        z_high = max(prices)
        tot_weight = sum(self.point_defs[c].weight_val for c in codes if c in self.point_defs)
        types = set(self.point_defs[c].liquidity_type for c in codes if c in self.point_defs)
        z_type = "Confluent Zone" if len(types) > 1 else (list(types)[0] if types else "Confluence")

        return LiquidityCluster(
            zone_low=z_low,
            zone_high=z_high,
            center_price=0.5 * (z_low + z_high),
            levels=codes,
            combined_weight=tot_weight,
            zone_type=z_type,
            width_atr=abs(z_high - z_low) / max(atr, 1e-4)
        )

    def classify_bar_behavior(
        self,
        bar_row: pd.Series,
        prev_bar_row: Optional[pd.Series],
        levels_dict: Dict[str, float],
        atr: float
    ) -> Tuple[str, Optional[str], str]:
        """
        Classifies price behavior at liquidity points according to PDF Section 5:
        1. Sweep: Wick pierces the level, then price closes back inside the range.
        2. Acceptance: Candle(s) close beyond the level and hold.
        3. Reaction: Price stalls or rejects near the level without piercing it.
        4. Magnet: Price moves steadily toward an untouched level.
        
        Returns: (behavior_name, interacting_level_code, description)
        """
        close = bar_row["close"]
        high = bar_row["high"]
        low = bar_row["low"]
        open_p = bar_row["open"]

        atr_val = max(atr, 1e-4)
        sweep_pen = self.sweep_threshold_atr * atr_val
        react_tol = self.reaction_proximity_atr * atr_val

        # 1. Check Sweep (Priority: reversals precede follow-through)
        # Check Highs (PDH, PWH, PMH) - Buy-side sweep
        for code in ["PMH", "PWH", "PDH"]:
            lvl = levels_dict.get(code, np.nan)
            if not np.isnan(lvl):
                if high >= lvl + sweep_pen and close < lvl:
                    return "Sweep", code, f"Wick pierced {code} ({lvl:.2f}) and closed back inside range (Buy-side sweep)."

        # Check Lows (PDL, PWL, PML) - Sell-side sweep
        for code in ["PML", "PWL", "PDL"]:
            lvl = levels_dict.get(code, np.nan)
            if not np.isnan(lvl):
                if low <= lvl - sweep_pen and close > lvl:
                    return "Sweep", code, f"Wick pierced {code} ({lvl:.2f}) and closed back inside range (Sell-side sweep)."

        # 2. Check Acceptance (Genuine Breakout)
        for code in ["PMH", "PWH", "PDH"]:
            lvl = levels_dict.get(code, np.nan)
            if not np.isnan(lvl):
                if close > lvl and open_p >= lvl - (0.1 * atr_val):
                    return "Acceptance", code, f"Candle closed and held above {code} ({lvl:.2f}) (Bullish acceptance)."

        for code in ["PML", "PWL", "PDL"]:
            lvl = levels_dict.get(code, np.nan)
            if not np.isnan(lvl):
                if close < lvl and open_p <= lvl + (0.1 * atr_val):
                    return "Acceptance", code, f"Candle closed and held below {code} ({lvl:.2f}) (Bearish acceptance)."

        # 3. Check Reaction (Level respected without piercing)
        for code in ["PMH", "PWH", "PDH"]:
            lvl = levels_dict.get(code, np.nan)
            if not np.isnan(lvl):
                dist = lvl - high
                if 0 <= dist <= react_tol and close < open_p:
                    return "Reaction", code, f"Price stalled near {code} ({lvl:.2f}) without piercing it; rejection observed."

        for code in ["PML", "PWL", "PDL"]:
            lvl = levels_dict.get(code, np.nan)
            if not np.isnan(lvl):
                dist = low - lvl
                if 0 <= dist <= react_tol and close > open_p:
                    return "Reaction", code, f"Price stalled near {code} ({lvl:.2f}) without piercing it; bounce observed."

        # 4. Check Magnet (Steady approach to untouched level)
        if prev_bar_row is not None:
            prev_close = prev_bar_row["close"]
            for code in ["PMH", "PWH", "PDH"]:
                lvl = levels_dict.get(code, np.nan)
                if not np.isnan(lvl) and lvl > close:
                    dist = (lvl - close) / atr_val
                    if 0.2 < dist <= self.magnet_range_atr and close > prev_close:
                        return "Magnet", code, f"Price steadily advancing toward untouched {code} ({lvl:.2f}) draw on liquidity."

            for code in ["PML", "PWL", "PDL"]:
                lvl = levels_dict.get(code, np.nan)
                if not np.isnan(lvl) and lvl < close:
                    dist = (close - lvl) / atr_val
                    if 0.2 < dist <= self.magnet_range_atr and close < prev_close:
                        return "Magnet", code, f"Price steadily declining toward untouched {code} ({lvl:.2f}) draw on liquidity."

        return "None", None, "No active liquidity point interaction on current bar."

    def evaluate_bar_checklist(
        self,
        bar_idx: int,
        df: pd.DataFrame,
        swept_session_history: Optional[List[str]] = None,
        swept_week_history: Optional[List[str]] = None
    ) -> ChecklistResult:
        """
        Executes the PDF Section 6 Quick Checklist on bar_idx:
        1. Are PDH/PDL, PWH/PWL, PMH/PML marked and up to date?
        2. Which levels are nearest above and below current price?
        3. Has any level already been swept this session or week?
        4. Do multiple levels cluster into one zone?
        5. Is price sweeping, accepting, or reacting, and does it fit HTF bias?
        """
        row = df.iloc[bar_idx]
        prev_row = df.iloc[bar_idx - 1] if bar_idx > 0 else None
        atr = max(row.get("atr", 1.0), 1e-4)
        close = row["close"]
        ts = row["timestamp"]

        # Item 1: Levels status
        levels = {
            "PDH": row.get("pdh", np.nan),
            "PDL": row.get("pdl", np.nan),
            "PWH": row.get("pwh", np.nan),
            "PWL": row.get("pwl", np.nan),
            "PMH": row.get("pmh", np.nan),
            "PML": row.get("pml", np.nan),
        }
        levels_up_to_date = all(not np.isnan(v) for v in levels.values())

        # Item 2: Nearest above and below
        above_candidates = [(k, v, (v - close) / atr) for k, v in levels.items() if not np.isnan(v) and v > close]
        below_candidates = [(k, v, (close - v) / atr) for k, v in levels.items() if not np.isnan(v) and v < close]

        nearest_above = min(above_candidates, key=lambda x: x[2]) if above_candidates else None
        nearest_below = min(below_candidates, key=lambda x: x[2]) if below_candidates else None

        # Item 3: Swept this session / week
        swept_session = swept_session_history or []
        swept_week = swept_week_history or []

        # Item 4: Confluence clusters
        clusters = self.detect_confluence_clusters(levels, atr)

        # Item 5: Active behavior & HTF bias
        behavior, interacting_code, _ = self.classify_bar_behavior(row, prev_row, levels, atr)

        # Trend bias: 20-period return or SMA
        ret_20 = row.get("return_20", 0.0)
        htf_bias = "Bullish" if ret_20 > 0.002 else ("Bearish" if ret_20 < -0.002 else "Neutral")

        # Behavior fit:
        # Bullish bias fits: Sell-side sweep (reversal up), Bullish Acceptance, or Magnet toward Buy-side High
        fits_bias = True
        if behavior == "Sweep":
            # If sell-side sweep (PDL, PWL, PML), it's a bullish reversal -> fits Bullish bias
            if interacting_code in ["PDL", "PWL", "PML"]:
                fits_bias = (htf_bias in ["Bullish", "Neutral"])
            elif interacting_code in ["PDH", "PWH", "PMH"]:
                fits_bias = (htf_bias in ["Bearish", "Neutral"])
        elif behavior == "Acceptance":
            if interacting_code in ["PDH", "PWH", "PMH"]:
                fits_bias = (htf_bias == "Bullish")
            elif interacting_code in ["PDL", "PWL", "PML"]:
                fits_bias = (htf_bias == "Bearish")

        return ChecklistResult(
            bar_index=bar_idx,
            timestamp=ts,
            current_price=close,
            atr=atr,
            levels_up_to_date=levels_up_to_date,
            all_levels=levels,
            nearest_above=nearest_above,
            nearest_below=nearest_below,
            swept_this_session=swept_session,
            swept_this_week=swept_week,
            confluence_clusters=clusters,
            active_behavior=behavior,
            interacting_level=interacting_code,
            htf_bias=htf_bias,
            behavior_fits_bias=fits_bias
        )
