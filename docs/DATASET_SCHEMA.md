# Dataset Schema & Data Dictionary

This document defines the raw OHLCV schema, engineered feature dictionary, and target label definitions for the ICT Liquidity Intelligence & Stop-Hunt Detection System across all four instruments: **Gold (XAUUSD)**, **Silver (XAGUSD)**, **Bitcoin (BTCUSD)**, and **Ethereum (ETHUSD)**.

---

## 1. Raw Market Data Schema

| Column Name | Data Type | Units / Format | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns, UTC]` | ISO 8601 (UTC) | Start time of the 15-minute candle interval. |
| `open` | `float64` | Currency ($) | Opening price of the candle. |
| `high` | `float64` | Currency ($) | Highest price traded during the candle. |
| `low` | `float64` | Currency ($) | Lowest price traded during the candle. |
| `close` | `float64` | Currency ($) | Final closing price of the candle. |
| `volume` | `float64` / `int64` | Contracts / Ticks | Traded volume or tick count during the candle. |
| `instrument` | `string` | Categorical | Symbol ticker: `XAUUSD`, `XAGUSD`, `BTCUSD`, `ETHUSD`. |
| `provider` | `string` | Metadata | Feed origin (e.g. `dukascopy`, `broker`, `binance`). |

---

## 2. Designated Liquidity Points (PDF Reference Specification)

As designated in the *Liquidity Points Reference* (Daily, Weekly & Monthly Highs and Lows):

| Code | Level Name | Type | Resting Orders | Weight | Visual Style |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PDH** | Previous Day High | Buy-side | Stops of shorts, breakout buys above | Low (1.0) | Dotted, Red (#EF5350) |
| **PDL** | Previous Day Low | Sell-side | Stops of longs, breakout sells below | Low (1.0) | Dotted, Green (#26A69A) |
| **PWH** | Previous Week High | Buy-side | Same, from swing positions | Medium (2.0) | Dashed, Purple (#AB47BC) |
| **PWL** | Previous Week Low | Sell-side | Same, from swing positions | Medium (2.0) | Dashed, Blue (#42A5F5) |
| **PMH** | Previous Month High | Buy-side | Same, from position traders | High (3.0) | Solid, Dark Red (#E91E63) |
| **PML** | Previous Month Low | Sell-side | Same, from position traders | High (3.0) | Solid, Cyan/Blue (#00B0FF) |

- **Confluence Zones:** When two or more levels sit within $\le 0.35 \times \text{ATR}$, they are treated as one combined stronger zone with aggregated weight ($\sum w_i$).
- **Behaviors at Levels:**
  1. `Sweep`: Wick pierces level, then price closes back inside range (liquidity taken $\implies$ reversal).
  2. `Acceptance`: Candle(s) close beyond level and hold (genuine breakout continuation).
  3. `Reaction`: Price stalls near level without piercing (level respected).
  4. `Magnet`: Price moves steadily toward untouched level (acts as a draw/target).

---

## 3. Engineered Pre-Event Features Schema (28 Features)

These features are computed strictly using information available on or before the candidate sweep bar ($t_{sweep}$).

| Feature Name | Data Type | Scale / Range | Description |
| :--- | :--- | :--- | :--- |
| `instrument_code` | `int32` | {0, 1, 2, 3} | Multi-asset encoding: `0`=XAUUSD, `1`=XAGUSD, `2`=BTCUSD, `3`=ETHUSD. |
| `pattern_type_code` | `int32` | {0, 1, 2, 3, 4, 5} | Categorical: `0`=DB, `1`=TB, `2`=Trendline, `3`=Daily HTF, `4`=Weekly HTF, `5`=Monthly HTF. |
| `direction_code` | `int32` | {0, 1} | `1`=Bullish (Low Sweep $\to$ Long), `0`=Bearish (High Sweep $\to$ Short). |
| `level_weight` | `float64` | {1.0, 1.5, 2.0, 3.0} | HTF weight: Daily=1.0, Weekly=2.0, Monthly=3.0, Patterns=1.5. |
| `equality_score` | `float64` | $\ge 0.0$ | $\|L_1 - L_2\| / \text{ATR}$. Cleanliness of equal lows (lower is cleaner). |
| `pattern_height_atr` | `float64` | $\ge 1.0$ | Distance from neckline peak to lowest low, in ATR units. |
| `touch_count` | `int32` | $\ge 1$ | Number of confirmed touches forming the liquidity zone. |
| `zone_width_atr` | `float64` | $\ge 0.0$ | Range between highest touch low and lowest touch low, in ATR units. |
| `bar_separation` | `int32` | $\ge 1$ | Number of bars between consecutive touches or cycle period. |
| `slope_atr` | `float64` | $[0.0, 1.5]$ | Fitted line slope in ATR per bar (0.0 for horizontal levels). |
| `line_age` | `int32` | $\ge 0$ | Total bars from pattern origin/open to sweep bar. |
| `sweep_depth_atr` | `float64` | $\ge 0.1$ | Penetration distance beyond level: $\|\text{Extreme} - \text{Level}\| / \text{ATR}$. |
| `relative_volume_sweep` | `float64` | $\ge 0.0$ | Candle volume relative to 20-period volume SMA on sweep bar. |
| `candle_lower_wick_ratio`| `float64` | $[0.0, 1.0]$ | $(\min(\text{Open}, \text{Close}) - \text{Low}) / (\text{High} - \text{Low})$. Lower rejection wick. |
| `candle_upper_wick_ratio`| `float64` | $[0.0, 1.0]$ | $(\text{High} - \max(\text{Open}, \text{Close})) / (\text{High} - \text{Low})$. Upper rejection wick. |
| `candle_body_ratio` | `float64` | $[0.0, 1.0]$ | $\|\text{Close} - \text{Open}\| / (\text{High} - \text{Low})$. Body proportion. |
| `candle_close_location` | `float64` | $[0.0, 1.0]$ | $(\text{Close} - \text{Low}) / (\text{High} - \text{Low})$. Position of close in candle range. |
| `dist_to_pdl_atr` | `float64` | Real | $(\text{Close} - \text{PDL}) / \text{ATR}$. Distance to Prior Day Low. |
| `dist_to_pdh_atr` | `float64` | Real | $(\text{PDH} - \text{Close}) / \text{ATR}$. Distance to Prior Day High. |
| `dist_to_pwl_atr` | `float64` | Real | $(\text{Close} - \text{PWL}) / \text{ATR}$. Distance to Prior Week Low. |
| `dist_to_pwh_atr` | `float64` | Real | $(\text{PWH} - \text{Close}) / \text{ATR}$. Distance to Prior Week High. |
| `dist_to_pml_atr` | `float64` | Real | $(\text{Close} - \text{PML}) / \text{ATR}$. Distance to Prior Month Low. |
| `dist_to_pmh_atr` | `float64` | Real | $(\text{PMH} - \text{Close}) / \text{ATR}$. Distance to Prior Month High. |
| `confluence_score` | `int32` | $\{0, 1, 2, 3+\}$ | Count of confluent session / daily / weekly levels within $0.5 \times \text{ATR}$. |
| `session_code` | `int32` | $\{0, 1, 2, 3, 4\}$ | Session: `0`=Asia, `1`=London, `2`=NYAM, `3`=NYPM, `4`=Other. |
| `rolling_volatility` | `float64` | $\ge 0.0$ | 20-bar standard deviation of log returns. |
| `return_5` | `float64` | Real | 5-bar cumulative percentage return prior to sweep. |
| `return_20` | `float64` | Real | 20-bar cumulative percentage return prior to sweep. |

---

## 4. Historical Outcome & Target Labels

| Field Name | Type | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `reclaimed` | `bool` | `{True, False}` | Whether price closed back inside range across level within $k$ bars ($k \le 3$). |
| `reclaim_speed_bars` | `int32` | $\{0, 1, 2, 3, -1\}$ | Number of bars from sweep to reclaim. **OUTCOME ONLY - NEVER A FEATURE**. |
| `max_favorable_atr` | `float64` | $\ge 0.0$ | Max favorable excursion in ATR within horizon $m$ ($m=16$). **OUTCOME ONLY**. |
| `max_adverse_atr` | `float64` | $\ge 0.0$ | Max adverse excursion in ATR within horizon $m$ ($m=16$). **OUTCOME ONLY**. |
| `label` | `string` | `{"1", "0", "Breakdown", "Unresolved"}` | Target classification: |
| | | `1` | Qualifying stop hunt: swept, reclaimed, moved $+1.5 \times \text{ATR}$ before invalidation. |
| | | `0` | Eligible non-hunt: swept, reclaimed, but hit stop-loss or failed to reach target. |
| | | `Breakdown` | Swept beyond level without closing back inside range within $k$ bars. |
| | | `Unresolved` | Incomplete horizon at dataset boundary (excluded from training). |
