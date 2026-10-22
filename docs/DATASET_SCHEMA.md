# Dataset Schema & Data Dictionary

This document defines the raw OHLCV schema, engineered feature dictionary, and target label definitions for the ICT Liquidity Intelligence & Stop-Hunt Detection System.

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
| `instrument` | `string` | Categorical | Symbol ticker: `XAUUSD` (Gold) or `XAGUSD` (Silver). |
| `provider` | `string` | Metadata | Feed origin (e.g. `dukascopy`, `broker`, `yfinance`). |

---

## 2. Engineered Pre-Event Features Schema

These features are computed strictly using information available on or before the candidate sweep bar ($t_{sweep}$).

| Feature Name | Data Type | Scale / Range | Description |
| :--- | :--- | :--- | :--- |
| `instrument_code` | `int32` | {0, 1} | Binary symbol encoding: `1` = XAUUSD, `0` = XAGUSD. |
| `pattern_type_code` | `int32` | {0, 1, 2} | Categorical encoding: `0` = Double Bottom, `1` = Triple Bottom, `2` = Trendline. |
| `equality_score` | `float64` | $\ge 0.0$ | $\|L_1 - L_2\| / \text{ATR}$. Cleanliness of equal lows (lower is cleaner). |
| `pattern_height_atr` | `float64` | $\ge 1.5$ | Distance from neckline peak to lowest low, in ATR units. |
| `touch_count` | `int32` | $\ge 2$ | Number of confirmed swing touches forming the liquidity zone. |
| `zone_width_atr` | `float64` | $\ge 0.0$ | Range between highest touch low and lowest touch low, in ATR units. |
| `bar_separation` | `int32` | $\ge 5$ | Number of bars between consecutive touches. |
| `slope_atr` | `float64` | $[0.0, 1.5]$ | Fitted line slope in ATR per bar (0.0 for horizontal patterns). |
| `line_age` | `int32` | $\ge 0$ | Total bars from the pattern's first anchor to the sweep bar. |
| `sweep_depth_atr` | `float64` | $\ge 0.1$ | Penetration distance beyond the liquidity level: $(\text{Level} - \text{Low}) / \text{ATR}$. |
| `relative_volume_sweep` | `float64` | $\ge 0.0$ | Candle volume on the sweep bar relative to 20-period volume SMA. |
| `candle_lower_wick_ratio` | `float64` | $[0.0, 1.0]$ | $(\min(\text{Open}, \text{Close}) - \text{Low}) / (\text{High} - \text{Low})$. Lower rejection wick size. |
| `candle_body_ratio` | `float64` | $[0.0, 1.0]$ | $\|\text{Close} - \text{Open}\| / (\text{High} - \text{Low})$. Body proportion of the candle. |
| `candle_close_location` | `float64` | $[0.0, 1.0]$ | $(\text{Close} - \text{Low}) / (\text{High} - \text{Low})$. Position of close within candle range. |
| `dist_to_pdl_atr` | `float64` | Real | $(\text{Close} - \text{PDL}) / \text{ATR}$. Distance to Prior Day Low. |
| `dist_to_pdh_atr` | `float64` | Real | $(\text{PDH} - \text{Close}) / \text{ATR}$. Distance to Prior Day High. |
| `confluence_score` | `int32` | $\{0, 1, 2, 3+\}$ | Number of overlapping session extremes / daily levels within $0.5 \times \text{ATR}$. |
| `session_code` | `int32` | $\{0, 1, 2, 3, 4\}$ | Session at sweep: `0` = Asia, `1` = London, `2` = NYAM, `3` = NYPM, `4` = Other. |
| `rolling_volatility` | `float64` | $\ge 0.0$ | 20-bar standard deviation of log returns. |
| `return_5` | `float64` | Real | 5-bar cumulative percentage return prior to sweep. |
| `return_20` | `float64` | Real | 20-bar cumulative percentage return prior to sweep. |

---

## 3. Historical Outcome & Target Labels

| Field Name | Type | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `reclaimed` | `bool` | `{True, False}` | Whether price closed back above the liquidity level within $k$ bars ($k \le 3$). |
| `reclaim_speed_bars` | `int32` | $\{0, 1, 2, 3, -1\}$ | Number of bars from sweep to reclaim. **OUTCOME ONLY - NEVER A FEATURE**. |
| `max_favorable_atr` | `float64` | $\ge 0.0$ | Max favorable excursion in ATR within horizon $m$ ($m=16$). **OUTCOME ONLY**. |
| `max_adverse_atr` | `float64` | $\ge 0.0$ | Max adverse excursion in ATR within horizon $m$ ($m=16$). **OUTCOME ONLY**. |
| `label` | `string` | `{"1", "0", "Breakdown", "Unresolved"}` | Target classification: |
| | | `1` | Qualifying stop hunt: swept, reclaimed, moved $+1.5 \times \text{ATR}$ before invalidation. |
| | | `0` | Eligible non-hunt: swept, reclaimed, but hit stop-loss or failed to reach target. |
| | | `Breakdown` | Swept beyond level without closing back above within $k$ bars. |
| | | `Unresolved` | Incomplete horizon at dataset boundary (excluded from training). |
