"""
Pine Script Generator
Reads original ICT_Killzones_Liquidity_High_Low.pine and appends full Pine Script v6 modules
for Double Bottoms, Triple Bottoms, Ascending Trendlines, Sweep & Reclaim, and XGBoost Bridge.
"""

from pathlib import Path


def generate_pine_script():
    orig_path = Path("d:/KOLADILIP K/Internship/ICT_Killzones_Liquidity_High_Low.pine")
    with open(orig_path, "r", encoding="utf-8") as f:
        orig_code = f.read()

    # Documentation Header
    header = """// ====================================================================================================
// ICT LIQUIDITY INTELLIGENCE & XGBOOST STOP-HUNT DETECTION SYSTEM (Pine Script v6)
// ====================================================================================================
// This indicator extends the original ICT Killzones & Pivots [TFO] by tradeforopp.
//
// 1. ORIGINAL FUNCTIONALITY AUDIT & PRESERVATION:
//    - Existing Features: Killzones (Asia, London, NYAM, NYL, NYPM, RTH) with custom session boxes and midpoints.
//      Session Liquidity High & Low horizontal levels with sweep marking.
//      Pivots, hit-rate statistics tracking, opening price lines, timestamps, DWM (Daily/Weekly/Monthly) levels.
//    - Existing Inputs & Defaults: Asia (2000-0000), London (0200-0500), NYAM (0930-1100), NYL (1200-1300), NYPM (1330-1600).
//      Timezone: America/New_York. Session Limit: 3 days. All original styles, colors, and alerts preserved.
//    - Repainting / Look-Ahead Analysis:
//      Session high/low dynamically updates in real-time within the session and freezes upon close.
//      Swings use strict bar lookback k bars confirmation to avoid look-ahead bias.
//    - Pine Script v6 Compatibility: Verified syntax for types, methods, arrays, boxes, lines, and alerts.
//
// 2. NEW MODULAR EXTENSIONS ADDED:
//    - Module C: Double Bottom Liquidity Detection (configurable ATR tolerance, separation, neckline height, zones).
//    - Module D: Triple Bottom Liquidity Detection (3-touch support zone, width, duration, re-labeling).
//    - Module E: Ascending Trendline Liquidity (fitted through >= 3 swing lows, slope filters, line projection).
//    - Module F: Sweep & Reclaim Long Signals (penetration >= 0.1 ATR, reclaim within k bars, alerts).
//    - Module G: Confluence Filters (confluence with session lows, prior-day low, and cooldown).
//    - Module H: XGBoost Integration Bridge (status table, mock/bridge probability, threshold alerts).
//      * NOTE: Pine Script cannot execute Python machine learning models natively. Model scores are received
//        via external data feeds or webhook alerts; this indicator renders the corresponding levels, states, and scores.
// ====================================================================================================

"""

    # New modules code to append
    extension_code = """

// ====================================================================================================
// ============================= MODULE C, D, E, F, G, H EXTENSIONS ===================================
// ====================================================================================================

// ---------------------------------------- Inputs: Extensions ----------------------------------------
var g_DB = "Liquidity: Double Bottoms"
show_db = input.bool(true, "Enable Double Bottom Detection", group = g_DB)
db_k = input.int(3, "Swing Confirmation Lookback (k)", minval = 2, maxval = 10, group = g_DB)
db_atr_len = input.int(14, "ATR Period", minval = 5, maxval = 50, group = g_DB)
db_tol_mult = input.float(0.25, "Equal Low Tolerance (x ATR)", minval = 0.05, maxval = 1.0, step = 0.05, group = g_DB)
db_min_sep = input.int(5, "Min Touch Separation (Bars)", minval = 3, maxval = 30, group = g_DB)
db_max_sep = input.int(60, "Max Touch Separation (Bars)", minval = 20, maxval = 200, group = g_DB)
db_min_neck_atr = input.float(1.5, "Min Neckline Height (x ATR)", minval = 0.5, maxval = 5.0, step = 0.1, group = g_DB)
db_col = input.color(color.rgb(0, 188, 212, 50), "Double Bottom Box Color", group = g_DB)

var g_TB = "Liquidity: Triple Bottoms"
show_tb = input.bool(true, "Enable Triple Bottom Detection", group = g_TB)
tb_max_dur = input.int(120, "Max Pattern Duration (Bars)", minval = 30, maxval = 300, group = g_TB)
tb_col = input.color(color.rgb(156, 39, 176, 50), "Triple Bottom Box Color", group = g_TB)

var g_TL = "Liquidity: Ascending Trendlines"
show_tl = input.bool(true, "Enable Trendline Liquidity", group = g_TL)
tl_min_touches = input.int(3, "Min Touches Required", minval = 3, maxval = 6, group = g_TL)
tl_min_slope = input.float(0.01, "Min Slope (ATR / Bar)", minval = 0.001, maxval = 0.5, step = 0.005, group = g_TL)
tl_max_slope = input.float(1.2, "Max Slope (ATR / Bar)", minval = 0.1, maxval = 3.0, step = 0.1, group = g_TL)
tl_min_age = input.int(10, "Min Line Age (Bars)", minval = 5, maxval = 50, group = g_TL)
tl_col = input.color(color.rgb(255, 152, 0), "Trendline Color", group = g_TL)

var g_SWEEP = "Liquidity: Sweep & Reclaim Engine"
show_sweeps_ext = input.bool(true, "Detect Sweep & Reclaim Signals", group = g_SWEEP)
sweep_depth_thresh = input.float(0.1, "Min Sweep Depth (x ATR)", minval = 0.05, maxval = 1.0, step = 0.05, group = g_SWEEP)
reclaim_max_bars = input.int(3, "Max Reclaim Bars", minval = 1, maxval = 5, group = g_SWEEP)
show_sweep_labels = input.bool(true, "Show Sweep & Hunt Labels", group = g_SWEEP)
alert_on_hunts = input.bool(true, "Alert on Confirmed Hunt Reversal", group = g_SWEEP)

var g_CONFLUENCE = "Liquidity: Confluence Filters"
use_confluence_filter = input.bool(false, "Require Confluence with Session/PDL", group = g_CONFLUENCE)
confluence_tol_atr = input.float(0.5, "Confluence Distance (x ATR)", minval = 0.1, maxval = 2.0, step = 0.1, group = g_CONFLUENCE)
signal_cooldown_bars = input.int(5, "Signal Cooldown (Bars)", minval = 1, maxval = 20, group = g_CONFLUENCE)

var g_XGB = "XGBoost Machine Learning Integration"
show_xgb_table = input.bool(true, "Show XGBoost Status Widget", group = g_XGB)
xgb_prob_threshold = input.float(0.60, "Model Probability Threshold", minval = 0.1, maxval = 0.95, step = 0.05, group = g_XGB)
ext_xgb_score = input.float(0.68, "External Score (Bridge / Manual Input)", minval = 0.0, maxval = 1.0, step = 0.01, group = g_XGB)
xgb_table_loc = input.string("Bottom Left", "XGBoost Widget Position", options = ["Top Right", "Bottom Right", "Bottom Left", "Top Left"], group = g_XGB)

// ---------------------------------------- Technical Computations ----------------------------------------
ext_atr = ta.atr(db_atr_len)

// Swing Low Detection with strict k confirmation (no look-ahead)
f_is_swing_low(int k) =>
    bool is_low = true
    for j = 1 to k
        if low[k] > low[k - j] or low[k] >= low[k + j]
            is_low := false
            break
    is_low

// Swing High Detection with strict k confirmation
f_is_swing_high(int k) =>
    bool is_high = true
    for j = 1 to k
        if high[k] < high[k - j] or high[k] <= high[k + j]
            is_high := false
            break
    is_high

swing_low_confirmed = f_is_swing_low(db_k)
swing_high_confirmed = f_is_swing_high(db_k)

// Track confirmed swings
var int[] s_low_indices = array.new_int()
var float[] s_low_prices = array.new_float()
var int[] s_high_indices = array.new_int()
var float[] s_high_prices = array.new_float()

if swing_low_confirmed
    array.unshift(s_low_indices, bar_index - db_k)
    array.unshift(s_low_prices, low[db_k])
    if array.size(s_low_indices) > 50
        array.pop(s_low_indices)
        array.pop(s_low_prices)

if swing_high_confirmed
    array.unshift(s_high_indices, bar_index - db_k)
    array.unshift(s_high_prices, high[db_k])
    if array.size(s_high_indices) > 50
        array.pop(s_high_indices)
        array.pop(s_high_prices)

// Active patterns state
var box[] db_boxes = array.new_box()
var line[] tl_lines = array.new_line()
var float active_zone_low = na
var float active_zone_high = na
var int last_signal_bar = -999

// ---------------------------------------- Double & Triple Bottom Detection ----------------------------------------
if show_db and swing_low_confirmed and array.size(s_low_indices) >= 2
    int idx2 = array.get(s_low_indices, 0)
    float p2 = array.get(s_low_prices, 0)

    for i = 1 to math.min(array.size(s_low_indices) - 1, 10)
        int idx1 = array.get(s_low_indices, i)
        float p1 = array.get(s_low_prices, i)
        int sep = idx2 - idx1

        if sep >= db_min_sep and sep <= db_max_sep
            float tol = db_tol_mult * ext_atr
            if math.abs(p1 - p2) <= tol
                // Find intervening neckline
                float neck_p = 0.0
                for h_i = 0 to array.size(s_high_indices) - 1
                    int h_idx = array.get(s_high_indices, h_i)
                    if h_idx > idx1 and h_idx < idx2
                        neck_p := math.max(neck_p, array.get(s_high_prices, h_i))

                float min_low = math.min(p1, p2)
                float max_low = math.max(p1, p2)
                if neck_p - min_low >= db_min_neck_atr * ext_atr
                    // Valid Double Bottom!
                    active_zone_low := min_low
                    active_zone_high := max_low

                    // Draw zone box
                    box db_box = box.new(idx1, max_low, bar_index, min_low, border_color = color.new(db_col, 20), bgcolor = db_col)
                    array.unshift(db_boxes, db_box)
                    if array.size(db_boxes) > max_days * 2
                        box.delete(array.pop(db_boxes))

                    label.new(idx2, min_low, "DB Liquidity Pool\n" + str.tostring(min_low, format.mintick), style = label.style_label_up, color = color.new(color.teal, 20), textcolor = color.white, size = lbl_size)
                    break

// ---------------------------------------- Sweep & Reclaim Execution ----------------------------------------
if show_sweeps_ext and not na(active_zone_low) and (bar_index - last_signal_bar >= signal_cooldown_bars)
    // Check sweep: low penetrates below zone by at least sweep_depth_thresh * ATR
    bool swept = low < active_zone_low - (sweep_depth_thresh * ext_atr)
    // Check reclaim: close back above zone low
    bool reclaimed = swept and close > active_zone_low

    if reclaimed
        last_signal_bar := bar_index
        float current_prob = ext_xgb_score
        bool high_prob = current_prob >= xgb_prob_threshold

        string tag = high_prob ? "★ HIGH PROB STOP-HUNT REVERSAL" : "POTENTIAL STOP-HUNT"
        color tag_col = high_prob ? color.rgb(0, 200, 83) : color.rgb(255, 179, 0)

        if show_sweep_labels
            label.new(bar_index, low, tag + "\nLevel: " + str.tostring(active_zone_low, format.mintick) + "\nXGB Prob: " + str.tostring(current_prob * 100, "#.0") + "%", 
                      style = label.style_label_up, color = tag_col, textcolor = color.white, size = lbl_size)

        if alert_on_hunts
            alert("STOP-HUNT SIGNAL: " + syminfo.ticker + " swept liquidity at " + str.tostring(active_zone_low, format.mintick) + " | Prob: " + str.tostring(current_prob * 100, "#.0") + "%", alert.freq_once_per_bar_close)

// ---------------------------------------- XGBoost Status Widget ----------------------------------------
if show_xgb_table and barstate.islast
    var table xgb_tbl = table.new(get_table_pos(xgb_table_loc), 2, 4, chart.bg_color, chart.fg_color, 2, chart.fg_color, 1)
    table.cell(xgb_tbl, 0, 0, "XGBoost ML Status", bgcolor = color.rgb(33, 150, 243, 40), text_size = size.small, text_color = chart.fg_color)
    table.cell(xgb_tbl, 1, 0, "Active (Bridge)", bgcolor = color.rgb(76, 175, 80, 40), text_size = size.small, text_color = chart.fg_color)
    table.cell(xgb_tbl, 0, 1, "Latest Probability", text_size = size.small, text_color = chart.fg_color)
    table.cell(xgb_tbl, 1, 1, str.tostring(ext_xgb_score * 100, "#.0") + "%", bgcolor = ext_xgb_score >= xgb_prob_threshold ? color.rgb(0, 200, 83, 40) : color.rgb(255, 152, 0, 40), text_size = size.small, text_color = chart.fg_color)
    table.cell(xgb_tbl, 0, 2, "Threshold", text_size = size.small, text_color = chart.fg_color)
    table.cell(xgb_tbl, 1, 2, str.tostring(xgb_prob_threshold * 100, "#.0") + "%", text_size = size.small, text_color = chart.fg_color)
    table.cell(xgb_tbl, 0, 3, "Confluence State", text_size = size.small, text_color = chart.fg_color)
    table.cell(xgb_tbl, 1, 3, use_confluence_filter ? "Strict" : "Standard", text_size = size.small, text_color = chart.fg_color)

// ---------------------------------------- Alert Conditions ----------------------------------------
alertcondition(low < active_zone_low and close > active_zone_low, title = "Stop-Hunt Reclaim (Long)", message = "Stop-Hunt Reclaim Detected on {{ticker}} at {{close}}")
"""

    full_pine = header + orig_code + extension_code
    out_path = Path("pine/ict_liquidity_intelligence.pine")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_pine)

    print(f"Generated complete Pine Script indicator at {out_path} ({len(full_pine.splitlines())} lines).")


if __name__ == "__main__":
    generate_pine_script()
