"""
Core Technical Indicators Module
Calculates ATR, normalized distances, wick/body ratios, returns, volatility, relative volume,
and daily reference levels (PDH, PDL) without look-ahead bias.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Computes True Range: max(high - low, abs(high - prev_close), abs(low - prev_close)).
    """
    prev_close = close.shift(1)
    hl = high - low
    hpc = (high - prev_close).abs()
    lpc = (low - prev_close).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    # First bar TR is simply high - low
    tr.iloc[0] = hl.iloc[0]
    return tr


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Wilder's Smoothed Average True Range (ATR).
    """
    tr = compute_true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    # Backfill warm-up period with expanding mean
    warmup = tr.expanding(min_periods=1).mean()
    atr = atr.fillna(warmup)
    return atr


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes candle anatomy ratios:
    - upper_wick_ratio
    - lower_wick_ratio
    - body_ratio
    - close_location (0.0 at low, 1.0 at high)
    """
    out = pd.DataFrame(index=df.index)
    candle_range = (df["high"] - df["low"]).clip(lower=1e-6)
    body_high = np.maximum(df["open"], df["close"])
    body_low = np.minimum(df["open"], df["close"])

    out["upper_wick_ratio"] = (df["high"] - body_high) / candle_range
    out["lower_wick_ratio"] = (body_low - df["low"]) / candle_range
    out["body_ratio"] = (body_high - body_low) / candle_range
    out["close_location"] = (df["close"] - df["low"]) / candle_range

    return out


def compute_returns_and_volatility(close: pd.Series, periods: List[int] = [1, 5, 20], vol_window: int = 20) -> pd.DataFrame:
    """
    Computes percentage returns over various horizons and rolling annualized/log volatility.
    """
    out = pd.DataFrame(index=close.index)
    for p in periods:
        out[f"return_{p}"] = close.pct_change(p).fillna(0.0)

    # Rolling standard deviation of 1-period log returns
    log_ret = np.log(close / close.shift(1).clip(lower=1e-6)).fillna(0.0)
    out["rolling_volatility"] = log_ret.rolling(window=vol_window, min_periods=5).std().fillna(0.0)
    return out


def compute_relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Computes relative volume: volume / SMA(volume, window).
    """
    sma = volume.rolling(window=window, min_periods=1).mean().clip(lower=1e-6)
    rvol = volume / sma
    return rvol.fillna(1.0)


def compute_previous_day_levels(df: pd.DataFrame, tz_name: str = "America/New_York") -> pd.DataFrame:
    """
    Computes Prior Day High (PDH), Prior Day Low (PDL), and Prior Day Close (PDC).
    
    CRITICAL NO-LOOK-AHEAD:
    The previous day's high/low/close are frozen at the end of the previous trading day
    and mapped only to the current trading day. Today's bars NEVER see today's eventual high/low.
    """
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts_local = ts.dt.tz_localize("UTC").dt.tz_convert(tz_name)
    else:
        ts_local = ts.dt.tz_convert(tz_name)

    dates = ts_local.dt.date

    # Daily aggregation
    daily_stats = df.groupby(dates).agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last")
    ).reset_index().rename(columns={"timestamp": "trading_date"})

    # Shift by 1 day so today gets yesterday's completed levels
    daily_stats["pdh"] = daily_stats["day_high"].shift(1)
    daily_stats["pdl"] = daily_stats["day_low"].shift(1)
    daily_stats["pdc"] = daily_stats["day_close"].shift(1)

    # Merge back to original dataframe by date
    merged = pd.merge(
        pd.DataFrame({"trading_date": dates}, index=df.index),
        daily_stats[["trading_date", "pdh", "pdl", "pdc"]],
        on="trading_date",
        how="left"
    )
    merged.index = df.index

    # Forward/backward fill any initial missing values
    merged["pdh"] = merged["pdh"].bfill()
    merged["pdl"] = merged["pdl"].bfill()
    merged["pdc"] = merged["pdc"].bfill()

    return merged[["pdh", "pdl", "pdc"]]


def compute_previous_week_levels(df: pd.DataFrame, tz_name: str = "America/New_York") -> pd.DataFrame:
    """
    Computes Prior Week High (PWH), Prior Week Low (PWL), and Prior Week Close (PWC).
    Refreshed at weekly open; strictly shifted by 1 week to eliminate look-ahead.
    """
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts_local = ts.dt.tz_localize("UTC").dt.tz_convert(tz_name)
    else:
        ts_local = ts.dt.tz_convert(tz_name)

    # ISO Year-Week string (e.g. '2025-01')
    week_keys = ts_local.dt.strftime("%G-%V")

    weekly_stats = df.groupby(week_keys).agg(
        w_high=("high", "max"),
        w_low=("low", "min"),
        w_close=("close", "last")
    ).reset_index().rename(columns={"timestamp": "week_period"})

    weekly_stats["pwh"] = weekly_stats["w_high"].shift(1)
    weekly_stats["pwl"] = weekly_stats["w_low"].shift(1)
    weekly_stats["pwc"] = weekly_stats["w_close"].shift(1)

    merged = pd.merge(
        pd.DataFrame({"week_period": week_keys}, index=df.index),
        weekly_stats[["week_period", "pwh", "pwl", "pwc"]],
        on="week_period",
        how="left"
    )
    merged.index = df.index

    merged["pwh"] = merged["pwh"].bfill()
    merged["pwl"] = merged["pwl"].bfill()
    merged["pwc"] = merged["pwc"].bfill()

    return merged[["pwh", "pwl", "pwc"]]


def compute_previous_month_levels(df: pd.DataFrame, tz_name: str = "America/New_York") -> pd.DataFrame:
    """
    Computes Prior Month High (PMH), Prior Month Low (PML), and Prior Month Close (PMC).
    Refreshed on the first trading day of the month; strictly shifted by 1 month to eliminate look-ahead.
    """
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts_local = ts.dt.tz_localize("UTC").dt.tz_convert(tz_name)
    else:
        ts_local = ts.dt.tz_convert(tz_name)

    # Year-Month string (e.g. '2025-01')
    month_keys = ts_local.dt.strftime("%Y-%m")

    monthly_stats = df.groupby(month_keys).agg(
        m_high=("high", "max"),
        m_low=("low", "min"),
        m_close=("close", "last")
    ).reset_index().rename(columns={"timestamp": "month_period"})

    monthly_stats["pmh"] = monthly_stats["m_high"].shift(1)
    monthly_stats["pml"] = monthly_stats["m_low"].shift(1)
    monthly_stats["pmc"] = monthly_stats["m_close"].shift(1)

    merged = pd.merge(
        pd.DataFrame({"month_period": month_keys}, index=df.index),
        monthly_stats[["month_period", "pmh", "pml", "pmc"]],
        on="month_period",
        how="left"
    )
    merged.index = df.index

    merged["pmh"] = merged["pmh"].bfill()
    merged["pml"] = merged["pml"].bfill()
    merged["pmc"] = merged["pmc"].bfill()

    return merged[["pmh", "pml", "pmc"]]


def add_core_indicators(df: pd.DataFrame, atr_period: int = 14, tz_name: str = "America/New_York") -> pd.DataFrame:
    """
    Enriches an OHLCV dataframe with all core market indicators and HTF liquidity points:
    ATR, anatomy ratios, returns, volatility, volume, and PDH/PDL, PWH/PWL, PMH/PML.
    """
    df_out = df.copy()

    # 1. ATR
    df_out["atr"] = compute_atr(df_out["high"], df_out["low"], df_out["close"], period=atr_period)

    # 2. Candle anatomy
    candle_feats = compute_candle_features(df_out)
    for col in candle_feats.columns:
        df_out[col] = candle_feats[col]

    # 3. Returns and Volatility
    ret_vol = compute_returns_and_volatility(df_out["close"])
    for col in ret_vol.columns:
        df_out[col] = ret_vol[col]

    # 4. Relative volume
    df_out["relative_volume"] = compute_relative_volume(df_out["volume"])

    # 5. Prior Day Levels (PDH, PDL, PDC)
    pdl_df = compute_previous_day_levels(df_out, tz_name=tz_name)
    df_out["pdh"] = pdl_df["pdh"]
    df_out["pdl"] = pdl_df["pdl"]
    df_out["pdc"] = pdl_df["pdc"]

    # 6. Prior Week Levels (PWH, PWL, PWC)
    pwl_df = compute_previous_week_levels(df_out, tz_name=tz_name)
    df_out["pwh"] = pwl_df["pwh"]
    df_out["pwl"] = pwl_df["pwl"]
    df_out["pwc"] = pwl_df["pwc"]

    # 7. Prior Month Levels (PMH, PML, PMC)
    pml_df = compute_previous_month_levels(df_out, tz_name=tz_name)
    df_out["pmh"] = pml_df["pmh"]
    df_out["pml"] = pml_df["pml"]
    df_out["pmc"] = pml_df["pmc"]

    # 8. Distances to HTF Liquidity Points in ATR units
    atr_safe = df_out["atr"].clip(lower=1e-6)
    df_out["dist_to_pdh_atr"] = (df_out["pdh"] - df_out["close"]) / atr_safe
    df_out["dist_to_pdl_atr"] = (df_out["close"] - df_out["pdl"]) / atr_safe
    df_out["dist_to_pwh_atr"] = (df_out["pwh"] - df_out["close"]) / atr_safe
    df_out["dist_to_pwl_atr"] = (df_out["close"] - df_out["pwl"]) / atr_safe
    df_out["dist_to_pmh_atr"] = (df_out["pmh"] - df_out["close"]) / atr_safe
    df_out["dist_to_pml_atr"] = (df_out["close"] - df_out["pml"]) / atr_safe

    return df_out
