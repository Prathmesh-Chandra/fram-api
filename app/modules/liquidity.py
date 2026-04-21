"""
app/modules/liquidity.py
========================
Part A (Step 6) — Liquidity Analysis

Pure functions only.  No matplotlib, no print(), no side effects.

Two liquidity proxies implemented:
    1. Turnover Ratio  — relative daily turnover vs 60-day rolling mean
    2. Amihud (2002)   — |R_t| / (Close_t × Volume_t)

Also provides:
    - vol_liquidity_correlation()  — Pearson corr matrix (Part A requirement)
    - full service function get_liquidity_analysis()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  Raw liquidity metrics
# ---------------------------------------------------------------------------

def compute_turnover_inr(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Daily traded value in INR:  Turnover = Close × Volume
    """
    return close * volume


def compute_turnover_ratio(
    turnover_inr: pd.Series,
    rolling_window: int = 60,
) -> pd.Series:
    """
    Relative turnover ratio = daily_turnover / rolling_mean(turnover, 60d)

    Values > 1 → above-average liquidity day.
    Values < 1 → below-average liquidity day.
    """
    rolling_mean = turnover_inr.rolling(rolling_window).mean()
    return (turnover_inr / rolling_mean).replace([np.inf, -np.inf], np.nan)


def compute_amihud(
    log_returns: pd.Series,
    turnover_inr: pd.Series,
    ma_window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Amihud (2002) daily illiquidity ratio:
        ILLIQ_t = |R_t| / (Close_t × Volume_t)

    Higher values → price impact per unit of volume is larger → more illiquid.

    Returns
    -------
    amihud_raw  : daily raw ratio  (very small numbers, scale ×10^7 for display)
    amihud_ma   : 20-day rolling mean  (smoother; used in Part A plots)
    """
    amihud_raw = (
        log_returns.abs() / turnover_inr
    ).replace([np.inf, -np.inf], np.nan)

    amihud_ma = amihud_raw.rolling(ma_window).mean()
    return amihud_raw, amihud_ma


# ---------------------------------------------------------------------------
# 2.  Enriched DataFrame builder
# ---------------------------------------------------------------------------

def build_liquidity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with at least:  Close, Volume, Log_Return

    Adds columns:
        Turnover_INR    — daily traded value
        Turnover_Ratio  — relative turnover (60d window)
        Amihud_Raw      — daily Amihud illiquidity
        Amihud_MA       — 20-day rolling Amihud
        Liq_Class       — "High (top 25%)" / "Mid" / "Low (bottom 25%)"

    Drops rows where any liquidity metric is NaN.
    """
    out = df.copy()

    out["Turnover_INR"]   = compute_turnover_inr(out["Close"], out["Volume"])
    out["Turnover_Ratio"] = compute_turnover_ratio(out["Turnover_INR"])

    out["Amihud_Raw"], out["Amihud_MA"] = compute_amihud(
        out["Log_Return"], out["Turnover_INR"]
    )

    out = out.dropna(subset=["Turnover_Ratio", "Amihud_MA"])
    out = _classify_liquidity(out)

    return out


def _classify_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row by where its Turnover_Ratio sits in the distribution.

    "High (top 25%)"   — above 75th percentile
    "Low (bottom 25%)" — below 25th percentile
    "Mid"              — everything else
    """
    q75 = df["Turnover_Ratio"].quantile(0.75)
    q25 = df["Turnover_Ratio"].quantile(0.25)
    out = df.copy()
    out["Liq_Class"] = "Mid"
    out.loc[out["Turnover_Ratio"] >= q75, "Liq_Class"] = "High (top 25%)"
    out.loc[out["Turnover_Ratio"] <= q25, "Liq_Class"] = "Low (bottom 25%)"
    return out


# ---------------------------------------------------------------------------
# 3.  Summary statistics
# ---------------------------------------------------------------------------

def liquidity_summary_stats(df: pd.DataFrame, ticker: str) -> dict:
    """
    Key liquidity statistics for the Part A summary table.
    """
    avg_turnover_cr   = float(df["Turnover_INR"].mean() / 1e7)   # Crores
    avg_amihud        = float(df["Amihud_Raw"].mean())
    avg_amihud_scaled = avg_amihud * 1e7                           # scaled for display

    liq_counts = df["Liq_Class"].value_counts().to_dict()

    return {
        "ticker":                        ticker,
        "avg_daily_turnover_cr":         round(avg_turnover_cr,   2),
        "avg_daily_turnover_inr":        round(avg_turnover_cr * 1e7, 0),
        "avg_amihud_raw":                round(avg_amihud,        10),
        "avg_amihud_scaled_1e7":         round(avg_amihud_scaled,  6),
        "turnover_ratio_mean":           round(float(df["Turnover_Ratio"].mean()), 4),
        "turnover_ratio_std":            round(float(df["Turnover_Ratio"].std()),  4),
        "liq_class_counts":              liq_counts,
    }


# ---------------------------------------------------------------------------
# 4.  Vol-liquidity correlation  (Part A requirement)
# ---------------------------------------------------------------------------

def vol_liquidity_correlation(df: pd.DataFrame, ticker: str) -> dict:
    """
    Pearson correlation matrix between:
        RolVol_20d  ↔  Turnover_Ratio  ↔  Amihud_MA

    This is the quantitative evidence for the relationship the assignment asks
    you to discuss.

    Returns
    -------
    {
        ticker      : str,
        corr_matrix : { "RolVol_20d": {"Turnover_Ratio": x, "Amihud_MA": y}, ... }
        interpretation : {
            vol_vs_turnover  : str,   ← "positive" | "negative" | "negligible"
            vol_vs_amihud    : str,
        }
    }
    """
    cols   = ["RolVol_20d", "Turnover_Ratio", "Amihud_MA"]
    merged = df[cols].dropna()
    corr   = merged.corr().round(4)

    def _label(r: float) -> str:
        if r > 0.3:   return "positive"
        if r < -0.3:  return "negative"
        return "negligible"

    r_vol_tr  = corr.loc["RolVol_20d", "Turnover_Ratio"]
    r_vol_ami = corr.loc["RolVol_20d", "Amihud_MA"]

    return {
        "ticker":      ticker,
        "n_obs":       len(merged),
        "corr_matrix": corr.to_dict(),
        "interpretation": {
            "vol_vs_turnover_ratio": _label(r_vol_tr),
            "vol_vs_amihud":         _label(r_vol_ami),
            "vol_vs_turnover_r":     round(float(r_vol_tr),  4),
            "vol_vs_amihud_r":       round(float(r_vol_ami), 4),
        },
    }


# ---------------------------------------------------------------------------
# 5.  Time-series payload  (for charting on the frontend)
# ---------------------------------------------------------------------------

def liquidity_timeseries_payload(df: pd.DataFrame, ticker: str) -> dict:
    """
    Chart-ready arrays:
        dates, turnover_inr_cr, turnover_ratio, amihud_ma_scaled
    """
    return {
        "ticker":                  ticker,
        "dates":                   df.index.strftime("%Y-%m-%d").tolist(),
        "turnover_inr_cr":         (df["Turnover_INR"] / 1e7).round(4).tolist(),
        "turnover_ratio":          df["Turnover_Ratio"].round(6).tolist(),
        "amihud_ma_scaled_1e7":    (df["Amihud_MA"] * 1e7).round(6).tolist(),
        "liq_class":               df["Liq_Class"].tolist(),
    }


# ---------------------------------------------------------------------------
# 6.  NIFTY 50 universe — stock selection helper
# ---------------------------------------------------------------------------

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "TECHM.NS", "NTPC.NS", "ONGC.NS",
    "M&M.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "BAJAJFINSV.NS",
    "GRASIM.NS", "BPCL.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "INDUSINDBK.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BRITANNIA.NS",
    "BAJAJ-AUTO.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "LTIM.NS",
    "UPL.NS", "SHRIRAMFIN.NS",
]


def rank_universe_by_turnover(period: str = "6mo") -> dict:
    """
    Download all NIFTY 50 stocks, compute average daily turnover,
    and return the top-25% (liquid) and bottom-25% (illiquid) candidates.

    Used to populate the /data/universe/classified endpoint.

    Returns
    -------
    {
        liquid   : [{ ticker, avg_turnover_cr, rank }, ...]   ← top 25%
        illiquid : [{ ticker, avg_turnover_cr, rank }, ...]   ← bottom 25%
        all      : [{ ticker, avg_turnover_cr, rank }, ...]   ← full ranked list
    }
    """
    raw = yf.download(
        NIFTY50_TICKERS,
        period=period,
        auto_adjust=True,
        progress=False,
    )

    close_df  = raw["Close"].dropna(axis=1, how="all")
    volume_df = raw["Volume"].dropna(axis=1, how="all")
    common    = close_df.columns.intersection(volume_df.columns)

    # Filter tickers with at least 60 rows
    valid = [
        t for t in common
        if close_df[t].dropna().shape[0] >= 60
        and volume_df[t].dropna().shape[0] >= 60
    ]

    turnover_avg = (close_df[valid] * volume_df[valid]).mean().sort_values(ascending=False)

    q75 = turnover_avg.quantile(0.75)
    q25 = turnover_avg.quantile(0.25)

    ranked = [
        {
            "ticker":          str(t),
            "avg_turnover_cr": round(float(v) / 1e7, 2),
            "rank":            int(i + 1),
        }
        for i, (t, v) in enumerate(turnover_avg.items())
    ]

    return {
        "liquid":   [r for r in ranked if turnover_avg[r["ticker"]] >= q75],
        "illiquid": [r for r in ranked if turnover_avg[r["ticker"]] <= q25],
        "all":      ranked,
    }


# ---------------------------------------------------------------------------
# 7.  Top-level service function  (called by the FastAPI router)
# ---------------------------------------------------------------------------

def get_liquidity_analysis(
    ticker: str,
    period: str = "6mo",
    log_returns: Optional[pd.Series] = None,
) -> dict:
    """
    Full Part A liquidity analysis for a single ticker.

    Parameters
    ----------
    ticker       : e.g. "RELIANCE.NS"
    period       : yfinance period string
    log_returns  : optional pre-computed returns Series (avoids a second download
                   if volatility analysis was already run for the same ticker)

    Returns
    -------
    {
        ticker         : str,
        period         : str,
        summary_stats  : dict,
        correlation    : dict,
        timeseries     : dict,
    }
    """
    # Import here to avoid circular imports when both modules are used together
    from app.modules.volatility import fetch_price_df, build_price_df

    df_raw = fetch_price_df(ticker, period=period)
    df     = build_price_df(df_raw)                  # adds Log_Return, RolVol_20d
    df     = build_liquidity_df(df)                  # adds Turnover_*, Amihud_*

    summary     = liquidity_summary_stats(df, ticker)
    correlation = vol_liquidity_correlation(df, ticker)
    ts          = liquidity_timeseries_payload(df, ticker)

    return {
        "ticker":        ticker,
        "period":        period,
        "summary_stats": summary,
        "correlation":   correlation,
        "timeseries":    ts,
    }
