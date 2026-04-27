"""
app/modules/volatility.py
=========================
Part A (Step 5) — Returns & Volatility Analysis

Pure functions only.  No matplotlib, no print(), no side effects.
All functions accept a pandas DataFrame (or Series) and return
plain Python dicts / lists that FastAPI can serialize directly.

DataFrame contract expected by most functions:
    index   : DatetimeIndex (trading days)
    Close   : float  — adjusted closing price
    Volume  : int    — daily volume
    (Log_Return, RolVol_20d are added by build_price_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import re

from app.utils.upstox_client import get_historical_candles


# ---------------------------------------------------------------------------
# 1.  Data acquisition
# ---------------------------------------------------------------------------

def fetch_price_df(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Download OHLCV from yfinance and return a clean DataFrame.

    Parameters
    ----------
    ticker : str  e.g. "RELIANCE.NS"
    period : str  yfinance period string  e.g. "6mo", "1y"

    Returns
    -------
    DataFrame with columns [Close, Volume], DatetimeIndex, no NaNs.

    Raises
    ------
    ValueError  if fewer than 60 rows were returned (likely a bad ticker).
    """
    period = (period or "6mo").strip().lower()
    day_match = re.fullmatch(r"(\d+)d", period)

    lookback_days = int(day_match.group(1)) if day_match else None

    response = get_historical_candles(ticker=ticker, period=period, interval="day")
    if response.get("status_code") == 401 or response.get("error") == "AUTH_REQUIRED":
        raise PermissionError("UPSTOX_AUTH_REQUIRED")
    if response.get("status") != "success":
        raise ValueError(response.get("message") or f"Failed to fetch historical candles for {ticker}")

    candles = response.get("data", [])
    if not candles:
        raise ValueError(f"No historical candles returned for {ticker}")

    df = pd.DataFrame(candles)
    if not {"date", "close", "volume"}.issubset(df.columns):
        raise ValueError(f"Historical candle payload missing required fields for {ticker}")

    df = pd.DataFrame(
        {
            "Close": pd.to_numeric(df["close"], errors="coerce"),
            "Volume": pd.to_numeric(df["volume"], errors="coerce"),
        },
        index=pd.to_datetime(df["date"], errors="coerce"),
    ).dropna()

    df.sort_index(inplace=True)

    if lookback_days is not None:
        if len(df) < lookback_days:
            raise ValueError(
                f"Only {len(df)} rows returned for {ticker}, fewer than requested {lookback_days} trading days."
            )
        df = df.tail(lookback_days)

    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} rows returned for {ticker}. "
            "Check the ticker or try a longer period."
        )

    return df


# ---------------------------------------------------------------------------
# 2.  Core return / volatility computations
# ---------------------------------------------------------------------------

def compute_log_returns(close: pd.Series) -> pd.Series:
    """
    Daily log returns:  r_t = ln(P_t / P_{t-1})

    Returns a Series aligned with `close`, first value is NaN (dropped by callers).
    """
    return np.log(close / close.shift(1))


def compute_rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling realised volatility, annualised by sqrt(252).

    Parameters
    ----------
    returns : daily log return Series
    window  : rolling window in trading days (default 20)
    """
    return returns.rolling(window).std() * np.sqrt(252)


def compute_vol_clustering(returns: pd.Series, lags: int = 20) -> dict:
    """
    Volatility clustering diagnostic via autocorrelation of squared returns.

    A significant ACF at lags 1-5 is the classic ARCH effect signature.

    Returns
    -------
    dict with keys:
        acf_squared_returns : list[float]  — ACF values at lags 1..`lags`
        ljung_box_stat      : float        — Ljung-Box Q statistic (lag 10)
        ljung_box_pvalue    : float        — p-value; < 0.05 → clustering present
        clustering_detected : bool
    """
    from statsmodels.stats.stattools import (
        durbin_watson,   # not used directly but keeps import light
    )
    from statsmodels.tsa.stattools import acf
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r2 = returns.dropna() ** 2

    # ACF of squared returns (lags 0..lags; drop lag 0 which is always 1)
    acf_vals = acf(r2, nlags=lags, fft=True)[1:]   # lags 1..lags

    lb = acorr_ljungbox(r2, lags=[10], return_df=True)
    lb_stat   = float(lb["lb_stat"].iloc[0])
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])

    return {
        "acf_squared_returns": [round(v, 6) for v in acf_vals.tolist()],
        "ljung_box_stat":      round(lb_stat,   4),
        "ljung_box_pvalue":    round(lb_pvalue, 4),
        "clustering_detected": lb_pvalue < 0.05,
    }


# ---------------------------------------------------------------------------
# 3.  Enriched DataFrame builder
# ---------------------------------------------------------------------------

def build_price_df(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    Add Log_Return and RolVol_20d columns to a raw price DataFrame.

    Drops leading NaN rows introduced by the rolling window.
    """
    out = df.copy()
    out["Log_Return"] = compute_log_returns(out["Close"])
    out["RolVol_20d"] = compute_rolling_vol(out["Log_Return"], window=vol_window)
    return out.dropna(subset=["Log_Return", "RolVol_20d"])


# ---------------------------------------------------------------------------
# 4.  Summary statistics
# ---------------------------------------------------------------------------

def vol_summary_stats(returns: pd.Series, ticker: str, n_trading_days: int | None = None) -> dict:
    """
    Compute the summary statistics table required by Part A.

    Returns a plain dict (JSON-serialisable).
    """
    from scipy import stats as sp_stats

    r = returns.dropna()
    ann_vol = r.std() * np.sqrt(252)

    return {
        "ticker":              ticker,
        "n_trading_days":      int(n_trading_days if n_trading_days is not None else len(r)),
        "mean_daily_return":   round(float(r.mean()),               6),
        "std_daily_return":    round(float(r.std()),                 6),
        "skewness":            round(float(sp_stats.skew(r)),        4),
        "excess_kurtosis":     round(float(sp_stats.kurtosis(r)),    4),
        "min_return":          round(float(r.min()),                  6),
        "max_return":          round(float(r.max()),                  6),
        "annualised_vol":      round(float(ann_vol),                  4),
        "annualised_vol_pct":  round(float(ann_vol * 100),            2),
    }


# ---------------------------------------------------------------------------
# 5.  Regime classification (normal vs high-vol)
# ---------------------------------------------------------------------------

def classify_vol_regime(
    df: pd.DataFrame,
    vol_col: str = "RolVol_20d",
    q: float = 0.75,
) -> pd.DataFrame:
    """
    Tag each row as "High-Vol" (top `q` quantile) or "Normal".

    Adds a column  Vol_Regime  to the returned DataFrame.
    The 'High-Vol' threshold is the top 25% of rolling vol days —
    consistent with Part D regime-split VaR.
    """
    threshold = df[vol_col].quantile(q)
    out = df.copy()
    out["Vol_Regime"] = np.where(
        out[vol_col] >= threshold, "High-Vol", "Normal"
    )
    return out


# ---------------------------------------------------------------------------
# 6.  Time-series payload (for charting on the frontend)
# ---------------------------------------------------------------------------

def vol_timeseries_payload(df: pd.DataFrame, ticker: str) -> dict:
    """
    Return chart-ready arrays for the frontend:
        dates, close, log_returns, rolling_vol_20d

    Dates are ISO-format strings.
    """
    return {
        "ticker": ticker,
        "dates":           df.index.strftime("%Y-%m-%d").tolist(),
        "close":           df["Close"].round(2).tolist(),
        "log_returns":     df["Log_Return"].round(6).tolist(),
        "rolling_vol_20d": df["RolVol_20d"].round(6).tolist(),
    }


# ---------------------------------------------------------------------------
# 7.  Top-level service function  (called by the FastAPI router)
# ---------------------------------------------------------------------------

def get_volatility_analysis(
    ticker: str,
    period: str = "6mo",
    vol_window: int = 20,
    acf_lags: int = 20,
) -> dict:
    """
    Full Part A volatility analysis for a single ticker.

    Returns
    -------
    {
        ticker          : str,
        period          : str,
        summary_stats   : dict,
        clustering      : dict,
        timeseries      : dict,        ← chart data
        regime_counts   : dict,        ← { "High-Vol": N, "Normal": N }
    }
    """
    df_raw = fetch_price_df(ticker, period=period)
    df     = build_price_df(df_raw, vol_window=vol_window)
    df     = classify_vol_regime(df)

    summary   = vol_summary_stats(df["Log_Return"], ticker, n_trading_days=len(df_raw))
    clustering = compute_vol_clustering(df["Log_Return"], lags=acf_lags)
    ts         = vol_timeseries_payload(df, ticker)

    regime_counts = df["Vol_Regime"].value_counts().to_dict()

    return {
        "ticker":        ticker,
        "period":        period,
        "summary_stats": summary,
        "clustering":    clustering,
        "timeseries":    ts,
        "regime_counts": regime_counts,
    }
