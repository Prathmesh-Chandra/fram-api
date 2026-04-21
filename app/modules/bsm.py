"""
app/modules/bsm.py
==================
Part B (Step 7) — Black-Scholes-Merton Pricing & Greeks

Pure functions only.  No prints, no side-effects, no matplotlib.

Covers
------
1.  Core BSM math  (d1/d2, price, Greeks, Theta)
2.  Implied volatility solver  (Newton-Raphson)
3.  Upstox option-chain filtering
    - Select expiries closest to 30d / 60d target maturities
    - Select ATM, OTM-Call (5-10%), OTM-Put (5-10%) legs
    - Extract real market prices (LTP → mid fallback)
4.  Part B table builder
    - BSM price vs market price side-by-side
    - Market IV extraction
    - Price deviation metrics

Constants
---------
RISK_FREE_RATE  : 0.068  (RBI repo rate proxy — adjust in .env ideally)
TRADING_DAYS    : 252
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import date, datetime
from typing import Literal, Optional
import warnings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_FREE_RATE = 0.068    # RBI 6-month T-bill / repo rate proxy
TRADING_DAYS   = 252


# ---------------------------------------------------------------------------
# 1.  Core BSM formulas  (verified against fram_project.py)
# ---------------------------------------------------------------------------

def bsm_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """
    Compute d1 and d2 for the Black-Scholes-Merton model.

    Parameters
    ----------
    S     : current spot price
    K     : strike price
    T     : time to expiry in years  (e.g. 30/252)
    r     : continuously compounded risk-free rate (decimal)
    sigma : annualised volatility (decimal)
    """
    if T <= 0 or sigma <= 0:
        raise ValueError(f"T and sigma must be positive. Got T={T}, sigma={sigma}")

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Black-Scholes-Merton European option price.

    Returns theoretical fair value in the same currency as S.
    """
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma)

    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """
    Compute Delta, Gamma, Vega, and Theta for a European option.

    Conventions (matching fram_project.py + Part C requirements):
    - Vega   : per 1% change in sigma  (divide S·N'(d1)·√T by 100)
    - Theta  : daily decay in price units  (divide annual theta by 252)
    - Delta  : call ∈ (0,1), put ∈ (-1,0)
    - Gamma  : always positive for both calls and puts

    Returns a dict with keys: delta, gamma, vega, theta
    """
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma)

    # Delta
    delta_call = float(norm.cdf(d1))
    delta      = delta_call if option_type.lower() == "call" else delta_call - 1.0

    # Gamma (same for call and put)
    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    # Vega per 1% change in sigma (same for call and put)
    vega = float(S * norm.pdf(d1) * np.sqrt(T) / 100)

    # Theta (annual → daily, per calendar convention use 365; per trading use 252)
    # We use 252 to stay consistent with the rest of the project
    if option_type.lower() == "call":
        theta_annual = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
    else:
        theta_annual = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

    theta_daily = float(theta_annual / TRADING_DAYS)

    return {
        "delta": round(delta,       6),
        "gamma": round(gamma,       8),
        "vega":  round(vega,        6),
        "theta": round(theta_daily, 6),
    }


# ---------------------------------------------------------------------------
# 2.  Implied Volatility solver  (Newton-Raphson with Brent fallback)
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Optional[float]:
    """
    Solve for the implied volatility that equates BSM price to market price.

    Algorithm: Newton-Raphson (fast convergence near ATM);
               falls back to Brent's method if NR diverges.

    Returns
    -------
    sigma_iv : float  — annualised implied volatility (decimal), or None if
                        convergence fails (deep ITM/OTM, zero price, etc.)
    """
    if market_price <= 0 or T <= 0:
        return None

    # Intrinsic value check — market price must exceed intrinsic
    intrinsic = max(0.0,
        (S - K) if option_type.lower() == "call" else (K - S)
    ) * np.exp(-r * T)
    if market_price < intrinsic - tol:
        return None

    def objective(sigma: float) -> float:
        try:
            return bsm_price(S, K, T, r, sigma, option_type) - market_price
        except (ValueError, FloatingPointError):
            return float("nan")

    # Newton-Raphson starting from a reasonable initial guess
    sigma = 0.25   # 25% vol is a sensible starting point for NIFTY 50 stocks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(max_iter):
            price_now = bsm_price(S, K, T, r, sigma, option_type)
            vega_now  = S * norm.pdf(
                bsm_d1_d2(S, K, T, r, sigma)[0]
            ) * np.sqrt(T)                        # raw vega (not per 1%)

            if abs(vega_now) < 1e-10:
                break
            sigma_new = sigma - (price_now - market_price) / vega_now
            if sigma_new <= 0:
                break
            if abs(sigma_new - sigma) < tol:
                return round(float(sigma_new), 6)
            sigma = sigma_new

        # Brent fallback over [0.01, 5.0]
        try:
            return round(float(brentq(objective, 0.005, 5.0, xtol=tol)), 6)
        except (ValueError, RuntimeError):
            return None


# ---------------------------------------------------------------------------
# 3.  Option chain filtering  (Upstox chain data → BSM input legs)
# ---------------------------------------------------------------------------

def select_expiries(
    available_expiries: list[str],
    target_days: list[int] = (30, 60),
    reference_date: Optional[date] = None,
) -> dict[int, str]:
    """
    From a list of expiry date strings, pick the expiry closest to each
    target maturity (in calendar days from reference_date).

    Parameters
    ----------
    available_expiries : list of "YYYY-MM-DD" strings from the chain API
    target_days        : e.g. [30, 60]
    reference_date     : defaults to today

    Returns
    -------
    { 30: "2026-05-26", 60: "2026-06-30" }
    """
    if reference_date is None:
        reference_date = date.today()

    expiry_dates = sorted([
        datetime.strptime(e, "%Y-%m-%d").date()
        for e in available_expiries
        # Only consider future expiries (at least 5 calendar days out)
        if datetime.strptime(e, "%Y-%m-%d").date() > reference_date + \
           __import__("datetime").timedelta(days=4)
    ])

    result = {}
    for target in target_days:
        if not expiry_dates:
            break
        best = min(expiry_dates, key=lambda d: abs((d - reference_date).days - target))
        result[target] = best.strftime("%Y-%m-%d")

    return result


def _market_price(option_side: dict) -> float:
    """
    Extract a usable market price from one side of a chain row.

    Priority:
      1. LTP (last traded price)  — if > 0
      2. Mid = (bid + ask) / 2   — if both are > 0
      3. Close price              — as a last resort

    Returns 0.0 if no price can be determined.
    """
    md = option_side.get("market_data", {})
    ltp   = float(md.get("ltp",         0) or 0)
    bid   = float(md.get("bid_price",   0) or 0)
    ask   = float(md.get("ask_price",   0) or 0)
    close = float(md.get("close_price", 0) or 0)

    if ltp > 0:
        return ltp
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2, 2)
    if close > 0:
        return close
    return 0.0


def _is_tradeable(option_side: dict) -> bool:
    """
    Return False for deep ITM / illiquid strikes where Upstox sets
    delta = ±1, iv = 0  (a sentinel for 'effectively no option value').
    These strikes are useless for BSM comparison.
    """
    greeks = option_side.get("option_greeks", {})
    iv    = float(greeks.get("iv",    0) or 0)
    delta = float(greeks.get("delta", 0) or 0)
    # Upstox sentinel: iv==0 AND |delta| exactly 1.0
    if iv == 0 and abs(abs(delta) - 1.0) < 1e-6:
        return False
    return True


def filter_chain_legs(
    chain_rows: list[dict],
    spot: float,
    otm_band: tuple[float, float] = (0.05, 0.10),
) -> dict:
    """
    From a list of chain rows (all same expiry), select the three legs
    required by Part B:

        ATM     — strike closest to spot
        OTM-Call — strike in (spot × (1 + low_band), spot × (1 + high_band))
        OTM-Put  — strike in (spot × (1 - high_band), spot × (1 - low_band))

    Tradeable-strike filter applied before selection.

    Parameters
    ----------
    chain_rows  : list of dicts as returned by /data/option-chain
    spot        : underlying_spot_price
    otm_band    : (low_pct, high_pct) defining the OTM range  (default 5-10%)

    Returns
    -------
    {
        "atm":      { strike, call_market_price, put_market_price,
                      call_iv, put_iv, days_to_expiry, expiry },
        "otm_call": { same keys, only call_market_price is meaningful },
        "otm_put":  { same keys, only put_market_price is meaningful },
    }
    or None if any leg cannot be found.
    """
    low, high = otm_band

    otm_call_lo, otm_call_hi = spot * (1 + low),  spot * (1 + high)
    otm_put_lo,  otm_put_hi  = spot * (1 - high), spot * (1 - low)

    # Filter to tradeable rows only
    tradeable = [
        row for row in chain_rows
        if _is_tradeable(row.get("call_options", {}))
        or _is_tradeable(row.get("put_options",  {}))
    ]
    if not tradeable:
        return None

    strikes = [float(r["strike_price"]) for r in tradeable]

    def _row_for_strike(k: float) -> Optional[dict]:
        for row in tradeable:
            if abs(float(row["strike_price"]) - k) < 1e-4:
                return row
        return None

    def _extract(row: dict) -> dict:
        expiry  = row.get("expiry", "")
        sp      = float(row["strike_price"])
        c_price = _market_price(row.get("call_options", {}))
        p_price = _market_price(row.get("put_options",  {}))
        c_iv    = float((row.get("call_options") or {}).get("option_greeks", {}).get("iv", 0) or 0)
        p_iv    = float((row.get("put_options")  or {}).get("option_greeks", {}).get("iv", 0) or 0)
        return {
            "strike":             round(sp, 2),
            "expiry":             expiry,
            "call_market_price":  round(c_price, 2),
            "put_market_price":   round(p_price, 2),
            "call_iv_upstox_pct": round(c_iv,    2),
            "put_iv_upstox_pct":  round(p_iv,    2),
        }

    # ATM — closest to spot
    atm_strike = min(strikes, key=lambda k: abs(k - spot))
    atm_row    = _row_for_strike(atm_strike)

    # OTM Call — pick the strike with best (closest to midpoint of band) among candidates
    call_cands = [k for k in strikes if otm_call_lo <= k <= otm_call_hi]
    if not call_cands:
        # relax: closest above spot in top 15%
        call_cands = [k for k in strikes if spot < k <= spot * 1.15]
    if not call_cands:
        return None
    otm_call_target = spot * (1 + (low + high) / 2)
    otm_call_strike = min(call_cands, key=lambda k: abs(k - otm_call_target))
    otm_call_row    = _row_for_strike(otm_call_strike)

    # OTM Put — pick the strike with best (closest to midpoint of band) among candidates
    put_cands = [k for k in strikes if otm_put_lo <= k <= otm_put_hi]
    if not put_cands:
        put_cands = [k for k in strikes if spot * 0.85 <= k < spot]
    if not put_cands:
        return None
    otm_put_target = spot * (1 - (low + high) / 2)
    otm_put_strike = min(put_cands, key=lambda k: abs(k - otm_put_target))
    otm_put_row    = _row_for_strike(otm_put_strike)

    if not (atm_row and otm_call_row and otm_put_row):
        return None

    return {
        "atm":      _extract(atm_row),
        "otm_call": _extract(otm_call_row),
        "otm_put":  _extract(otm_put_row),
    }


# ---------------------------------------------------------------------------
# 4.  Part B table builder
# ---------------------------------------------------------------------------

def build_part_b_table(
    ticker: str,
    expiry_chains: dict[int, dict],      # { 30: { "legs": {...}, "chain_rows": [...] } }
    spot: float,
    hist_vol: float,
    r: float = RISK_FREE_RATE,
    reference_date: Optional[date] = None,
) -> list[dict]:
    """
    Build the full Part B pricing comparison table for one ticker.

    Parameters
    ----------
    ticker         : e.g. "RELIANCE.NS"
    expiry_chains  : { target_days: { "legs": filter_chain_legs output,
                                       "expiry_date": "YYYY-MM-DD" } }
    spot           : current spot price
    hist_vol       : annualised historical volatility (decimal) from Part A
    r              : risk-free rate (default RISK_FREE_RATE)
    reference_date : date from which to measure T  (defaults to today)

    Returns
    -------
    list of dicts — one row per (maturity, moneyness, option_type) combination.
    Columns:
        ticker, target_maturity_days, actual_expiry, resolved_T_years,
        moneyness, option_type, spot, strike,
        market_price, bsm_price_hist, price_deviation, price_deviation_pct,
        delta, gamma, vega, theta,
        market_iv_pct, hist_vol_pct,
        moneyness_pct (strike / spot - 1)
    """
    if reference_date is None:
        reference_date = date.today()

    rows = []

    for target_days, chain_info in sorted(expiry_chains.items()):
        legs        = chain_info["legs"]
        expiry_str  = chain_info["expiry_date"]

        if not legs:
            continue

        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        T_cal       = (expiry_date - reference_date).days          # calendar days
        T_years     = T_cal / 252                                  # trading-day convention

        if T_years <= 0:
            continue

        # Three legs per maturity: ATM call + put, OTM call, OTM put
        for leg_key, opt_type, price_key in [
            ("atm",      "call", "call_market_price"),
            ("atm",      "put",  "put_market_price"),
            ("otm_call", "call", "call_market_price"),
            ("otm_put",  "put",  "put_market_price"),
        ]:
            leg = legs.get(leg_key)
            if not leg:
                continue

            K            = leg["strike"]
            market_price = leg[price_key]

            # BSM fair value with historical vol
            try:
                bsm_p = round(bsm_price(spot, K, T_years, r, hist_vol, opt_type), 4)
            except Exception:
                bsm_p = None

            # Greeks
            try:
                greeks = bsm_greeks(spot, K, T_years, r, hist_vol, opt_type)
            except Exception:
                greeks = {"delta": None, "gamma": None, "vega": None, "theta": None}

            # Price deviation: market - BSM  (positive → market more expensive)
            dev     = round(market_price - bsm_p, 4) if (bsm_p and market_price > 0) else None
            dev_pct = round((dev / bsm_p) * 100, 2)  if (dev is not None and bsm_p and bsm_p > 0) else None

            # Market IV (from Upstox greeks; divide by 100 to get decimal)
            iv_key   = "call_iv_upstox_pct" if opt_type == "call" else "put_iv_upstox_pct"
            mkt_iv   = leg.get(iv_key, 0)

            # Moneyness label
            if leg_key == "atm":
                moneyness_label = "ATM"
            elif leg_key == "otm_call":
                moneyness_label = "OTM Call"
            else:
                moneyness_label = "OTM Put"

            rows.append({
                "ticker":                  ticker,
                "target_maturity_days":    target_days,
                "actual_expiry":           expiry_str,
                "T_calendar_days":         T_cal,
                "T_years":                 round(T_years, 4),
                "moneyness":               moneyness_label,
                "option_type":             opt_type.upper(),
                "spot":                    round(spot, 2),
                "strike":                  K,
                "moneyness_pct":           round((K / spot - 1) * 100, 2),
                "market_price":            market_price,
                "bsm_price_hist_vol":      bsm_p,
                "price_deviation":         dev,
                "price_deviation_pct":     dev_pct,
                "hist_vol_pct":            round(hist_vol * 100, 4),
                "market_iv_pct":           mkt_iv,
                "iv_spread_pct":           round(mkt_iv - hist_vol * 100, 4) if mkt_iv else None,
                "delta":                   greeks["delta"],
                "gamma":                   greeks["gamma"],
                "vega":                    greeks["vega"],
                "theta":                   greeks["theta"],
                "risk_free_rate":          r,
            })

    return rows


# ---------------------------------------------------------------------------
# 5.  Add GARCH prices to an existing Part B table
#     (called after garch.py fits the model)
# ---------------------------------------------------------------------------

def add_garch_prices(
    part_b_rows: list[dict],
    garch_vol: float,
    r: float = RISK_FREE_RATE,
) -> list[dict]:
    """
    Re-price every row in the Part B table using GARCH-implied volatility
    and append the comparison columns.

    Parameters
    ----------
    part_b_rows : output of build_part_b_table()
    garch_vol   : annualised GARCH 1-step-ahead conditional volatility (decimal)

    Added columns per row:
        garch_vol_pct        — the GARCH vol used (%)
        bsm_price_garch_vol  — BSM price at GARCH vol
        garch_vs_hist_diff   — BSM(GARCH) - BSM(hist)   price difference
        garch_vs_market_diff — BSM(GARCH) - market price difference
    """
    out = []
    for row in part_b_rows:
        r_copy = dict(row)
        spot   = row["spot"]
        K      = row["strike"]
        T      = row["T_years"]
        opt    = row["option_type"].lower()

        try:
            garch_p = round(bsm_price(spot, K, T, r, garch_vol, opt), 4)
        except Exception:
            garch_p = None

        hist_p   = row.get("bsm_price_hist_vol")
        mkt_p    = row.get("market_price", 0)

        r_copy["garch_vol_pct"]        = round(garch_vol * 100, 4)
        r_copy["bsm_price_garch_vol"]  = garch_p
        r_copy["garch_vs_hist_diff"]   = round(garch_p - hist_p, 4) if (garch_p and hist_p) else None
        r_copy["garch_vs_market_diff"] = round(garch_p - mkt_p,  4) if (garch_p and mkt_p)  else None
        out.append(r_copy)

    return out


# ---------------------------------------------------------------------------
# 6.  Lightweight single-option pricer  (used by POST /pricing/bsm)
# ---------------------------------------------------------------------------

def price_single_option(
    S: float,
    K: float,
    T_days: int,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    market_price: Optional[float] = None,
) -> dict:
    """
    Price one European option and return a complete analytics dict.

    If market_price is supplied, also computes implied vol and price deviation.
    """
    T = T_days / TRADING_DAYS

    price  = round(bsm_price(S, K, T, r, sigma, option_type), 4)
    greeks = bsm_greeks(S, K, T, r, sigma, option_type)

    result = {
        "spot":          S,
        "strike":        K,
        "T_days":        T_days,
        "T_years":       round(T, 6),
        "r":             r,
        "sigma_pct":     round(sigma * 100, 4),
        "option_type":   option_type.upper(),
        "bsm_price":     price,
        **greeks,
    }

    if market_price is not None and market_price > 0:
        iv = implied_vol(market_price, S, K, T, r, option_type)
        result["market_price"]      = market_price
        result["implied_vol_pct"]   = round(iv * 100, 4) if iv else None
        result["price_deviation"]   = round(market_price - price, 4)
        result["price_deviation_pct"] = round(
            (market_price - price) / price * 100, 2
        ) if price > 0 else None

    return result
