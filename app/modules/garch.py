"""
app/modules/garch.py
====================
Part B (Step 8, Optional) — GARCH(1,1) Volatility Estimation & Re-pricing

Pure functions only.  No prints, no matplotlib, no side-effects.

What this module does
---------------------
1.  Fits a GARCH(1,1) model on a log-return series using the `arch` library
2.  Extracts the 1-step-ahead conditional volatility forecast (annualised)
3.  Returns model parameters and fit diagnostics
4.  The actual re-pricing is done via bsm.add_garch_prices()
    (so this module stays decoupled from BSM math)

GARCH(1,1) model (arch library)
--------------------------------
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Input returns are scaled to %, i.e. multiplied by 100 before fitting
(arch library convention: works better numerically in % space).

After fitting:
    annualised_vol = sqrt(forecast_variance_pct) × sqrt(252) / 100
                   = sqrt(forecast_variance_pct × 252) / 100
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1.  GARCH model fitting
# ---------------------------------------------------------------------------

def fit_garch(
    log_returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> Optional[dict]:
    """
    Fit a GARCH(p,q) model on a daily log-return series.

    Parameters
    ----------
    log_returns : pandas Series of daily log returns (decimal, NOT %)
    p, q        : GARCH order (default 1,1)
    dist        : innovation distribution ("normal" or "t")

    Returns
    -------
    dict with:
        omega, alpha, beta            — model parameters
        persistence                   — alpha + beta (< 1 means mean-reverting)
        long_run_vol_pct              — unconditional volatility (%)
        log_likelihood                — model fit quality
        aic, bic                      — information criteria
        cond_vol_series_pct           — daily conditional vol series (%)
        last_cond_vol_pct             — most recent conditional vol (%)
        forecast_1step_var_pct2       — 1-step-ahead variance in %² units
        forecast_annualised_vol       — 1-step annualised vol (decimal)
        forecast_annualised_vol_pct   — 1-step annualised vol (%)

    Returns None if arch library is not installed.
    """
    if not ARCH_AVAILABLE:
        return None

    # arch expects % returns (avoids numerical issues near zero)
    ret_pct = log_returns.dropna() * 100

    model  = arch_model(
        ret_pct,
        vol="Garch",
        p=p,
        q=q,
        dist=dist,
        rescale=False,
    )

    try:
        result = model.fit(disp="off", show_warning=False)
    except Exception as e:
        return {"error": str(e)}

    # ── Parameters ──────────────────────────────────────────────────────────
    params = result.params
    omega  = float(params.get("omega",     params.get("Const",   0)))
    alpha  = float(params.get("alpha[1]",  params.get("alpha",   0)))
    beta   = float(params.get("beta[1]",   params.get("beta",    0)))

    persistence  = alpha + beta
    # Unconditional variance in %² = omega / (1 - alpha - beta)
    uncond_var   = omega / max(1 - persistence, 1e-8)
    long_run_vol = float(np.sqrt(uncond_var) * np.sqrt(252) / 100)   # annualised decimal

    # ── Conditional volatility series ────────────────────────────────────────
    cond_vol_pct = result.conditional_volatility   # daily, in % units

    # ── 1-step-ahead forecast ────────────────────────────────────────────────
    forecast   = result.forecast(horizon=1, reindex=False)
    var_1step  = float(forecast.variance.iloc[-1, 0])            # %² units
    ann_vol    = float(np.sqrt(var_1step * 252) / 100)           # annualised decimal

    return {
        # Model parameters
        "omega":                      round(float(omega),        8),
        "alpha":                      round(float(alpha),        6),
        "beta":                       round(float(beta),         6),
        "persistence":                round(float(persistence),  6),
        "long_run_vol_pct":           round(long_run_vol * 100,  4),

        # Fit quality
        "log_likelihood":             round(float(result.loglikelihood), 4),
        "aic":                        round(float(result.aic),           4),
        "bic":                        round(float(result.bic),           4),

        # Conditional volatility time series (for charting)
        "cond_vol_dates":             cond_vol_pct.index.strftime("%Y-%m-%d").tolist(),
        "cond_vol_series_pct":        cond_vol_pct.round(6).tolist(),
        "last_cond_vol_pct":          round(float(cond_vol_pct.iloc[-1]), 6),

        # 1-step-ahead forecast
        "forecast_1step_var_pct2":    round(var_1step,  8),
        "forecast_annualised_vol":    round(ann_vol,    6),
        "forecast_annualised_vol_pct":round(ann_vol * 100, 4),
    }


# ---------------------------------------------------------------------------
# 2.  Volatility comparison helper
# ---------------------------------------------------------------------------

def vol_comparison(
    hist_vol: float,
    garch_result: dict,
) -> dict:
    """
    Build the vol comparison block needed for the Part B discussion section:
    "sensitivity of option prices to volatility assumptions."

    Parameters
    ----------
    hist_vol      : annualised historical volatility (decimal)
    garch_result  : output of fit_garch()

    Returns a dict suitable for JSON serialisation.
    """
    if not garch_result or "error" in garch_result:
        return {"available": False, "reason": garch_result.get("error", "arch not installed")}

    garch_vol = garch_result["forecast_annualised_vol"]
    diff      = garch_vol - hist_vol

    return {
        "available":             True,
        "hist_vol_pct":          round(hist_vol * 100, 4),
        "garch_vol_pct":         garch_result["forecast_annualised_vol_pct"],
        "long_run_vol_pct":      garch_result["long_run_vol_pct"],
        "difference_pct":        round(diff * 100, 4),
        "garch_higher":          diff > 0,
        "garch_params": {
            "omega":             garch_result["omega"],
            "alpha":             garch_result["alpha"],
            "beta":              garch_result["beta"],
            "persistence":       garch_result["persistence"],
        },
        "fit_quality": {
            "log_likelihood":    garch_result["log_likelihood"],
            "aic":               garch_result["aic"],
            "bic":               garch_result["bic"],
        },
        "interpretation": _interpret_garch(garch_result),
    }


def _interpret_garch(g: dict) -> str:
    """Generate a one-line economic interpretation of the GARCH fit."""
    alpha = g["alpha"]
    beta  = g["beta"]
    pers  = g["persistence"]
    llv   = g["long_run_vol_pct"]
    cvol  = g["last_cond_vol_pct"] * np.sqrt(252) / 100 * 100  # annualised %

    shock_impact = "strong" if alpha > 0.15 else ("moderate" if alpha > 0.07 else "weak")
    decay_speed  = "slow"   if beta  > 0.85 else ("moderate" if beta  > 0.70 else "fast")
    mean_rev     = "high persistence" if pers > 0.95 else ("moderate persistence" if pers > 0.85 else "low persistence")

    return (
        f"GARCH(1,1) shows {shock_impact} shock response (α={alpha:.3f}), "
        f"{decay_speed} decay (β={beta:.3f}), {mean_rev} (α+β={pers:.3f}). "
        f"Long-run vol = {llv:.2f}%. Current conditional vol ≈ {cvol:.2f}% annualised."
    )


# ---------------------------------------------------------------------------
# 3.  Top-level service function  (called by the FastAPI router)
# ---------------------------------------------------------------------------

def get_garch_analysis(
    log_returns: pd.Series,
    ticker: str,
    hist_vol: float,
) -> dict:
    """
    Fit GARCH, compute vol comparison, return everything the pricing
    router needs to call bsm.add_garch_prices().

    Returns
    -------
    {
        ticker          : str,
        arch_available  : bool,
        garch_fit       : dict (fit_garch output),
        vol_comparison  : dict,
        garch_vol       : float  ← annualised decimal, ready for bsm.add_garch_prices()
    }
    """
    if not ARCH_AVAILABLE:
        return {
            "ticker":         ticker,
            "arch_available": False,
            "garch_fit":      None,
            "vol_comparison": {"available": False, "reason": "arch library not installed"},
            "garch_vol":      None,
        }

    garch_fit = fit_garch(log_returns)

    return {
        "ticker":         ticker,
        "arch_available": True,
        "garch_fit":      garch_fit,
        "vol_comparison": vol_comparison(hist_vol, garch_fit),
        "garch_vol":      garch_fit.get("forecast_annualised_vol") if garch_fit else None,
    }
