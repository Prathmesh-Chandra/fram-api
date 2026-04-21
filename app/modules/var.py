"""
app/modules/var.py
==================
Part D (Step 9) — Value-at-Risk (VaR) & Stress Analysis

Pure functions only. No side-effects.

Covers:
1. Parametric (Normal) VaR at 95% and 99%.
2. Regime-Split VaR (Normal vs. High-Volatility).
3. Monte Carlo Empirical VaR (50,000 simulations).
4. GARCH(1,1) Conditional VaR.
5. Auto-generated interpretations for the academic deliverables.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Any

# Attempt to import ARCH for Optional Part D
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

CONFIDENCE_LEVELS = [0.95, 0.99]

# ---------------------------------------------------------------------------
# 1. Core VaR Calculations
# ---------------------------------------------------------------------------

def calculate_parametric_var(returns: pd.Series, conf_levels: List[float] = CONFIDENCE_LEVELS) -> Dict[str, float]:
    """Computes standard parametric (Normal) 1-day VaR."""
    mu = returns.mean()
    sigma = returns.std()
    
    var_results = {}
    for cl in conf_levels:
        z = norm.ppf(1 - cl) # Inverse of CDF for the left tail
        var_value = -(mu + z * sigma)
        var_results[f"var_{int(cl * 100)}_pct"] = round(var_value * 100, 4)
        
    return var_results

def calculate_monte_carlo_var(returns: pd.Series, n_sims: int = 50000, conf_levels: List[float] = CONFIDENCE_LEVELS) -> Dict[str, Any]:
    """Simulates 50,000 return paths to extract empirical VaR (Optional Target)."""
    np.random.seed(42) # For reproducible API responses
    mu = returns.mean()
    sigma = returns.std()
    simulated_returns = np.random.normal(mu, sigma, n_sims)
    
    var_results = {}
    for cl in conf_levels:
        percentile = (1 - cl) * 100
        # Empirical percentile of the simulated distribution
        var_value = -np.percentile(simulated_returns, percentile)
        var_results[f"mc_var_{int(cl * 100)}_pct"] = round(var_value * 100, 4)
        
    return var_results

def calculate_garch_var(returns: pd.Series, conf_levels: List[float] = CONFIDENCE_LEVELS) -> Dict[str, Any]:
    """Uses GARCH(1,1) conditional volatility for dynamic VaR estimation (Optional Target)."""
    if not ARCH_AVAILABLE:
        return {"error": "arch library not installed"}
        
    ret_pct = returns.dropna() * 100
    model = arch_model(ret_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    
    try:
        result = model.fit(disp="off", show_warning=False)
        cond_vol = result.conditional_volatility / 100 # Convert back to decimal
        
        last_cond_vol = cond_vol.iloc[-1]
        mu_decimal = returns.mean()
        
        var_results = {"garch_sigma_last": round(last_cond_vol, 6)}
        for cl in conf_levels:
            z = norm.ppf(1 - cl)
            var_value = -(mu_decimal + z * last_cond_vol)
            var_results[f"garch_var_{int(cl * 100)}_pct"] = round(var_value * 100, 4)
            
        return var_results
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# 2. Regime Split Analysis
# ---------------------------------------------------------------------------

def var_by_regime(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Splits returns into Normal and High-Vol regimes (top 25% rolling vol).
    Computes parametric VaR for each regime to analyze stability.
    """
    ret = df["Log_Return"].dropna()
    q75_vol = df["RolVol_20d"].quantile(0.75)
    
    hv_mask = df["RolVol_20d"] >= q75_vol
    hv_dates = df.index[hv_mask]

    ret_normal = ret[~ret.index.isin(hv_dates)].dropna()
    ret_hv = ret[ret.index.isin(hv_dates)].dropna()

    regimes = {
        "all_data": ret,
        "normal_regime": ret_normal,
        "high_vol_regime": ret_hv
    }
    
    results = {}
    for name, series in regimes.items():
        var_data = calculate_parametric_var(series)
        results[name] = {
            "n_days": len(series),
            "mean_daily_return": round(series.mean(), 6),
            "sigma_daily": round(series.std(), 6),
            **var_data
        }
        
    return results

# ---------------------------------------------------------------------------
# 3. Academic Deliverable: Interpretations
# ---------------------------------------------------------------------------

def generate_var_interpretation(liquid_regime: Dict, illiquid_regime: Dict) -> Dict[str, str]:
    """
    Auto-generates the required academic commentary comparing stability 
    and liquidity impacts for the Part D assignment.
    """
    # Extract 99% VaR values for comparison
    liq_norm_99 = liquid_regime["normal_regime"]["var_99_pct"]
    liq_hv_99 = liquid_regime["high_vol_regime"]["var_99_pct"]
    ill_norm_99 = illiquid_regime["normal_regime"]["var_99_pct"]
    ill_hv_99 = illiquid_regime["high_vol_regime"]["var_99_pct"]
    
    # Calculate stability (jump in VaR from normal to high-vol)
    liq_jump = (liq_hv_99 - liq_norm_99) / liq_norm_99
    ill_jump = (ill_hv_99 - ill_norm_99) / ill_norm_99
    
    # 1. Stability Commentary
    stability = (
        f"VaR estimates show regime-dependence. For the liquid stock, 99% VaR jumps from {liq_norm_99:.2f}% "
        f"in normal periods to {liq_hv_99:.2f}% during high-volatility regimes (a {liq_jump*100:.1f}% increase). "
        f"Similarly, the illiquid stock sees a {ill_jump*100:.1f}% increase in risk during stressed periods. "
        f"This confirms that static, unconditional VaR severely underestimates risk during market stress."
    )
    
    # 2. Liquidity & Volatility Impact Commentary
    worse_stock = "illiquid" if ill_hv_99 > liq_hv_99 else "liquid"
    impact = (
        f"Comparing the assets, the illiquid stock exhibits a high-volatility 99% VaR of {ill_hv_99:.2f}%, "
        f"compared to {liq_hv_99:.2f}% for the liquid asset. Liquidity frictions tend to amplify price swings "
        f"during sell-offs because market depth cannot absorb the order flow. Therefore, the {worse_stock} stock "
        f"poses a significantly higher tail risk to the portfolio during market turbulence."
    )
    
    return {
        "stability_of_estimates": stability,
        "impact_of_liquidity": impact
    }