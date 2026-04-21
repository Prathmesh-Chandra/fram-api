"""
app/modules/portfolio.py
========================
Part C (Step 7) — Portfolio Construction, Greeks, & Hedging

Pure functions only. No side-effects.

What this module does
---------------------
1. Aggregates position-level Greeks to calculate Net Portfolio Greeks.
2. Computes the raw and liquidity-adjusted Delta Hedge.
3. Simulates Portfolio PnL under Price and Volatility shocks using 
   Delta-Gamma-Vega Taylor expansion approximations.
4. Auto-generates academic interpretations for hedging effectiveness.
"""

from __future__ import annotations
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# 1. Greeks Aggregation
# ---------------------------------------------------------------------------

def aggregate_portfolio(positions: List[Dict[str, Any]], spot: float) -> Dict[str, Any]:
    """
    Computes position-level costs and Greeks, and aggregates them 
    into net portfolio metrics.
    
    Expected keys in each position dict:
      - quantity (int/float, positive for long, negative for short)
      - bsm_price (float)
      - delta (float)
      - gamma (float)
      - vega (float)
      - theta (float, optional)
    """
    enriched_positions = []
    net_cost = 0.0
    net_delta = 0.0
    net_gamma = 0.0
    net_vega = 0.0
    net_theta = 0.0

    for pos in positions:
        qty = pos.get("quantity", 0)
        
        # Position-level metrics
        pos_cost  = pos.get("bsm_price", 0) * qty
        pos_delta = pos.get("delta", 0) * qty
        pos_gamma = pos.get("gamma", 0) * qty
        pos_vega  = pos.get("vega", 0) * qty
        pos_theta = pos.get("theta", 0) * qty

        # Enrich the original dict for frontend display
        enriched = dict(pos)
        enriched["position_cost"] = round(pos_cost, 2)
        enriched["position_delta"] = round(pos_delta, 4)
        enriched["position_gamma"] = round(pos_gamma, 6)
        enriched["position_vega"] = round(pos_vega, 4)
        enriched["position_theta"] = round(pos_theta, 4)
        enriched_positions.append(enriched)

        # Accumulate net values
        net_cost += pos_cost
        net_delta += pos_delta
        net_gamma += pos_gamma
        net_vega += pos_vega
        net_theta += pos_theta

    return {
        "positions": enriched_positions,
        "aggregate": {
            "total_cost": round(net_cost, 2),
            "net_delta": round(net_delta, 4),
            "net_gamma": round(net_gamma, 6),
            "net_vega": round(net_vega, 4),
            "net_theta": round(net_theta, 4)
        }
    }


# ---------------------------------------------------------------------------
# 2. Delta Hedging (Liquidity Adjusted)
# ---------------------------------------------------------------------------

def calculate_delta_hedge(net_delta: float, spot: float, turnover_ratio: float = 1.0) -> Dict[str, Any]:
    """
    Determines shares of the underlying needed to delta-neutralize the portfolio.
    Adjusts the hedge size down if the liquidity proxy (turnover ratio) is < 1.0.
    """
    # To neutralize a net delta of +50, you must short 50 shares (-50)
    shares_needed_raw = -net_delta
    
    # Adjust for liquidity constraints
    adj_factor = min(turnover_ratio, 1.0) if turnover_ratio > 0 else 1.0
    shares_adjusted = shares_needed_raw * adj_factor

    # Auto-generate interpretation for the assignment deliverable
    if abs(shares_needed_raw) < 1e-4:
        interpretation = "Portfolio is currently delta-neutral. No immediate hedging is required."
    elif adj_factor == 1.0:
        interpretation = "Sufficient liquidity (Turnover Ratio >= 1.0). The full delta hedge can be executed in the underlying market without assuming significant adverse price impact (slippage)."
    else:
        unhedged_delta = abs(shares_needed_raw - shares_adjusted)
        interpretation = (f"Reduced liquidity environment (Turnover Ratio = {turnover_ratio:.2f}). "
                          f"Executing the full hedge may cause severe market impact. The adjusted hedge "
                          f"scales the required trade down by the liquidity factor, leaving a residual "
                          f"unhedged delta equivalent to {unhedged_delta:.2f} shares to balance risk vs. execution cost.")

    return {
        "net_portfolio_delta": round(net_delta, 4),
        "shares_to_hedge_raw": round(shares_needed_raw, 4),
        "liquidity_adjustment_factor": round(adj_factor, 4),
        "shares_to_hedge_adjusted": round(shares_adjusted, 4),
        "hedge_cost_raw_inr": round(abs(shares_needed_raw) * spot, 2),
        "hedge_cost_adjusted_inr": round(abs(shares_adjusted) * spot, 2),
        "interpretation": interpretation
    }


# ---------------------------------------------------------------------------
# 3. PnL Stress Simulation
# ---------------------------------------------------------------------------

def simulate_pnl(
    net_delta: float, 
    net_gamma: float, 
    net_vega: float, 
    spot: float, 
    current_vol_decimal: float,
    price_shocks: List[float] = [-0.02, -0.01, 0.01, 0.02],
    vol_shocks: List[float] = [-0.20, 0.20]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Approximates theoretical PnL using the Taylor expansion:
    PnL = (Delta * dS) + (0.5 * Gamma * dS^2) + (Vega * dVol_Percentage_Points)
    """
    simulations = {"price_shocks": [], "volatility_shocks": []}

    # Price Shocks (Delta-Gamma approximation)
    for pct in price_shocks:
        dS = spot * pct
        pnl = (net_delta * dS) + (0.5 * net_gamma * (dS ** 2))
        simulations["price_shocks"].append({
            "shock_label": f"Price {pct:+.0%}",
            "shock_value_pct": round(pct * 100, 1),
            "spot_change": round(dS, 2),
            "simulated_pnl_inr": round(pnl, 2)
        })

    # Volatility Shocks (Vega approximation)
    for pct in vol_shocks:
        # If current vol is 20%, a +20% shock means new vol is 24% (+4 percentage points)
        dVol_decimal = current_vol_decimal * pct
        dVol_percentage_points = dVol_decimal * 100 
        
        # Vega is conventionally defined as PnL per 1 percentage point change in IV
        pnl = net_vega * dVol_percentage_points
        simulations["volatility_shocks"].append({
            "shock_label": f"Vol {pct:+.0%}",
            "shock_value_pct": round(pct * 100, 1),
            "vol_change_pts": round(dVol_percentage_points, 2),
            "simulated_pnl_inr": round(pnl, 2)
        })

    return simulations


# ---------------------------------------------------------------------------
# 4. Top-Level Service Function
# ---------------------------------------------------------------------------

def get_portfolio_analysis(
    positions: List[Dict[str, Any]], 
    spot: float, 
    current_vol_pct: float, 
    turnover_ratio: float = 1.0
) -> Dict[str, Any]:
    """
    Executes the full Part C pipeline.
    """
    current_vol_decimal = current_vol_pct / 100.0
    
    # 1. Aggregate
    portfolio = aggregate_portfolio(positions, spot)
    agg = portfolio["aggregate"]
    
    # 2. Hedge
    hedge = calculate_delta_hedge(agg["net_delta"], spot, turnover_ratio)
    
    # 3. Simulate
    simulations = simulate_pnl(
        net_delta=agg["net_delta"],
        net_gamma=agg["net_gamma"],
        net_vega=agg["net_vega"],
        spot=spot,
        current_vol_decimal=current_vol_decimal
    )

    return {
        "portfolio": portfolio,
        "hedging": hedge,
        "simulations": simulations
    }