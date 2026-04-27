"""
app/routers/risk.py
===================
FastAPI router for Part D: Risk Measurement & Stress Analysis (VaR).

Mounts under the prefix /risk (registered in main.py).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import asyncio
import concurrent.futures

from app.modules.volatility import fetch_price_df, build_price_df
from app.modules.var import (
    calculate_monte_carlo_var, 
    calculate_garch_var, 
    var_by_regime, 
    generate_var_interpretation
)

router = APIRouter(prefix="/risk", tags=["risk"])

class VarCompareRequest(BaseModel):
    liquid_ticker: str = Field(..., example="RELIANCE.NS")
    illiquid_ticker: str = Field(..., example="DIVISLAB.NS")
    period: str = Field("6mo")

def _run_single_var_pipeline(ticker: str, period: str) -> dict:
    """Synchronous pipeline to fetch data and run all VaR math for one ticker."""
    df_raw = fetch_price_df(ticker, period=period)
    df = build_price_df(df_raw)
    returns = df["Log_Return"].dropna()
    
    regime_var = var_by_regime(df)
    mc_var = calculate_monte_carlo_var(returns)
    garch_var = calculate_garch_var(returns)

    empirical_returns_pct = (returns * 100).round(6).tolist()
    
    return {
        "ticker": ticker,
        "empirical_returns_pct": empirical_returns_pct,
        "parametric_regime_var": regime_var,
        "monte_carlo_var": mc_var,
        "garch_var": garch_var
    }

@router.post("/compare")
async def compare_var(req: VarCompareRequest):
    """
    Executes the full Part D analysis. Runs the Liquid and Illiquid stocks 
    concurrently, computes all VaR regimes, and auto-generates the academic interpretations.
    """
    loop = asyncio.get_event_loop()
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            liq_fut = loop.run_in_executor(pool, _run_single_var_pipeline, req.liquid_ticker, req.period)
            ill_fut = loop.run_in_executor(pool, _run_single_var_pipeline, req.illiquid_ticker, req.period)
            
            liq_result, ill_result = await asyncio.gather(liq_fut, ill_fut)
            
        # Generate the dynamic academic commentary
        interpretations = generate_var_interpretation(
            liquid_regime=liq_result["parametric_regime_var"],
            illiquid_regime=ill_result["parametric_regime_var"]
        )
            
        return {
            "status": "success",
            "data": {
                "liquid_asset": liq_result,
                "illiquid_asset": ill_result,
                "interpretations": interpretations
            }
        }
        
    except PermissionError:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UPSTOX_AUTH_REQUIRED",
                "message": "Upstox login required",
                "login_url": "/data/upstox/login",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VaR analysis failed: {str(e)}")