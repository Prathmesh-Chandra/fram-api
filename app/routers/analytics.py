"""
app/routers/analytics.py
========================
FastAPI router for Part A analytics endpoints.

Mounts under the prefix /analytics (registered in main.py).

Endpoints
---------
POST /analytics/volatility
POST /analytics/liquidity
POST /analytics/part-a          ← combined single call for the frontend

All endpoints return JSON that matches the payload shapes defined in the
module files so the frontend can chart directly.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.modules.volatility import get_volatility_analysis
from app.modules.liquidity  import get_liquidity_analysis, vol_liquidity_correlation

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class VolatilityRequest(BaseModel):
    ticker:     str         = Field(...,  example="RELIANCE.NS")
    period:     str         = Field("122d", example="122d")
    vol_window: int         = Field(20,   ge=5, le=60)
    acf_lags:   int         = Field(20,   ge=5, le=40)


class LiquidityRequest(BaseModel):
    ticker: str = Field(...,  example="RELIANCE.NS")
    period: str = Field("122d", example="122d")


class PartARequest(BaseModel):
    """
    Combined request — runs both volatility and liquidity for two tickers
    so the frontend can populate Part A in a single call.
    """
    liquid_ticker:   str = Field(..., example="RELIANCE.NS")
    illiquid_ticker: str = Field(..., example="DIVISLAB.NS")
    period:          str = Field("122d", example="122d")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/volatility")
async def volatility_endpoint(req: VolatilityRequest):
    """
    Returns daily log returns, 20-day rolling realised volatility,
    summary statistics, and volatility-clustering diagnostics.

    Response shape
    --------------
    {
        ticker          : str,
        period          : str,
        summary_stats   : { mean_daily_return, std_daily_return, skewness,
                            excess_kurtosis, min_return, max_return,
                            annualised_vol, annualised_vol_pct, n_trading_days },
        clustering      : { acf_squared_returns, ljung_box_stat,
                            ljung_box_pvalue, clustering_detected },
        timeseries      : { dates, close, log_returns, rolling_vol_20d },
        regime_counts   : { "High-Vol": N, "Normal": N }
    }
    """
    try:
        result = get_volatility_analysis(
            ticker=req.ticker,
            period=req.period,
            vol_window=req.vol_window,
            acf_lags=req.acf_lags,
        )
        return {"status": "success", "data": result}

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
        raise HTTPException(status_code=500, detail=f"Volatility analysis failed: {str(e)}")


@router.post("/liquidity")
async def liquidity_endpoint(req: LiquidityRequest):
    """
    Returns turnover ratio, Amihud illiquidity (20d MA), liquidity
    classification, and correlation with rolling volatility.

    Response shape
    --------------
    {
        ticker          : str,
        period          : str,
        summary_stats   : { avg_daily_turnover_cr, avg_amihud_scaled_1e7,
                            turnover_ratio_mean, turnover_ratio_std,
                            liq_class_counts },
        correlation     : { corr_matrix, interpretation },
        timeseries      : { dates, turnover_inr_cr, turnover_ratio,
                            amihud_ma_scaled_1e7, liq_class }
    }
    """
    try:
        result = get_liquidity_analysis(
            ticker=req.ticker,
            period=req.period,
        )
        return {"status": "success", "data": result}

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
        raise HTTPException(status_code=500, detail=f"Liquidity analysis failed: {str(e)}")


@router.post("/part-a")
async def part_a_combined(req: PartARequest):
    """
    Runs the full Part A analysis for both selected stocks in parallel and
    returns everything the frontend needs in a single response:

      - Volatility analysis for both tickers
      - Liquidity analysis for both tickers
      - Cross-stock comparison object

    This is the endpoint the frontend dashboard calls for the Part A panel.

    Response shape
    --------------
    {
        liquid   : { volatility: {...}, liquidity: {...} },
        illiquid : { volatility: {...}, liquidity: {...} },
        comparison : {
            annualised_vol    : { liquid: x%, illiquid: y% },
            avg_turnover_cr   : { liquid: x,  illiquid: y  },
            avg_amihud_1e7    : { liquid: x,  illiquid: y  },
            correlation_liquid   : { vol_vs_turnover_r, vol_vs_amihud_r },
            correlation_illiquid : { vol_vs_turnover_r, vol_vs_amihud_r },
        }
    }
    """
    import asyncio

    async def _run_vol(ticker):
        return get_volatility_analysis(ticker=ticker, period=req.period)

    async def _run_liq(ticker):
        return get_liquidity_analysis(ticker=ticker, period=req.period)

    try:
        # Run all four analyses concurrently in a thread pool
        import concurrent.futures
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                "lv": loop.run_in_executor(pool, get_volatility_analysis, req.liquid_ticker,   req.period),
                "iv": loop.run_in_executor(pool, get_volatility_analysis, req.illiquid_ticker, req.period),
                "ll": loop.run_in_executor(pool, get_liquidity_analysis,  req.liquid_ticker,   req.period),
                "il": loop.run_in_executor(pool, get_liquidity_analysis,  req.illiquid_ticker, req.period),
            }
            lv, iv, ll, il = await asyncio.gather(
                futures["lv"], futures["iv"],
                futures["ll"], futures["il"],
            )

        # Build comparison block
        comparison = {
            "annualised_vol": {
                "liquid":   lv["summary_stats"]["annualised_vol_pct"],
                "illiquid": iv["summary_stats"]["annualised_vol_pct"],
            },
            "avg_turnover_cr": {
                "liquid":   ll["summary_stats"]["avg_daily_turnover_cr"],
                "illiquid": il["summary_stats"]["avg_daily_turnover_cr"],
            },
            "avg_amihud_1e7": {
                "liquid":   ll["summary_stats"]["avg_amihud_scaled_1e7"],
                "illiquid": il["summary_stats"]["avg_amihud_scaled_1e7"],
            },
            "avg_amihud_1e10": {
                "liquid":   ll["summary_stats"]["avg_amihud_scaled_1e10"],
                "illiquid": il["summary_stats"]["avg_amihud_scaled_1e10"],
            },
            "correlation_liquid": {
                "vol_vs_turnover_r": ll["correlation"]["interpretation"]["vol_vs_turnover_r"],
                "vol_vs_amihud_r":   ll["correlation"]["interpretation"]["vol_vs_amihud_r"],
            },
            "correlation_illiquid": {
                "vol_vs_turnover_r": il["correlation"]["interpretation"]["vol_vs_turnover_r"],
                "vol_vs_amihud_r":   il["correlation"]["interpretation"]["vol_vs_amihud_r"],
            },
        }

        return {
            "status": "success",
            "data": {
                "liquid":     {"volatility": lv, "liquidity": ll},
                "illiquid":   {"volatility": iv, "liquidity": il},
                "comparison": comparison,
            },
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
        raise HTTPException(status_code=500, detail=f"Part A analysis failed: {str(e)}")
