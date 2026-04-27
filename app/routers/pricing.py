"""
app/routers/pricing.py
======================
FastAPI router for Part B: Option Pricing & Volatility Inputs

Mounts under the prefix /pricing (registered in main.py).

Endpoints
---------
POST /pricing/bsm
    Price a single European option (quick utility).

POST /pricing/part-b
    Full Part B table for one ticker:
      - Fetches option chain from Upstox
      - Selects expiries closest to 30d / 60d
      - Picks ATM, OTM-Call, OTM-Put legs
      - Computes BSM price (historical vol) + Greeks
      - Optionally re-prices with GARCH vol
      - Returns comparison table + vol diagnostics

How data flows
--------------
  1. Compute historical vol            → volatility.get_volatility_analysis()
  2. Fetch 2 expiry chains             → upstox_client.get_option_chain()
  3. Select target expiries            → bsm.select_expiries()
  4. Filter ATM / OTM legs per expiry  → bsm.filter_chain_legs()
  5. Build pricing table               → bsm.build_part_b_table()
  6. (Optional) Fit GARCH              → garch.get_garch_analysis()
  7. (Optional) Add GARCH prices       → bsm.add_garch_prices()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.modules.bsm import (
    RISK_FREE_RATE,
    price_single_option,
    select_expiries,
    filter_chain_legs,
    build_part_b_table,
    add_garch_prices,
)
from app.modules.garch    import get_garch_analysis, ARCH_AVAILABLE
from app.modules.volatility import fetch_price_df, build_price_df
from app.utils.upstox_client import get_option_chain_data   # existing util

router = APIRouter(prefix="/pricing", tags=["pricing"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SingleOptionRequest(BaseModel):
    spot:         float = Field(...,   gt=0,  example=1355.5)
    strike:       float = Field(...,   gt=0,  example=1360.0)
    T_days:       int   = Field(...,   gt=0,  le=730, example=30)
    r:            float = Field(RISK_FREE_RATE, example=0.068)
    sigma:        float = Field(...,   gt=0,  example=0.25)
    option_type:  str   = Field("call", pattern="^(call|put)$")
    market_price: Optional[float] = Field(None, ge=0)


class PartBRequest(BaseModel):
    ticker:          str           = Field(...,   example="RELIANCE.NS")
    target_maturities: list[int]   = Field([30, 60], example=[30, 60])
    r:               float         = Field(RISK_FREE_RATE, example=0.068)
    include_garch:   bool          = Field(True)
    period:          str           = Field("6mo")   # for historical vol calculation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_ticker(ticker: str) -> str:
    """Strip .NS suffix for Upstox  (RELIANCE.NS → RELIANCE)."""
    return ticker.replace(".NS", "").replace(".BSE", "").upper()


def _build_expiry_chains(
    ticker: str,
    target_maturities: list[int],
    available_expiries: list[str],
) -> dict[int, dict]:
    """
    For each target maturity, fetch the full option chain for the
    closest expiry and return a dict keyed by target_days.

    Returns
    -------
    {
        30: { "legs": filter_chain_legs output, "expiry_date": "YYYY-MM-DD" },
        60: { ... },
    }
    """
    expiry_map    = select_expiries(available_expiries, target_maturities)
    expiry_chains = {}

    for target_days, expiry_date in expiry_map.items():
        nifty_ticker = _normalise_ticker(ticker)

        # Call Upstox for this specific expiry
        chain_response = get_option_chain_data(
            ticker=nifty_ticker,
            expiry=expiry_date,
        )

        if chain_response.get("status") != "success":
            continue

        chain_rows = chain_response.get("data", [])
        if not chain_rows:
            continue

        spot = float(chain_rows[0]["underlying_spot_price"])
        legs = filter_chain_legs(chain_rows, spot)

        expiry_chains[target_days] = {
            "legs":        legs,
            "expiry_date": expiry_date,
            "spot":        spot,
            "chain_rows":  chain_rows,
        }

    return expiry_chains


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/bsm")
async def single_bsm_price(req: SingleOptionRequest):
    """
    Price a single European option with optional market-price comparison.

    Returns BSM price, Greeks (delta/gamma/vega/theta), and if
    market_price is supplied, also implied vol and price deviation.
    """
    try:
        result = price_single_option(
            S=req.spot,
            K=req.strike,
            T_days=req.T_days,
            r=req.r,
            sigma=req.sigma,
            option_type=req.option_type,
            market_price=req.market_price,
        )
        return {"status": "success", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BSM pricing failed: {str(e)}")


@router.post("/part-b")
async def part_b_pricing_table(req: PartBRequest):
    """
    Full Part B analysis for one ticker.

    Steps performed
    ---------------
    1. Fetch 6mo of price data → compute historical volatility
    2. Call Upstox for option chain → get available_expiries
    3. Select the two expiries closest to requested maturities
    4. For each expiry: pick ATM / OTM-Call / OTM-Put legs
    5. Compute BSM fair value + Greeks for all legs
    6. (Optional) Fit GARCH(1,1) → re-price all legs at GARCH vol
    7. Return the full comparison table + vol diagnostics

    Response shape
    --------------
    {
        status    : "success",
        data      : {
            ticker          : str,
            spot            : float,
            hist_vol_pct    : float,
            pricing_table   : [ row, ... ],   ← Part B deliverable
            garch           : {               ← null if include_garch=false
                vol_comparison  : {...},
                cond_vol_series : {...},
            },
            expiry_map      : { "30": "YYYY-MM-DD", "60": "YYYY-MM-DD" },
        }
    }
    """
    loop = asyncio.get_event_loop()

    try:
        # ── Step 1: Historical vol ───────────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor() as pool:
            df_raw_fut = loop.run_in_executor(
                pool, fetch_price_df, req.ticker, req.period
            )
            df_raw = await df_raw_fut

        df      = build_price_df(df_raw)
        returns = df["Log_Return"].dropna()
        hist_vol = float(returns.std() * (252 ** 0.5))
        spot     = float(df["Close"].iloc[-1])

        # ── Step 2: Get available expiries (call chain with no expiry filter) ─
        nifty_ticker = _normalise_ticker(req.ticker)
        chain_resp   = get_option_chain_data(ticker=nifty_ticker, expiry=None)
        
        if chain_resp.get("status_code") == 401 or chain_resp.get("error") == "AUTH_REQUIRED":
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "UPSTOX_AUTH_REQUIRED",
                    "message": chain_resp.get("message", "Upstox login required"),
                    "login_url": chain_resp.get("login_url", "/data/upstox/login"),
                },
            )

        if chain_resp.get("status") != "success":
            raise HTTPException(
                status_code=502,
                detail="Option chain unavailable — check Upstox API."
            )

        available_expiries = chain_resp.get("available_expiries", [])
        if not available_expiries:
            raise HTTPException(status_code=502, detail="No expiries returned from Upstox")

        # ── Steps 3-5: Build per-expiry chains and pricing table ─────────────
        expiry_chains = _build_expiry_chains(
            req.ticker, req.target_maturities, available_expiries
        )

        if not expiry_chains:
            raise HTTPException(
                status_code=422,
                detail=f"Could not resolve option legs for {req.ticker}. "
                       f"Available expiries: {available_expiries}"
            )

        # Use the spot from the chain (most current)
        spot_from_chain = next(
            (v["spot"] for v in expiry_chains.values() if v.get("spot")), spot
        )

        pricing_table = build_part_b_table(
            ticker=req.ticker,
            expiry_chains=expiry_chains,
            spot=spot_from_chain,
            hist_vol=hist_vol,
            r=req.r,
        )

        # ── Step 6: GARCH (optional) ─────────────────────────────────────────
        garch_payload = None
        if req.include_garch:
            garch_result = get_garch_analysis(returns, req.ticker, hist_vol)
            garch_vol    = garch_result.get("garch_vol")

            if garch_vol:
                pricing_table = add_garch_prices(pricing_table, garch_vol, req.r)

            # Return conditional vol series for charting + vol comparison
            garch_fit = garch_result.get("garch_fit") or {}
            garch_payload = {
                "arch_available":  garch_result["arch_available"],
                "vol_comparison":  garch_result["vol_comparison"],
                "cond_vol_series": {
                    "dates":   garch_fit.get("cond_vol_dates", []),
                    "vol_pct": garch_fit.get("cond_vol_series_pct", []),
                },
            }

        # ── Assemble response ─────────────────────────────────────────────────
        expiry_map = {
            str(k): v["expiry_date"]
            for k, v in expiry_chains.items()
        }

        return {
            "status": "success",
            "data": {
                "ticker":         req.ticker,
                "spot":           round(spot_from_chain, 2),
                "hist_vol_pct":   round(hist_vol * 100, 4),
                "pricing_table":  pricing_table,
                "garch":          garch_payload,
                "expiry_map":     expiry_map,
            },
        }

    except HTTPException:
        raise
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
        raise HTTPException(status_code=500, detail=f"Part B analysis failed: {str(e)}")


@router.post("/part-b/compare")
async def part_b_compare_stocks(
    liquid_ticker:   str,
    illiquid_ticker: str,
    target_maturities: list[int] = None,
    include_garch:   bool = True,
    period:          str  = "6mo",
    r:               float = RISK_FREE_RATE,
):
    """
    Run Part B for both the liquid and illiquid stock and return a side-by-side
    comparison — the deliverable for the "Differences between liquid and illiquid
    stocks" discussion point.

    Runs both tickers concurrently.
    """
    if target_maturities is None:
        target_maturities = [30, 60]

    def _empty_part_b_payload(ticker: str, error_detail=None, status_code: int | None = None) -> dict:
        return {
            "ticker": ticker,
            "spot": None,
            "hist_vol_pct": None,
            "pricing_table": [],
            "garch": None,
            "expiry_map": {},
            "error": {
                "status_code": status_code,
                "detail": error_detail,
            } if error_detail is not None else None,
        }

    req_liquid   = PartBRequest(
        ticker=liquid_ticker,   target_maturities=target_maturities,
        r=r, include_garch=include_garch, period=period,
    )
    req_illiquid = PartBRequest(
        ticker=illiquid_ticker, target_maturities=target_maturities,
        r=r, include_garch=include_garch, period=period,
    )

    async def _run_safe(req: PartBRequest):
        try:
            result = await part_b_pricing_table(req)
            return {
                "ok": True,
                "data": result["data"],
                "error": None,
                "status_code": 200,
            }
        except HTTPException as exc:
            return {
                "ok": False,
                "data": _empty_part_b_payload(req.ticker, exc.detail, exc.status_code),
                "error": exc.detail,
                "status_code": exc.status_code,
            }
        except Exception as exc:
            return {
                "ok": False,
                "data": _empty_part_b_payload(req.ticker, str(exc), 500),
                "error": str(exc),
                "status_code": 500,
            }

    liquid_result, illiquid_result = await asyncio.gather(
        _run_safe(req_liquid),
        _run_safe(req_illiquid),
    )

    # Preserve existing auth flow: if Upstox auth is invalid, bubble up 401.
    for result in (liquid_result, illiquid_result):
        if result.get("status_code") == 401:
            raise HTTPException(status_code=401, detail=result.get("error"))

    liquid_data   = liquid_result["data"]
    illiquid_data = illiquid_result["data"]

    # Summary comparison block
    def _avg_dev(table):
        devs = [r["price_deviation_pct"] for r in table if r.get("price_deviation_pct")]
        return round(sum(devs) / len(devs), 2) if devs else None

    def _vol_spread(table):
        spreads = [r["iv_spread_pct"] for r in table if r.get("iv_spread_pct")]
        return round(sum(spreads) / len(spreads), 2) if spreads else None

    comparison = {
        "hist_vol_pct": {
            "liquid":   liquid_data["hist_vol_pct"],
            "illiquid": illiquid_data["hist_vol_pct"],
        },
        "avg_price_deviation_pct": {
            "liquid":   _avg_dev(liquid_data["pricing_table"]),
            "illiquid": _avg_dev(illiquid_data["pricing_table"]),
        },
        "avg_iv_spread_pct": {
            "liquid":   _vol_spread(liquid_data["pricing_table"]),
            "illiquid": _vol_spread(illiquid_data["pricing_table"]),
        },
    }

    if include_garch and liquid_data.get("garch") and illiquid_data.get("garch"):
        comparison["garch_vol_pct"] = {
            "liquid":   liquid_data["garch"]["vol_comparison"].get("garch_vol_pct"),
            "illiquid": illiquid_data["garch"]["vol_comparison"].get("garch_vol_pct"),
        }

    ok_count = int(bool(liquid_result.get("ok"))) + int(bool(illiquid_result.get("ok")))
    overall_status = "success" if ok_count == 2 else "partial_success"

    return {
        "status": overall_status,
        "data": {
            "liquid":     liquid_data,
            "illiquid":   illiquid_data,
            "comparison": comparison,
            "errors": {
                "liquid": None if liquid_result.get("ok") else liquid_result.get("error"),
                "illiquid": None if illiquid_result.get("ok") else illiquid_result.get("error"),
            },
        },
    }
  
