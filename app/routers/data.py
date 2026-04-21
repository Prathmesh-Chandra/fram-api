from fastapi import APIRouter, HTTPException
import os
from app.utils.yfinance_client import get_history
from app.data.nifty50 import NIFTY50
from app.modules.liquidity import rank_universe_by_turnover
from app.data.instrument_keys import get_instrument_key_candidates, normalize_ticker
from app.utils.upstox_client import (
    get_auth_url, exchange_code_for_token,
    get_token_status, get_option_chain, is_authenticated
)
from fastapi.responses import RedirectResponse
from functools import lru_cache

router = APIRouter()

@router.get("/universe")
def universe():
    return _get_universe_payload()


@lru_cache(maxsize=1)
def _get_universe_payload():
    try:
        ranked = rank_universe_by_turnover(period="6mo")
        return {
            "liquid": ranked.get("liquid", []),
            "illiquid": ranked.get("illiquid", []),
            "all": ranked.get("all", []),
            "count": len(ranked.get("all", [])),
        }
    except Exception:
        # Safe fallback keeps frontend functional if live ranking fails.
        midpoint = len(NIFTY50) // 2
        liquid = [{"ticker": t, "avg_turnover_cr": None} for t in NIFTY50[:midpoint]]
        illiquid = [{"ticker": t, "avg_turnover_cr": None} for t in NIFTY50[midpoint:]]
        all_stocks = liquid + illiquid
        return {
            "liquid": liquid,
            "illiquid": illiquid,
            "all": all_stocks,
            "count": len(all_stocks),
            "warning": "Using static fallback universe because turnover ranking failed",
        }

@router.get("/history")
def history(ticker: str, period: str = "6mo"):
    data = get_history(ticker, period)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    return data


@router.get("/upstox/login")
def upstox_login():
    return RedirectResponse(url=get_auth_url())

@router.get("/upstox/callback")
def upstox_callback(code: str):
    exchange_code_for_token(code)
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")
    return RedirectResponse(url=f"{frontend_url}/")

@router.get("/upstox/status")
def upstox_status():
    return get_token_status()

@router.get("/option-chain")
def option_chain(ticker: str, expiry: str):
    """
    Fetches the live option chain for a given NIFTY 50 ticker and expiry date.
    Expects expiry in 'YYYY-MM-DD' format.
    """
    normalized_ticker = normalize_ticker(ticker)
    instrument_candidates = get_instrument_key_candidates(normalized_ticker)

    if not is_authenticated():
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UPSTOX_AUTH_REQUIRED",
                "message": "Upstox login required",
                "login_url": "/data/upstox/login",
            },
        )

    if not instrument_candidates:
        raise HTTPException(
            status_code=400, 
            detail=f"Ticker '{ticker}' not found in the instrument key mapper. Please add it to app/data/instrument_keys.py"
        )

    results = []
    errors = []

    for instrument_key in instrument_candidates:
        data = get_option_chain(instrument_key, expiry)
        if "error" in data:
            if data.get("status_code") == 401:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "code": "UPSTOX_AUTH_REQUIRED",
                        "message": data.get("message", "Upstox login required"),
                        "login_url": data.get("login_url", "/data/upstox/login"),
                    },
                )
            errors.append({"instrument_key": instrument_key, "error": data["error"]})
            continue

        data["requested_ticker"] = ticker
        data["normalized_ticker"] = normalized_ticker
        data["instrument_key_used"] = instrument_key
        results.append(data)

        if data.get("data"):
            return data

    if results:
        fallback = results[0]
        fallback["warning"] = fallback.get("warning") or "No contracts returned for requested input"
        fallback["instrument_keys_tried"] = instrument_candidates
        return fallback

    raise HTTPException(
        status_code=400,
        detail={
            "message": "Failed to fetch option chain from Upstox",
            "instrument_keys_tried": instrument_candidates,
            "errors": errors,
        },
    )