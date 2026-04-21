from fastapi import APIRouter, HTTPException
import os
from app.utils.yfinance_client import get_history, get_live_indices_snapshot
from app.data.nifty50 import NIFTY50
from app.modules.liquidity import rank_universe_by_turnover
from app.data.instrument_keys import get_instrument_key_candidates, normalize_ticker
from app.utils.upstox_client import (
    get_auth_url, exchange_code_for_token,
    get_token_status, get_option_chain, is_authenticated
)
from fastapi.responses import RedirectResponse
from functools import lru_cache
from urllib.parse import urlparse

router = APIRouter()


def _normalize_public_url(raw: str, local_default_scheme: str = "http") -> str:
    value = (raw or "").strip().strip('"').strip("'")
    if not value:
        return value

    if urlparse(value).scheme:
        return value

    if value.startswith("localhost") or value.startswith("127.0.0.1"):
        return f"{local_default_scheme}://{value}"

    return f"https://{value}"

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


@router.get("/indices/live")
def live_indices():
    rows = get_live_indices_snapshot()
    ok_count = sum(1 for row in rows if row.get("value") is not None)

    if ok_count == 0:
        raise HTTPException(status_code=502, detail="Unable to fetch live index data")

    return {
        "status": "success" if ok_count == len(rows) else "partial_success",
        "data": {
            "indices": rows,
        },
    }


@router.get("/upstox/login")
def upstox_login():
    try:
        return RedirectResponse(url=get_auth_url())
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upstox/callback")
def upstox_callback(code: str):
    try:
        exchange_code_for_token(code)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    frontend_url = _normalize_public_url(
        os.getenv("FRONTEND_URL", "http://localhost:5173")
    ).rstrip("/")
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