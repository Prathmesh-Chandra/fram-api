from fastapi import APIRouter, HTTPException
from app.utils.yfinance_client import get_history
from app.data.nifty50 import NIFTY50
from app.data.instrument_keys import get_instrument_key_candidates, normalize_ticker
from app.utils.upstox_client import (
    get_auth_url, exchange_code_for_token,
    get_token_status, get_option_chain
)
from fastapi.responses import RedirectResponse, HTMLResponse

router = APIRouter()

@router.get("/universe")
def universe():
    return {"stocks": NIFTY50, "count": len(NIFTY50)}

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
    result = exchange_code_for_token(code)
    return HTMLResponse(f"""
        <html><body>
            <h2>Authenticated</h2>
            <p>Token valid for <b>{result['expires_at']}</b></p>
            <p>You can close this tab.</p>
        </body></html>
    """)

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