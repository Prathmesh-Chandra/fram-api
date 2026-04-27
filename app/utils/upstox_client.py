import os
import time
import requests
from datetime import datetime, timedelta
from urllib.parse import urlencode, urlparse


def _clean_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


def _normalize_url(value: str | None) -> str | None:
    if not value:
        return value

    parsed = urlparse(value)
    if parsed.scheme:
        return value

    if value.startswith("localhost") or value.startswith("127.0.0.1"):
        return f"http://{value}"

    return f"https://{value}"


UPSTOX_API_KEY = _clean_env("UPSTOX_API_KEY")
UPSTOX_API_SECRET = _clean_env("UPSTOX_API_SECRET")
UPSTOX_REDIRECT_URI = _normalize_url(
    _clean_env(
        "UPSTOX_REDIRECT_URI",
        "https://web-production-06c6e.up.railway.app/data/upstox/callback",
    )
)
BASE_URL = "https://api.upstox.com/v2"

_token_state = {
    "access_token": None,
    "expires_at":   0,
}


def auth_required_response() -> dict:
    return {
        "error": "AUTH_REQUIRED",
        "message": "Not authenticated. Visit /data/upstox/login",
        "status_code": 401,
        "login_url": "/data/upstox/login",
    }

def get_auth_url() -> str:
    if not UPSTOX_API_KEY or not UPSTOX_REDIRECT_URI:
        raise ValueError(
            "Upstox auth is misconfigured. Set UPSTOX_API_KEY and "
            "UPSTOX_REDIRECT_URI in backend environment variables."
        )

    query = urlencode(
        {
            "response_type": "code",
            "client_id": UPSTOX_API_KEY,
            "redirect_uri": UPSTOX_REDIRECT_URI,
        }
    )
    return (
        f"{BASE_URL}/login/authorization/dialog?{query}"
    )

def exchange_code_for_token(auth_code: str) -> dict:
    if not UPSTOX_API_KEY or not UPSTOX_API_SECRET or not UPSTOX_REDIRECT_URI:
        raise ValueError(
            "Upstox token exchange is misconfigured. Set UPSTOX_API_KEY, "
            "UPSTOX_API_SECRET, and UPSTOX_REDIRECT_URI."
        )

    response = requests.post(
        f"{BASE_URL}/login/authorization/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "code":          auth_code,
            "client_id":     UPSTOX_API_KEY,
            "client_secret": UPSTOX_API_SECRET,
            "redirect_uri":  UPSTOX_REDIRECT_URI,
            "grant_type":    "authorization_code",
        },
    )
    response.raise_for_status()
    data = response.json()
    _token_state["access_token"] = data["access_token"]
    _token_state["expires_at"]   = time.time() + data.get("expires_in", 86400)
    return _token_state

def is_authenticated() -> bool:
    return bool(_token_state["access_token"]) and time.time() < _token_state["expires_at"]

def get_headers() -> dict:
    return {
        "Authorization": f"Bearer {_token_state['access_token']}",
        "Accept": "application/json",
    }

def get_token_status() -> dict:
    if not _token_state["access_token"]:
        return {
            "authenticated": False,
            "reason": "No token. Visit /data/upstox/login",
            "login_url": "/data/upstox/login",
        }
    time_left = _token_state["expires_at"] - time.time()
    if time_left <= 0:
        return {
            "authenticated": False,
            "reason": "Token expired. Visit /data/upstox/login",
            "login_url": "/data/upstox/login",
        }
    return {
        "authenticated": True,
        "expires_in_hours": round(time_left / 3600, 1),
        "user_name": _get_user_display_name(),
    }


def _request_upstox(path: str, params: dict) -> dict:
    """Makes a resilient GET request to an Upstox endpoint and returns parsed JSON."""
    url = f"{BASE_URL}{path}"
    try:
        response = requests.get(url, headers=get_headers(), params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Upstox API request failed: {str(e)}"}


def _period_to_calendar_days(period: str) -> int:
    p = (period or "6mo").strip().lower()
    mapping = {
        "1mo": 35,
        "3mo": 100,
        "6mo": 200,
        "1y": 380,
    }
    if p in mapping:
        return mapping[p]
    if p.endswith("d") and p[:-1].isdigit():
        # Add a margin for weekends/holidays.
        return int(p[:-1]) + 30
    return 200


def _resolve_historical_key_candidates(ticker: str) -> list[str]:
    from app.data.instrument_keys import get_instrument_key_candidates, normalize_ticker

    normalized_ticker = normalize_ticker(ticker)
    candidates = get_instrument_key_candidates(normalized_ticker)
    return candidates


def _parse_upstox_candle_row(row: list) -> dict | None:
    if not isinstance(row, list) or len(row) < 6:
        return None

    # Upstox candle row format is expected as:
    # [timestamp, open, high, low, close, volume, ...]
    ts_raw = row[0]
    try:
        ts = str(ts_raw)
        if "T" in ts:
            date_key = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        else:
            date_key = ts[:10]

        return {
            "date": date_key,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": int(float(row[5])) if row[5] is not None else 0,
        }
    except (ValueError, TypeError):
        return None


def get_historical_candles(ticker: str, period: str = "6mo", interval: str = "day") -> dict:
    """
    Fetch historical candles from Upstox for a mapped NSE equity instrument.

    Returns
    -------
    {
      status: "success" | "error",
      data: [{date, open, high, low, close, volume}, ...],
      instrument_key_used: str,
      period: str,
      interval: str,
      ...auth fields on 401
    }
    """
    if not is_authenticated():
        return auth_required_response()

    candidates = _resolve_historical_key_candidates(ticker)
    if not candidates:
        return {
            "status": "error",
            "message": f"Ticker '{ticker}' is not mapped to an Upstox instrument key",
        }

    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=_period_to_calendar_days(period))

    for instrument_key in candidates:
        path = f"/historical-candle/{instrument_key}/{interval}/{to_date.isoformat()}/{from_date.isoformat()}"
        payload = _request_upstox(path, {})
        if "error" in payload:
            continue

        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            continue

        parsed = [_parse_upstox_candle_row(row) for row in candles]
        parsed = [row for row in parsed if row is not None]
        parsed.sort(key=lambda row: row["date"])

        if not parsed:
            continue

        return {
            "status": "success",
            "period": period,
            "interval": interval,
            "instrument_key_used": instrument_key,
            "data": parsed,
        }

    return {
        "status": "error",
        "message": f"Failed to fetch historical candles from Upstox for {ticker}",
        "period": period,
        "interval": interval,
    }


def _get_user_display_name() -> str | None:
    if not is_authenticated():
        return None

    profile = _request_upstox("/user/profile", {})
    if "error" in profile:
        return None

    data = profile.get("data") or {}
    return (
        data.get("user_name")
        or data.get("name")
        or data.get("user_id")
        or data.get("email")
    )


def get_option_contracts(instrument_key: str) -> dict:
    """Fetches option contracts for an underlying to discover valid expiries."""
    if not is_authenticated():
        return auth_required_response()
    return _request_upstox("/option/contract", {"instrument_key": instrument_key})


def _parse_expiry_date(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def _resolve_expiry(requested_expiry: str, available_expiries: list[str]) -> str | None:
    """Returns nearest valid expiry: same date if present, else next available, else latest."""
    if not available_expiries:
        return None
    if requested_expiry in available_expiries:
        return requested_expiry

    parsed = []
    for expiry in available_expiries:
        parsed_date = _parse_expiry_date(expiry)
        if parsed_date:
            parsed.append((parsed_date, expiry))

    if not parsed:
        return available_expiries[0]

    parsed.sort(key=lambda item: item[0])
    requested_date = _parse_expiry_date(requested_expiry)

    if not requested_date:
        return parsed[0][1]

    for date_obj, expiry in parsed:
        if date_obj >= requested_date:
            return expiry

    return parsed[-1][1]

def get_option_chain(instrument_key: str, expiry_date: str) -> dict:
    """
    Fetches the live option chain from Upstox for a specific instrument and expiry.
    """
    if not is_authenticated():
        return auth_required_response()

    first_attempt = _request_upstox(
        "/option/chain",
        {
            "instrument_key": instrument_key,
            "expiry_date": expiry_date,
        },
    )

    if "error" in first_attempt:
        return first_attempt

    chain_data = first_attempt.get("data", [])
    if chain_data:
        first_attempt["requested_expiry"] = expiry_date
        first_attempt["resolved_expiry"] = expiry_date
        return first_attempt

    contracts = get_option_contracts(instrument_key)
    if "error" in contracts:
        first_attempt["warning"] = "Option chain empty and contracts lookup failed"
        return first_attempt

    available_expiries = sorted(
        {
            contract.get("expiry")
            for contract in contracts.get("data", [])
            if contract.get("expiry")
        }
    )

    resolved_expiry = _resolve_expiry(expiry_date, available_expiries)
    if not resolved_expiry:
        first_attempt["warning"] = "No option expiries available for this underlying"
        first_attempt["available_expiries"] = []
        return first_attempt

    if resolved_expiry == expiry_date:
        first_attempt["available_expiries"] = available_expiries
        return first_attempt

    second_attempt = _request_upstox(
        "/option/chain",
        {
            "instrument_key": instrument_key,
            "expiry_date": resolved_expiry,
        },
    )

    if "error" in second_attempt:
        first_attempt["warning"] = "Failed to fetch fallback expiry chain"
        first_attempt["available_expiries"] = available_expiries
        return first_attempt

    second_attempt["requested_expiry"] = expiry_date
    second_attempt["resolved_expiry"] = resolved_expiry
    second_attempt["available_expiries"] = available_expiries
    return second_attempt

def get_option_chain_data(ticker: str, expiry: str | None = None) -> dict:
    """
    Adapter for the pricing router. Fetches either the full chain (if expiry is given)
    or just the available expiries (if expiry is None).
    """
    from app.data.instrument_keys import get_instrument_key_candidates, normalize_ticker
    
    normalized_ticker = normalize_ticker(ticker)
    candidates = get_instrument_key_candidates(normalized_ticker)

    if not candidates:
        return {"status": "error", "message": f"Ticker '{ticker}' not mapped."}

    # Scenario 1: Only fetching available expiries
    if not expiry:
        for key in candidates:
            contracts = get_option_contracts(key)
            if contracts.get("status_code") == 401:
                return contracts
            if "error" not in contracts and "data" in contracts:
                expiries = sorted({c.get("expiry") for c in contracts.get("data", []) if c.get("expiry")})
                if expiries:
                    return {"status": "success", "available_expiries": expiries}
        return {"status": "error", "message": "No expiries found."}

    # Scenario 2: Fetching the actual option chain for a specific expiry
    for key in candidates:
        data = get_option_chain(key, expiry)
        if data.get("status_code") == 401:
            return data
        if "error" not in data and data.get("data"):
             return {
                 "status": "success", 
                 "data": data["data"], 
                 "available_expiries": data.get("available_expiries", [])
             }

    return {"status": "error", "message": "Failed to fetch option chain from Upstox."}