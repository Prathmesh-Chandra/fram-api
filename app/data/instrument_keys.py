"""
Mapping of standard Yahoo Finance tickers to Upstox Instrument Keys.
Upstox format for cash market equities: 'NSE_EQ|ISIN'
"""

from typing import List


UPSTOX_INSTRUMENT_KEYS = {
    "RELIANCE.NS": "NSE_EQ|INE002A01018",
    "TCS.NS": "NSE_EQ|INE467B01029",
    "HDFCBANK.NS": "NSE_EQ|INE040A01034",
    "ICICIBANK.NS": "NSE_EQ|INE090A01021",
    "INFY.NS": "NSE_EQ|INE009A01021",
    "SBIN.NS": "NSE_EQ|INE062A01020",
    "BHARTIARTL.NS": "NSE_EQ|INE397D01024",
    "ITC.NS": "NSE_EQ|INE154A01025",
    "ASIANPAINT.NS": "NSE_EQ|INE021A01026",
    "HINDUNILVR.NS": "NSE_EQ|INE030A01027",
    
    # Note: We can expand this dictionary later to include the remaining 40 NIFTY 50 stocks.
    # For now, these top 10 are perfect for testing the option chain endpoints.
}


def normalize_ticker(ticker: str) -> str:
    """Normalizes ticker input like 'reliance' or 'RELIANCE.NS' to 'RELIANCE.NS'."""
    symbol = ticker.strip().upper()
    if not symbol.endswith(".NS"):
        symbol = f"{symbol}.NS"
    return symbol


def _to_equity_symbol_key(normalized_ticker: str) -> str:
    """Builds alternate Upstox key format used by some option endpoints."""
    base_symbol = normalized_ticker.removesuffix(".NS")
    return f"NSE_EQ|{base_symbol}"

def get_instrument_key(ticker: str) -> str | None:
    """
    Safely retrieves the Upstox instrument key for a given ticker.
    Returns None if the ticker is not found in our static mapper.
    """
    normalized = normalize_ticker(ticker)
    return UPSTOX_INSTRUMENT_KEYS.get(normalized)


def get_instrument_key_candidates(ticker: str) -> List[str]:
    """Returns candidate underlying keys in priority order for option APIs."""
    normalized = normalize_ticker(ticker)
    candidates: List[str] = []

    mapped = UPSTOX_INSTRUMENT_KEYS.get(normalized)
    if mapped:
        candidates.append(mapped)

    # Some Upstox option endpoints work with NSE_EQ|<SYMBOL> for underlyings.
    candidates.append(_to_equity_symbol_key(normalized))

    # Keep order while removing duplicates.
    return list(dict.fromkeys(candidates))