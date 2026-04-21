"""
Mapping of standard Yahoo Finance tickers to Upstox Instrument Keys.
Upstox format for cash market equities: 'NSE_EQ|ISIN'
"""

from typing import List


UPSTOX_INSTRUMENT_KEYS = {
    "ADANIENT.NS": "NSE_EQ|INE423A01024",
    "ADANIPORTS.NS": "NSE_EQ|INE742F01042",
    "APOLLOHOSP.NS": "NSE_EQ|INE437A01024",
    "ASIANPAINT.NS": "NSE_EQ|INE021A01026",
    "AXISBANK.NS": "NSE_EQ|INE238A01034",
    "BAJAJ-AUTO.NS": "NSE_EQ|INE917I01010",
    "BAJFINANCE.NS": "NSE_EQ|INE296A01024",
    "BAJAJFINSV.NS": "NSE_EQ|INE918I01026",
    "BPCL.NS": "NSE_EQ|INE029A01011",
    "BHARTIARTL.NS": "NSE_EQ|INE397D01024",
    "BRITANNIA.NS": "NSE_EQ|INE216A01030",
    "CIPLA.NS": "NSE_EQ|INE059A01026",
    "COALINDIA.NS": "NSE_EQ|INE522F01014",
    "DIVISLAB.NS": "NSE_EQ|INE361B01024",
    "DRREDDY.NS": "NSE_EQ|INE089A01031",
    "EICHERMOT.NS": "NSE_EQ|INE066A01021",
    "GRASIM.NS": "NSE_EQ|INE047A01021",
    "HCLTECH.NS": "NSE_EQ|INE860A01027",
    "HDFCBANK.NS": "NSE_EQ|INE040A01034",
    "HDFC.NS": "NSE_EQ|INE001A01036",
    "HEROMOTOCO.NS": "NSE_EQ|INE158A01026",
    "HINDALCO.NS": "NSE_EQ|INE038A01020",
    "HINDUNILVR.NS": "NSE_EQ|INE030A01027",
    "ICICIBANK.NS": "NSE_EQ|INE090A01021",
    "ITC.NS": "NSE_EQ|INE154A01025",
    "INDUSINDBK.NS": "NSE_EQ|INE095A01012",
    "INFY.NS": "NSE_EQ|INE009A01021",
    "JSWSTEEL.NS": "NSE_EQ|INE019A01038",
    "KOTAKBANK.NS": "NSE_EQ|INE237A01028",
    "LT.NS": "NSE_EQ|INE018A01030",
    "M&M.NS": "NSE_EQ|INE101A01026",
    "MARUTI.NS": "NSE_EQ|INE585B01010",
    "NESTLEIND.NS": "NSE_EQ|INE239A01024",
    "NTPC.NS": "NSE_EQ|INE733E01010",
    "ONGC.NS": "NSE_EQ|INE213A01029",
    "POWERGRID.NS": "NSE_EQ|INE752E01010",
    "RELIANCE.NS": "NSE_EQ|INE002A01018",
    "SBILIFE.NS": "NSE_EQ|INE123W01016",
    "SBIN.NS": "NSE_EQ|INE062A01020",
    "SUNPHARMA.NS": "NSE_EQ|INE044A01036",
    "TCS.NS": "NSE_EQ|INE467B01029",
    "TATACONSUM.NS": "NSE_EQ|INE192A01025",
    "TATAMOTORS.NS": "NSE_EQ|INE155A01022",
    "TATASTEEL.NS": "NSE_EQ|INE081A01020",
    "TECHM.NS": "NSE_EQ|INE669C01036",
    "TITAN.NS": "NSE_EQ|INE280A01028",
    "ULTRACEMCO.NS": "NSE_EQ|INE481G01011",
    "UPL.NS": "NSE_EQ|INE628A01036",
    "WIPRO.NS": "NSE_EQ|INE075A01022"
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