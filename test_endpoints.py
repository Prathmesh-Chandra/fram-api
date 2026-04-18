"""
Test suite for FRAM API endpoints.
Run with: python -m pytest test_endpoints.py -v
Or run directly: python test_endpoints.py
"""

import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

# NIFTY 50 tickers to test
VALID_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]

INVALID_TICKERS = [
    "NOTAREALTICKER.NS",
    "FAKEINVALID.NS",
]


def test_history_endpoint_returns_200(ticker: str) -> Dict[str, Any]:
    """Test that a valid ticker returns 200 with properly structured JSON."""
    url = f"{BASE_URL}/data/history?ticker={ticker}&period=6mo"
    response = requests.get(url)
    assert response.status_code == 200, f"{ticker}: expected 200, got {response.status_code}"
    
    data = response.json()
    assert isinstance(data, dict), f"{ticker}: response should be a dict"
    assert "ticker" in data, f"{ticker}: missing 'ticker' field"
    assert "dates" in data, f"{ticker}: missing 'dates' field"
    assert "open" in data, f"{ticker}: missing 'open' field"
    assert "high" in data, f"{ticker}: missing 'high' field"
    assert "low" in data, f"{ticker}: missing 'low' field"
    assert "close" in data, f"{ticker}: missing 'close' field"
    assert "volume" in data, f"{ticker}: missing 'volume' field"
    
    # Verify all arrays have same length
    n = len(data["dates"])
    assert len(data["open"]) == n, f"{ticker}: open length mismatch"
    assert len(data["high"]) == n, f"{ticker}: high length mismatch"
    assert len(data["low"]) == n, f"{ticker}: low length mismatch"
    assert len(data["close"]) == n, f"{ticker}: close length mismatch"
    assert len(data["volume"]) == n, f"{ticker}: volume length mismatch"
    
    # Verify data types
    for date_str in data["dates"]:
        assert isinstance(date_str, str), f"{ticker}: date should be string"
        assert len(date_str) == 10, f"{ticker}: date format should be YYYY-MM-DD"
    
    for val in data["open"]:
        assert val is None or isinstance(val, (int, float)), f"{ticker}: open should be float or None"
    
    for val in data["volume"]:
        assert val is None or isinstance(val, int), f"{ticker}: volume should be int or None"
    
    return data


def test_history_endpoint_returns_404(ticker: str) -> None:
    """Test that invalid tickers return 404 with proper error message."""
    url = f"{BASE_URL}/data/history?ticker={ticker}&period=6mo"
    response = requests.get(url)
    assert response.status_code == 404, f"{ticker}: expected 404, got {response.status_code}"
    
    error = response.json()
    assert "detail" in error, f"{ticker}: error should have 'detail' field"
    print(f"  ✓ {ticker}: correctly rejected with 404")


def pretty_print_response(data: Dict[str, Any]) -> None:
    """Pretty print a sample of the API response data."""
    ticker = data["ticker"]
    n_records = len(data["dates"])
    
    print(f"\n{'='*70}")
    print(f"Ticker: {ticker}")
    print(f"Total Records: {n_records}")
    print(f"{'='*70}")
    print(f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>15}")
    print(f"{'-'*70}")
    
    # Show first 5, last 5 records
    indices = list(range(min(5, n_records))) + (
        ["..."] if n_records > 10 else []
    ) + list(range(max(0, n_records - 5), n_records))
    
    prev_index = -2
    for idx in indices:
        if isinstance(idx, str):
            print(f"{idx:<12}")
            prev_index = -2
            continue
        
        if idx != prev_index + 1 and prev_index >= 0:
            print(f"{'-'*70}")
        
        date_str = data["dates"][idx]
        open_val = data["open"][idx]
        high_val = data["high"][idx]
        low_val = data["low"][idx]
        close_val = data["close"][idx]
        vol_val = data["volume"][idx]
        
        # Format numbers nicely
        open_str = f"{open_val:.2f}" if open_val is not None else "N/A"
        high_str = f"{high_val:.2f}" if high_val is not None else "N/A"
        low_str = f"{low_val:.2f}" if low_val is not None else "N/A"
        close_str = f"{close_val:.2f}" if close_val is not None else "N/A"
        vol_str = f"{vol_val:,}" if vol_val is not None else "N/A"
        
        print(f"{date_str:<12} {open_str:>10} {high_str:>10} {low_str:>10} {close_str:>10} {vol_str:>15}")
        prev_index = idx


def run_tests():
    """Run all tests and pretty-print results."""
    print("\n" + "="*70)
    print("FRAM API Test Suite")
    print("="*70)
    
    # Test valid tickers
    print(f"\n✓ Testing {len(VALID_TICKERS)} valid NIFTY 50 tickers for correct response structure...")
    valid_data = {}
    for ticker in VALID_TICKERS:
        try:
            data = test_history_endpoint_returns_200(ticker)
            valid_data[ticker] = data
            print(f"  ✓ {ticker}: {len(data['dates'])} records, all fields present and properly typed")
        except AssertionError as e:
            print(f"  ✗ {ticker}: {e}")
            return False
        except Exception as e:
            print(f"  ✗ {ticker}: Unexpected error: {e}")
            return False
    
    # Test invalid tickers
    print(f"\n✓ Testing {len(INVALID_TICKERS)} invalid tickers for 404 response...")
    for ticker in INVALID_TICKERS:
        try:
            test_history_endpoint_returns_404(ticker)
        except AssertionError as e:
            print(f"  ✗ {ticker}: {e}")
            return False
        except Exception as e:
            print(f"  ✗ {ticker}: Unexpected error: {e}")
            return False
    
    # Pretty-print first valid ticker's data
    if valid_data:
        first_ticker = VALID_TICKERS[0]
        pretty_print_response(valid_data[first_ticker])
    
    print(f"\n{'='*70}")
    print("All tests passed! ✓")
    print(f"{'='*70}\n")
    return True


if __name__ == "__main__":
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Cannot connect to API at {BASE_URL}")
        print("  Make sure uvicorn is running: uvicorn app.main:app --reload")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)
