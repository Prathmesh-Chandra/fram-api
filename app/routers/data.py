from fastapi import APIRouter, HTTPException
from app.utils.yfinance_client import get_history
from app.data.nifty50 import NIFTY50

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