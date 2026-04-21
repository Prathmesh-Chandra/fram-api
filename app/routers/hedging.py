"""
app/routers/portfolio.py
========================
FastAPI router for Part C: Portfolio Construction, Greeks, & Hedging.

Mounts under the prefix /portfolio (registered in main.py).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from app.modules.portfolio import get_portfolio_analysis

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class PositionItem(BaseModel):
    """Represents a single option leg in the custom portfolio."""
    identifier: str = Field(..., description="E.g., '30d ATM Call'")
    quantity: float = Field(..., description="Positive for long, negative for short")
    bsm_price: float
    delta: float
    gamma: float
    vega: float
    theta: Optional[float] = 0.0

class PortfolioRequest(BaseModel):
    """The full payload submitted by the React frontend."""
    ticker: str = Field(..., example="RELIANCE.NS")
    spot: float = Field(..., gt=0)
    current_vol_pct: float = Field(..., gt=0, description="Annualized historical volatility in %")
    turnover_ratio: float = Field(1.0, description="Liquidity proxy from Part A")
    positions: List[PositionItem] = Field(..., min_items=1)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_custom_portfolio(req: PortfolioRequest):
    """
    Evaluates a custom options portfolio. 
    Returns aggregated Net Greeks, liquidity-adjusted Delta hedging requirements, 
    and Taylor Expansion stress tests (±1/2% price, ±20% vol).
    """
    try:
        # Convert Pydantic list to standard dict list for the math module
        positions_dict = [pos.model_dump() for pos in req.positions]
        
        result = get_portfolio_analysis(
            positions=positions_dict,
            spot=req.spot,
            current_vol_pct=req.current_vol_pct,
            turnover_ratio=req.turnover_ratio
        )
        
        return {
            "status": "success",
            "data": {
                "ticker": req.ticker,
                "underlying_spot": req.spot,
                "analysis": result
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")