from __future__ import annotations

from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Import your project modules (repo root must contain both `api/` and `src/`)
from src import extractingesg, utilityfunc, logicopt


# -----------------------
# FastAPI app & middleware
# -----------------------
app = FastAPI(title="ECOALPHA API", version="1.0.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten to your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Request/Response models
# -----------------------
class OptimizeRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, description="List of ticker symbols")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    mock_esg: bool = Field(True, description="Use varied mock ESG (good for demos)")
    seed: Optional[int] = Field(2025, description="Seed for mock ESG randomness")

    @field_validator("tickers")
    @classmethod
    def strip_tickers(cls, v: List[str]) -> List[str]:
        return [t.strip().upper() for t in v if t.strip()]

class OptimizeResponse(BaseModel):
    esg: Dict[str, Dict[str, float]]
    weights: Dict[str, float]


# -----------------------
# Health check
# -----------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# -----------------------
# Main endpoint
# -----------------------
@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """
    1) Build ESG signals (mock or live).
    2) Download prices.
    3) Optimize portfolio (ESG-aware), with safe fallback to min volatility.
    """
    # 1) ESG signals
    try:
        if req.mock_esg:
            raw = extractingesg.build_mock_raw(req.tickers, seed=req.seed or 2025)
        else:
            raw = extractingesg.download_and_extract(req.tickers)
        esg = extractingesg.run_esg_analysis(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ESG pipeline failed: {e}")

    # 2) Prices
    try:
        prices = utilityfunc.download_price_data(req.tickers, req.start_date, req.end_date)
        if prices is None or prices.empty:
            raise ValueError("No price data returned (empty DataFrame).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Price download failed: {e}")

    # 3) Optimize (ESG-aware), with robust fallback
    try:
        weights = logicopt.optimize_portfolio(prices, esg_scores=esg)
    except Exception as err:
        # Fallback if max_sharpe (or solver) fails
        try:
            weights = logicopt.optimize_portfolio(prices)  # non-ESG fallback
        except Exception as err2:
            raise HTTPException(
                status_code=500,
                detail=f"Optimization failed: {err}; fallback failed: {err2}",
            )

    return OptimizeResponse(esg=esg, weights=weights)
