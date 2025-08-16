from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import src.utilityfunc as utilityfunc
import src.logicopt as logicopt
import src.extractingesg as extractingesg

# Initialize FastAPI app
app = FastAPI(title="ECOALPHA API", version="1.0.0")

# -------------------------------
# Models for request validation
# -------------------------------
class OptimizeRequest(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    mock_esg: bool = False  # allow testing without real Yahoo ESG data

# -------------------------------
# Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    """Landing page at root /"""
    return """
    <html>
      <head><title>ECOALPHA API</title></head>
      <body style="font-family:system-ui;margin:2rem">
        <h1>🌿 ECOALPHA API</h1>
        <p>Your backend is live.</p>
        <ul>
          <li><a href="/docs">OpenAPI docs</a></li>
          <li><a href="/health">Health check</a></li>
          <li><a href="/optimize">Optimize endpoint (POST)</a></li>
        </ul>
      </body>
    </html>
    """

@app.get("/health")
def health():
    """Simple health check"""
    return {"status": "ok"}

@app.post("/optimize")
def optimize(request: OptimizeRequest):
    """Optimize portfolio with ESG considerations"""
    print(f"ESG-based portfolio optimization for: {request.tickers}")

    # Step 1: ESG data
    if request.mock_esg:
        raw_data = {t: {"environmentScore": 70, "socialScore": 60, "governanceScore": 80} for t in request.tickers}
    else:
        raw_data = extractingesg.download_and_extract(request.tickers)

    esg_results = extractingesg.run_esg_analysis(raw_data)
    print("ESG Scores:", esg_results)

    # Step 2: Market data
    price_df = utilityfunc.download_price_data(request.tickers, request.start_date, request.end_date)
    print(f"Retrieved {len(price_df)} rows of stock price data.")

    # Step 3: Optimization
    weights = logicopt.optimize_portfolio(price_df, esg_results)

    return {"esg": esg_results, "weights": weights}

# -------------------------------
# Run locally (ignored on Render)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
