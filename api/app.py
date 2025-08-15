import streamlit as st
import pandas as pd
from datetime import date, timedelta

from src import extractingesg, utilityfunc, logicopt

st.set_page_config(page_title="ECOALPHA", page_icon="🌿", layout="centered")

st.title("🌿 ECOALPHA — ESG-Informed Portfolio Optimizer")

# --- Sidebar controls ---
st.sidebar.header("Settings")
default_tickers = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN"]
tickers = st.sidebar.text_input("Tickers (space-separated)", " ".join(default_tickers)).split()
start_date = st.sidebar.date_input("Start date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End date", date.today())
use_mock = st.sidebar.checkbox("Use mock ESG (varied, deterministic)", value=True)
run_button = st.sidebar.button("Run Optimization")

st.markdown(
    "Use **mock ESG** to guarantee varied scores for demos. "
    "Turn it off to try live Yahoo sustainability + news (+ optional local filings in `data/raw/esg_reports/`)."
)

def run_pipeline(_tickers, _start, _end, _mock):
    # 1) ESG sources
    if _mock:
        raw = extractingesg.build_mock_raw(_tickers, seed=2025)
    else:
        raw = extractingesg.download_and_extract(_tickers)

    # 2) ESG scores (0..1)
    esg_scores = extractingesg.run_esg_analysis(raw)
    st.subheader("ESG Scores (normalized 0–1)")
    st.write(pd.DataFrame(esg_scores).T)

    # 3) Prices
    prices = utilityfunc.download_price_data(_tickers, str(_start), str(_end))
    st.subheader("Price Data (head)")
    st.write(prices.head())

    # 4) Optimize with ESG
    try:
        weights = logicopt.optimize_portfolio(prices, esg_scores=esg_scores)
    except Exception as e:
        st.warning(f"Max Sharpe failed ({e}). Falling back to min vol.")
        weights = logicopt.optimize_portfolio(prices)

    # Output
    st.subheader("Optimized Portfolio Weights")
    wdf = pd.DataFrame({"weight": weights}).sort_values("weight", ascending=False)
    st.write(wdf.style.format({"weight": "{:.2%}"}))

    # Simple chart
    st.bar_chart(wdf)

if run_button:
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        run_pipeline(tickers, start_date, end_date, use_mock)
else:
    st.info("Set tickers & dates in the sidebar, then click **Run Optimization**.")