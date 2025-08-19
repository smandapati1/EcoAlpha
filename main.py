# main.py  — Streamlit UI (Tropir-style)
from datetime import date, timedelta
from typing import List, Tuple, Dict
import pandas as pd
import streamlit as st

from src import extractingesg, utilityfunc, logicopt

# ---------------- Core pipeline (unchanged logic) ----------------
def run_pipeline(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_mock: bool = True,
    seed: int = 2025,
) -> Tuple[Dict, pd.DataFrame, pd.Series]:
    # 1) ESG scores
    if use_mock:
        raw_esg = extractingesg.build_mock_raw(tickers, seed=seed)
    else:
        raw_esg = extractingesg.download_and_extract(tickers)
    esg_results = extractingesg.run_esg_analysis(raw_esg)

    # 2) Prices
    price_df = utilityfunc.download_price_data(tickers, start_date, end_date)
    if price_df is None or price_df.empty:
        raise ValueError("No price data found for the given dates/tickers.")

    # 3) Optimize (ESG-aware, fallback)
    try:
        weights = logicopt.optimize_portfolio(price_df, esg_scores=esg_results)
    except Exception:
        weights = logicopt.optimize_portfolio(price_df)

    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights)
    return esg_results, price_df, weights.sort_values(ascending=False)

# ---------------- Page config ----------------
st.set_page_config(
    page_title="EcoAlpha — ESG-Informed Portfolio Optimizer",
    page_icon="🌿",
    layout="wide",
)

# ---------------- Global CSS (Tropir-like) ----------------
st.markdown(
    """
<style>
:root{
  --primary:#7C3AED; --accent:#06B6D4; --bg:#0B0F17;
  --text:#E5E7EB; --muted:#9CA3AF;
}
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 10% -10%, #171B24 0%, #0B0F17 40%, #0B0F17 100%) fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1200px; }

/* Gradient headline */
.h-hero {
  font-weight: 800; letter-spacing:-.02em; line-height:1.05;
  background: linear-gradient(90deg, var(--accent), var(--primary));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* Glass card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 20px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.45);
  backdrop-filter: blur(10px);
}

/* Pill + button */
.pill { display:inline-flex; gap:.5rem; align-items:center; padding:.5rem .8rem; border-radius:999px;
        background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.08); }
.button-gradient {
  display:inline-block; padding:.8rem 1.1rem; border-radius:999px; font-weight:700; border:0;
  background: linear-gradient(90deg, #10B981, #34D399); /* green shades */
  color:white;
}
.button-gradient:hover { filter: brightness(1.06); }

/* Decorative blobs */
.blob-a, .blob-b { position: fixed; filter: blur(70px); opacity:.35; z-index:-1; }
.blob-a { width: 420px; height: 420px; left:-100px; top:-60px; background: radial-gradient(circle at 30% 30%, #7C3AED55, transparent 60%); }
.blob-b { width: 380px; height: 380px; right:-80px; bottom:-40px; background: radial-gradient(circle at 70% 70%, #06B6D455, transparent 60%); }

.dataframe tbody, .dataframe thead { color: var(--text); }
.subtle { color: var(--muted); }
</style>
<div class="blob-a"></div><div class="blob-b"></div>
""",
    unsafe_allow_html=True,
)

# ---------------- HERO ----------------
left, right = st.columns([1.2, 1])
with left:
    st.markdown('<h1 class="h-hero">ESG-Informed Portfolio Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Build balanced portfolios that respect sustainability signals — with clear risk/return controls.</p>', unsafe_allow_html=True)
    st.markdown('<a class="button-gradient" href="#optimize">Run Optimization ▶</a>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="card" style="text-align:center; min-height:170px;">📈<br><b>EcoAlpha</b><br><span class="subtle">Smart weights from your tickers + ESG text</span></div>', unsafe_allow_html=True)

st.write("")  # spacer

# ---------------- Feature chips ----------------
c1, c2, c3 = st.columns(3)
c1.markdown('<div class="card"><div class="pill">🌿 <b>ESG aware</b></div><p class="subtle">Uses NLP-derived E/S/G scores</p></div>', unsafe_allow_html=True)
c2.markdown('<div class="card"><div class="pill">⚖️ <b>Risk controls</b></div><p class="subtle">Target return with variance & caps</p></div>', unsafe_allow_html=True)
c3.markdown('<div class="card"><div class="pill">⚡ <b>Fast</b></div><p class="subtle">Optimizes in seconds</p></div>', unsafe_allow_html=True)

st.write("")

# ---------------- Form (in-page; no sidebar) ----------------
st.markdown('<h3 id="optimize">Take a look</h3>', unsafe_allow_html=True)
with st.container():
    with st.form("run"):
        default_tickers = "AAPL, MSFT, TSLA, GOOG, AMZN"
        tickers_raw = st.text_input("Tickers (comma-separated)", value=default_tickers)
        today = date.today()
        start_dt = st.date_input("Start date", value=today - timedelta(days=365))
        end_dt = st.date_input("End date", value=today)
        use_live = st.toggle("Use live ESG data", value=False, help="Off = mock ESG for speed")
        seed = st.number_input("Random seed (mock ESG)", value=2025, step=1)
        submit = st.form_submit_button("Optimize", use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if submit:
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
        elif start_dt >= end_dt:
            st.error("Start date must be before end date.")
        else:
            with st.spinner("Crunching numbers…"):
                try:
                    esg_results, price_df, weights = run_pipeline(
                        tickers,
                        start_dt.isoformat(),
                        end_dt.isoformat(),
                        use_mock=not use_live,
                        seed=int(seed),
                    )

                    st.subheader("ESG Scores")
                    esg_df = pd.DataFrame(esg_results).T[["E", "S", "G"]].sort_index().round(3)
                    st.dataframe(esg_df, use_container_width=True)

                    st.caption(f"Retrieved **{len(price_df)}** rows of price data.")

                    # Price chart preview
                    try:
                        st.subheader("Price Trend (preview)")
                        preview = price_df.copy()
                        if isinstance(preview.columns, pd.MultiIndex):
                            if ("Adj Close" in preview.columns.get_level_values(-1)):
                                preview = preview.xs("Adj Close", axis=1, level=-1)
                            else:
                                last_level = preview.columns.levels[-1][0]
                                preview = preview.xs(last_level, axis=1, level=-1)
                        if preview.shape[1] > 5:
                            preview = preview.iloc[:, :5]
                        st.line_chart(preview, use_container_width=True)
                    except Exception:
                        st.caption("Price chart preview not available for this data shape.")
                        st.dataframe(price_df.head(), use_container_width=True)

                    st.subheader("Optimized Portfolio Weights")
                    wdf = weights.to_frame("Weight").applymap(lambda x: round(float(x), 4))
                    st.bar_chart(wdf, use_container_width=True)
                    st.dataframe(wdf.style.format({"Weight": "{:.4f}"}), use_container_width=True)

                    st.success("Optimization complete ✅")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown('<div style="margin-top:3rem; text-align:center; color:#9CA3AF;">🌿 EcoAlpha — built for transparent sustainable investing.</div>', unsafe_allow_html=True)
