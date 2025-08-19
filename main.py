# main.py
import argparse
from datetime import date, timedelta
from typing import List, Tuple, Dict

import pandas as pd

# Your modules
from src import extractingesg, utilityfunc, logicopt

# ---------- CORE PIPELINE ----------
def run_pipeline(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_mock: bool = True,
    seed: int = 2025,
) -> Tuple[Dict, pd.DataFrame, pd.Series]:
    """
    Returns (esg_results, price_df, weights)

    esg_results: nested dict like {'AAPL': {'E':..., 'S':..., 'G':...}, ...}
    price_df: DataFrame with OHLCV or adjusted close (as your utility returns)
    weights: pd.Series with ticker -> weight
    """
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

    # 3) Optimize (try ESG-aware first, then fallback)
    try:
        weights = logicopt.optimize_portfolio(price_df, esg_scores=esg_results)
    except Exception as e:
        # Fallback to non-ESG if Max Sharpe with constraints fails
        weights = logicopt.optimize_portfolio(price_df)

    # Ensure weights are a pd.Series for nicer display
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights)
    weights = weights.sort_values(ascending=False)

    return esg_results, price_df, weights


# ---------- STREAMLIT UI ----------
def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="EcoAlpha — ESG-Informed Portfolio Optimizer",
        page_icon="🌿",
        layout="wide",
    )

    # ---------- THEME + GLOBAL CSS ----------
    st.markdown(
        """
<style>
:root{
  --primary:#7C3AED;        /* violet */
  --accent:#06B6D4;         /* cyan */
  --bg:#0B0F17;             /* page background */
  --panel:#111827CC;        /* glass card */
  --text:#E5E7EB;
  --muted:#9CA3AF;
  --ring: 0 0 0 2px rgba(124,58,237,.4);
}
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 10% -10%, #171B24 0%, #0B0F17 40%, #0B0F17 100%) no-repeat fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1200px; }

/* Gradient headline */
.h-hero {
  font-weight: 800; letter-spacing:-.02em; line-height:1.05;
  background: linear-gradient(90deg, var(--accent), var(--primary));
  -webkit-background-clip:text; background-clip:text; color: transparent;
}

/* Subheading */
.subtle { color: var(--muted); font-size: 1.05rem }

/* Glass cards */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 10px 30px rgba(0,0,0,.45);
  border-radius: 20px; padding: 20px;
  backdrop-filter: blur(10px);
}

/* Gradient pill buttons */
.button-gradient {
  display:inline-block; padding: .7rem 1rem; border-radius: 999px;
  background: linear-gradient(90deg, var(--primary), var(--accent));
  color: white; font-weight: 700; text-decoration:none; border:0;
}
.button-gradient:hover { filter: brightness(1.06); box-shadow: var(--ring); }

/* Metric pills */
.pill {
  display:inline-flex; gap:.5rem; align-items:center;
  padding:.5rem .8rem; border-radius:999px;
  background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.08);
}

/* Blurry decor blobs */
.blob-a, .blob-b { position: fixed; filter: blur(70px); opacity:.35; z-index:-1; }
.blob-a { width: 420px; height: 420px; left: -100px; top: -60px; background: radial-gradient(circle at 30% 30%, #7C3AED55, transparent 60%); }
.blob-b { width: 380px; height: 380px; right: -80px; bottom: -40px; background: radial-gradient(circle at 70% 70%, #06B6D455, transparent 60%); }

.dataframe tbody, .dataframe thead { color: var(--text) }
</style>
<div class="blob-a"></div><div class="blob-b"></div>
""",
        unsafe_allow_html=True,
    )

    # ---------- HERO ----------
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            '<h1 class="h-hero">ESG-Informed Portfolio Optimizer</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="subtle">Build balanced portfolios that respect sustainability signals — with clear risk/return controls.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<a class="button-gradient" href="#optimize">Run Optimization ▶</a>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="card" style="text-align:center; min-height:170px;">'
            '📈<br><b>EcoAlpha</b><br><span class="subtle">Smart weights from your tickers + ESG text</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")  # spacer

    # ---------- QUICK BADGES ----------
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="card"><div class="pill">🌿 <b>ESG aware</b></div>'
        '<p class="subtle">Uses NLP-derived E/S/G scores</p></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="card"><div class="pill">⚖️ <b>Risk controls</b></div>'
        '<p class="subtle">Target return with variance & caps</p></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        '<div class="card"><div class="pill">⚡ <b>Fast</b></div>'
        '<p class="subtle">Optimizes in seconds</p></div>',
        unsafe_allow_html=True,
    )

    st.write("")  # spacer

    # ---------- FORM ----------
    st.markdown('<h3 id="optimize">Try it</h3>', unsafe_allow_html=True)
    with st.container():
        with st.form("run"):
            default_tickers = "AAPL, MSFT, TSLA"
            ti_raw = st.text_input("Tickers (comma-separated)", value=default_tickers)
            today = date.today()
            default_start = today - timedelta(days=365)
            start_dt = st.date_input("Start date", value=default_start)
            end_dt = st.date_input("End date", value=today)
            use_live = st.toggle("Use live ESG data", value=False, help="Off = mock ESG for speed")
            seed = st.number_input("Random seed (mock ESG)", value=2025, step=1)
            submit = st.form_submit_button("Optimize", use_container_width=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if submit:
            tickers = [t.strip().upper() for t in ti_raw.split(",") if t.strip()]
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

                        # --- Results: ESG table
                        st.subheader("ESG Scores")
                        esg_df = (
                            pd.DataFrame(esg_results)
                            .T[["E", "S", "G"]]
                            .sort_index()
                            .round(2)
                        )
                        st.dataframe(esg_df, use_container_width=True)

                        # --- Results: Prices / rows info
                        st.caption(f"Retrieved **{len(price_df)}** rows of price data.")

                        # --- Chart: simple price chart if 'Adj Close' or similar wide format exists
                        # Try to find a sensible column layout; otherwise show head
                        try:
                            # If price_df is wide with multiindex columns or multiple symbols,
                            # try a simple line on the first found column per ticker.
                            st.subheader("Price Trend (preview)")
                            preview = price_df.copy()
                            # If it's multiindex columns, pick 'Adj Close' level if present
                            if isinstance(preview.columns, pd.MultiIndex):
                                if ("Adj Close" in preview.columns.get_level_values(-1)):
                                    preview = preview.xs("Adj Close", axis=1, level=-1)
                                else:
                                    # fallback to the last level first column group
                                    last_level = preview.columns.levels[-1][0]
                                    preview = preview.xs(last_level, axis=1, level=-1)
                            # If it's single-index and too many columns, just pick first 5
                            if preview.shape[1] > 5:
                                preview = preview.iloc[:, :5]
                            st.line_chart(preview, use_container_width=True)
                        except Exception:
                            st.caption("Price chart preview not available for this data shape.")
                            st.dataframe(price_df.head(), use_container_width=True)

                        # --- Results: Weights
                        st.subheader("Optimized Portfolio Weights")
                        wdf = weights.to_frame("Weight").applymap(lambda x: round(float(x), 4))
                        st.bar_chart(wdf, use_container_width=True)
                        st.dataframe(wdf.style.format({"Weight": "{:.4f}"}), use_container_width=True)

                        # --- Notes
                        st.success("Optimization complete ✅")

                    except Exception as e:
                        st.error(f"Something went wrong: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- FOOTER ----------
    st.markdown(
        """
<div style="margin-top:3rem; text-align:center; color:#9CA3AF;">
  🌿 EcoAlpha — built for transparent sustainable investing.
</div>
""",
        unsafe_allow_html=True,
    )


# ---------- CLI (original behavior) ----------
def run_cli():
    parser = argparse.ArgumentParser(description="ESG Portfolio Optimization CLI")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of stock tickers")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--live", action="store_true", help="Use live ESG data instead of mock")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for mock ESG")

    args = parser.parse_args()

    print(f"\n🌿 ESG-based portfolio optimization for: {args.tickers}\n")

    try:
        esg_results, price_df, weights = run_pipeline(
            args.tickers, args.start_date, args.end_date, use_mock=not args.live, seed=args.seed
        )
    except Exception as e:
        print(f"⚠️ {e}")
        return

    print("ESG Scores:", esg_results)
    print(f"Retrieved {len(price_df)} rows of stock price data.")
    print("\nOptimized Portfolio Weights:")
    print(weights)


if __name__ == "__main__":
    # If launched by Streamlit (`streamlit run main.py`), Streamlit will call run_streamlit().
    # If executed directly as a script (python main.py ...), we run CLI.
    # Streamlit sets the global __name__ == "__main__" too, but it doesn't pass CLI args;
    # the safe approach is to detect Streamlit through its runtime flag.
    try:
        # This import only succeeds during a Streamlit session
        import streamlit.runtime.scriptrunner.script_runner as _st_sr  # type: ignore
        # If we get here, assume we're in a Streamlit context:
        run_streamlit()
    except Exception:
        # Fallback to CLI mode
        run_cli()
