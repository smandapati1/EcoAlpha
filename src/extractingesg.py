import os
import time
import math
from typing import Dict, List, Tuple
import yfinance as yf
import pandas as pd

# NLP: lightweight sentiment for headlines and text
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
_SIA = SentimentIntensityAnalyzer()

# ----------------------------
# Config knobs (tune as you like)
# ----------------------------
SUSTAIN_WEIGHT = 0.5   # Yahoo sustainability signal
NEWS_WEIGHT    = 0.35  # News sentiment signal
FILINGS_WEIGHT = 0.15  # Local 10-K text signal
DECAY_DAYS     = 21    # News recency decay window (about 1 month trading days)

# ESG keyword buckets for mapping headline/text sentiment into E/S/G
E_KEYWORDS = ("environment", "climate", "emission", "carbon", "sustainab", "renewable", "green", "energy")
S_KEYWORDS = ("social", "community", "diversity", "inclusion", "labor", "employee", "human rights", "safety")
G_KEYWORDS = ("governance", "board", "audit", "ethic", "compliance", "transparen", "shareholder", "corruption")

def _sigmoid(x: float) -> float:
    # squashes sentiment to (0,1)
    return 1 / (1 + math.exp(-x))

def _norm01(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5
    return (v - vmin) / (vmax - vmin)

def _headline_to_bucket_scores(headline: str) -> Tuple[float, float, float]:
    """
    Map a headline's sentiment to E/S/G buckets based on keyword presence.
    Returns (E,S,G) in [0,1].
    """
    h_low = headline.lower()
    sent = _SIA.polarity_scores(headline)["compound"]  # -1..1
    s01 = (sent + 1) / 2                               # 0..1

    has_e = any(k in h_low for k in E_KEYWORDS)
    has_s = any(k in h_low for k in S_KEYWORDS)
    has_g = any(k in h_low for k in G_KEYWORDS)

    # If none match, treat as neutral small impact across all
    if not (has_e or has_s or has_g):
        return (0.33*s01, 0.33*s01, 0.33*s01)

    e = s01 if has_e else 0.0
    s = s01 if has_s else 0.0
    g = s01 if has_g else 0.0
    return (e, s, g)

def _decay_weight(age_days: float) -> float:
    # exponential decay so fresher news counts more
    if age_days < 0:  # safety
        age_days = 0
    return math.exp(-age_days / DECAY_DAYS)

def _score_local_text_block(text: str) -> Tuple[float, float, float]:
    """
    Score a longer text block: average sentence sentiment, weight by keyword buckets.
    Output three scores in [0,1] for E/S/G.
    """
    if not text or not text.strip():
        return (0.5, 0.5, 0.5)

    # quick sentence-ish split
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.split()) >= 4]
    if not sentences:
        sentences = [text]

    e_vals, s_vals, g_vals = [], [], []
    for s in sentences[:300]:  # cap to keep it fast
        e, so, g = _headline_to_bucket_scores(s)
        e_vals.append(e); s_vals.append(so); g_vals.append(g)

    # average and lightly re-center with overall sentiment
    compound = _SIA.polarity_scores(text)["compound"]
    bias = (compound + 1)/2  # 0..1
    e = 0.7*(sum(e_vals)/len(e_vals) if e_vals else 0.5) + 0.3*bias
    s = 0.7*(sum(s_vals)/len(s_vals) if s_vals else 0.5) + 0.3*bias
    g = 0.7*(sum(g_vals)/len(g_vals) if g_vals else 0.5) + 0.3*bias
    return (max(0,min(1,e)), max(0,min(1,s)), max(0,min(1,g)))

def _fetch_sustainability_esg(ticker: str) -> Dict[str, float]:
    """
    Pull Yahoo Finance sustainability data if available and normalize to 0..1.
    If missing, return {} so caller can fallback.
    """
    try:
        t = yf.Ticker(ticker)
        sustain = t.sustainability
        if isinstance(sustain, pd.DataFrame):
            d = sustain.to_dict().get("Value", {})
            # Yahoo fields are often on 0..100 or other scales; normalize defensively.
            e = d.get("environmentScore", None)
            s = d.get("socialScore", None)
            g = d.get("governanceScore", None)
            out = {}
            if e is not None: out["E"] = round(_norm01(float(e), 0, 100), 3)
            if s is not None: out["S"] = round(_norm01(float(s), 0, 100), 3)
            if g is not None: out["G"] = round(_norm01(float(g), 0, 100), 3)
            return out
    except Exception:
        pass
    return {}

def _fetch_news_esg(ticker: str) -> Tuple[float, float, float]:
    """
    Score recent Yahoo Finance news headlines for a ticker with recency decay.
    Returns (E,S,G) in 0..1.
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news or []  # list of dicts with 'title' and 'providerPublishTime'
    except Exception:
        news = []

    if not news:
        return (0.5, 0.5, 0.5)

    now = time.time()
    e_acc = s_acc = g_acc = 0.0
    w_acc = 0.0
    for item in news[:40]:
        title = item.get("title", "")
        ts = item.get("providerPublishTime", now)
        age_days = max(0.0, (now - float(ts)) / 86400.0)
        w = _decay_weight(age_days)
        e, s, g = _headline_to_bucket_scores(title)
        e_acc += w*e; s_acc += w*s; g_acc += w*g
        w_acc += w
    if w_acc == 0:
        return (0.5, 0.5, 0.5)
    return (e_acc/w_acc, s_acc/w_acc, g_acc/w_acc)

def _fetch_local_filing_esg(ticker: str) -> Tuple[float, float, float]:
    """
    If you have a local text file at data/raw/esg_reports/TICKER.txt, score it.
    """
    path = os.path.join("data", "raw", "esg_reports", f"{ticker}.txt")
    if not os.path.exists(path):
        return (0.5, 0.5, 0.5)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return _score_local_text_block(text)
    except Exception:
        return (0.5, 0.5, 0.5)

# ----------------------------
# Public API expected by main.py
# ----------------------------
def download_and_extract(tickers: List[str]) -> Dict[str, dict]:
    """
    For parity with your existing flow, this now returns a dict of raw sources
    so run_esg_analysis can combine them. Keys: 'sustain', 'news', 'filing' (per ticker).
    """
    raw = {}
    for tk in tickers:
        raw[tk] = {
            "sustain": _fetch_sustainability_esg(tk),          # dict with any of E/S/G present
            "news": _fetch_news_esg(tk),                       # tuple(E,S,G)
            "filing": _fetch_local_filing_esg(tk)              # tuple(E,S,G)
        }
    return raw

def run_esg_analysis(raw_data: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """
    Combine sustainability, news, and filings into final E/S/G per ticker.
    Then normalize across the peer set to ensure varied, comparable outputs.
    """
    # First pass: weighted combo per ticker
    combined = {}
    for tk, src in raw_data.items():
        # sustainability may be partial (e.g., only E present) — fill gracefully
        sust = src.get("sustain", {})
        e_sust = sust.get("E", 0.5); s_sust = sust.get("S", 0.5); g_sust = sust.get("G", 0.5)
        e_news, s_news, g_news = src.get("news", (0.5, 0.5, 0.5))
        e_file, s_file, g_file = src.get("filing", (0.5, 0.5, 0.5))

        e = SUSTAIN_WEIGHT*e_sust + NEWS_WEIGHT*e_news + FILINGS_WEIGHT*e_file
        s = SUSTAIN_WEIGHT*s_sust + NEWS_WEIGHT*s_news + FILINGS_WEIGHT*s_file
        g = SUSTAIN_WEIGHT*g_sust + NEWS_WEIGHT*g_news + FILINGS_WEIGHT*g_file
        combined[tk] = {"E": e, "S": s, "G": g}

    # Second pass: peer normalization (min-max per pillar) to create variation across the set
    if not combined:
        return {}

    df = pd.DataFrame(combined).T  # rows=tickers, cols=E/S/G
    for col in ("E","S","G"):
        vmin, vmax = float(df[col].min()), float(df[col].max())
        df[col] = df[col].apply(lambda v: _norm01(v, vmin, vmax))

    # Optional: overall composite can be the mean of pillars; keep pillars for transparency
    out = {}
    for tk, row in df.iterrows():
        out[tk] = {"E": round(row["E"], 3), "S": round(row["S"], 3), "G": round(row["G"], 3)}
    
    return out