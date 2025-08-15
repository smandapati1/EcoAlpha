# src/extractingesg.py
"""
ESG signal extraction & fusion.

This module:
- Pulls Yahoo Finance sustainability data (if available) and normalizes to 0..1.
- Scores recent Yahoo Finance news headlines per ticker with VADER sentiment,
  mapping to E/S/G via keyword buckets and applying a recency decay.
- Optionally scores local text filings (data/raw/esg_reports/TICKER.txt) with
  the same keyword-bucket approach.
- Fuses the three signals with configurable weights, then peer-normalizes
  across the input tickers so outputs are varied and comparable.
- Provides a build_mock_raw(tickers, seed) helper to generate varied mock
  signals for testing/demos (pair with --mock_esg in main.py).
"""

from __future__ import annotations
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
# Tunable weights / settings
# ----------------------------
SUSTAIN_WEIGHT = 0.50   # Yahoo sustainability pillar
NEWS_WEIGHT    = 0.35   # News sentiment pillar
FILINGS_WEIGHT = 0.15   # Local 10-K text pillar
DECAY_DAYS     = 21     # News recency half-life (~1 trading month)

# ESG keyword buckets for mapping sentiment into pillars
E_KEYWORDS = ("environment", "climate", "emission", "carbon", "sustainab", "renewable", "green", "energy")
S_KEYWORDS = ("social", "community", "diversity", "inclusion", "labor", "employee", "human rights", "safety")
G_KEYWORDS = ("governance", "board", "audit", "ethic", "compliance", "transparen", "shareholder", "corruption")


# ----------------------------
# Helpers
# ----------------------------
def _norm01(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5
    return (v - vmin) / (vmax - vmin)

def _headline_to_bucket_scores(headline: str) -> Tuple[float, float, float]:
    """
    Map a headline's sentiment to E/S/G buckets based on keyword presence.
    Returns (E,S,G) each in [0,1].
    """
    h_low = headline.lower()
    sent = _SIA.polarity_scores(headline)["compound"]  # -1..1
    s01 = (sent + 1) / 2                               # 0..1

    has_e = any(k in h_low for k in E_KEYWORDS)
    has_s = any(k in h_low for k in S_KEYWORDS)
    has_g = any(k in h_low for k in G_KEYWORDS)

    # If none match, treat as mild diffuse impact
    if not (has_e or has_s or has_g):
        return (0.33 * s01, 0.33 * s01, 0.33 * s01)

    e = s01 if has_e else 0.0
    s = s01 if has_s else 0.0
    g = s01 if has_g else 0.0
    return (e, s, g)

def _decay_weight(age_days: float) -> float:
    # exponential decay so fresher news counts more
    if age_days < 0:
        age_days = 0
    return math.exp(-age_days / DECAY_DAYS)

def _score_local_text_block(text: str) -> Tuple[float, float, float]:
    """
    Score a longer text block: average sentence sentiment, weighted by keyword buckets.
    Output three scores in [0,1] for E/S/G.
    """
    if not text or not text.strip():
        return (0.5, 0.5, 0.5)

    # quick sentence-ish split
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.split()) >= 4]
    if not sentences:
        sentences = [text]

    e_vals, s_vals, g_vals = [], [], []
    for s in sentences[:300]:  # cap for speed
        e, so, g = _headline_to_bucket_scores(s)
        e_vals.append(e); s_vals.append(so); g_vals.append(g)

    # overall sentiment bias
    compound = _SIA.polarity_scores(text)["compound"]
    bias = (compound + 1) / 2  # 0..1

    def _blend(vals):
        base = sum(vals)/len(vals) if vals else 0.5
        return 0.7 * base + 0.3 * bias

    e = _blend(e_vals); s = _blend(s_vals); g = _blend(g_vals)
    return (max(0, min(1, e)), max(0, min(1, s)), max(0, min(1, g)))


# ----------------------------
# Data fetchers (Sustainability / News / Local filings)
# ----------------------------
def _fetch_sustainability_esg(ticker: str) -> Dict[str, float]:
    """
    Pull Yahoo Finance sustainability data and normalize to 0..1 where possible.
    Returns partial dict (any subset of {"E","S","G"}) or {} if unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        sustain = t.sustainability
        if isinstance(sustain, pd.DataFrame):
            d = sustain.to_dict().get("Value", {})
            # Yahoo often uses 0..100-ish scales
            out = {}
            if "environmentScore" in d:
                out["E"] = round(_norm01(float(d["environmentScore"]), 0, 100), 3)
            if "socialScore" in d:
                out["S"] = round(_norm01(float(d["socialScore"]), 0, 100), 3)
            if "governanceScore" in d:
                out["G"] = round(_norm01(float(d["governanceScore"]), 0, 100), 3)
            return out
    except Exception:
        pass
    return {}

def _fetch_news_esg(ticker: str) -> Tuple[float, float, float]:
    """
    Score recent Yahoo Finance news headlines for a ticker with recency decay.
    Returns (E,S,G) in [0,1].
    """
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", None) or []
    except Exception:
        news = []

    if not news:
        return (0.5, 0.5, 0.5)

    now = time.time()
    e_acc = s_acc = g_acc = 0.0
    w_acc = 0.0
    for item in news[:40]:
        title = item.get("title", "") or ""
        ts = item.get("providerPublishTime", now)
        try:
            age_days = max(0.0, (now - float(ts)) / 86400.0)
        except Exception:
            age_days = 0.0
        w = _decay_weight(age_days)
        e, s, g = _headline_to_bucket_scores(title)
        e_acc += w * e; s_acc += w * s; g_acc += w * g
        w_acc += w

    if w_acc == 0:
        return (0.5, 0.5, 0.5)
    return (e_acc / w_acc, s_acc / w_acc, g_acc / w_acc)

def _fetch_local_filing_esg(ticker: str) -> Tuple[float, float, float]:
    """
    If local text exists at data/raw/esg_reports/TICKER.txt, score it.
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
# Public API used by main.py
# ----------------------------
def download_and_extract(tickers: List[str]) -> Dict[str, dict]:
    """
    Collect raw ESG sources per ticker.
    Output structure per ticker: {"sustain": dict(E/S/G partial),
                                  "news": (E,S,G),
                                  "filing": (E,S,G)}
    """
    raw = {}
    for tk in tickers:
        raw[tk] = {
            "sustain": _fetch_sustainability_esg(tk),
            "news": _fetch_news_esg(tk),
            "filing": _fetch_local_filing_esg(tk),
        }
    return raw

def run_esg_analysis(raw_data: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """
    Fuse sustainability, news, and filings into final E/S/G per ticker,
    then peer-normalize each pillar across the input set for comparability.
    Returns {ticker: {"E": x, "S": y, "G": z}} with values in [0,1].
    """
    if not raw_data:
        return {}

    # First pass: weighted fusion per ticker (handle partial sustainability gracefully)
    combined = {}
    for tk, src in raw_data.items():
        sust = src.get("sustain", {})
        e_sust = sust.get("E", 0.5); s_sust = sust.get("S", 0.5); g_sust = sust.get("G", 0.5)
        e_news, s_news, g_news = src.get("news", (0.5, 0.5, 0.5))
        e_file, s_file, g_file = src.get("filing", (0.5, 0.5, 0.5))

        e = SUSTAIN_WEIGHT*e_sust + NEWS_WEIGHT*e_news + FILINGS_WEIGHT*e_file
        s = SUSTAIN_WEIGHT*s_sust + NEWS_WEIGHT*s_news + FILINGS_WEIGHT*s_file
        g = SUSTAIN_WEIGHT*g_sust + NEWS_WEIGHT*g_news + FILINGS_WEIGHT*g_file
        combined[tk] = {"E": e, "S": s, "G": g}

    # Second pass: peer normalization per pillar to produce variation across tickers
    df = pd.DataFrame(combined).T  # rows=tickers, cols=E/S/G
    for col in ("E", "S", "G"):
        vmin, vmax = float(df[col].min()), float(df[col].max())
        df[col] = df[col].apply(lambda v: _norm01(v, vmin, vmax))

    out = {tk: {"E": round(float(df.loc[tk, "E"]), 3),
                "S": round(float(df.loc[tk, "S"]), 3),
                "G": round(float(df.loc[tk, "G"]), 3)} for tk in df.index}
    return out


# ----------------------------
# Mock source builder (for demos/tests)
# ----------------------------
import random

def build_mock_raw(tickers: List[str], seed: int = 1234) -> Dict[str, dict]:
    """
    Build varied, deterministic mock ESG sources per ticker:
      - 'sustain': dict with E/S/G in [0,1]
      - 'news'   : tuple(E,S,G) in [0,1]
      - 'filing' : tuple(E,S,G) in [0,1]
    This mirrors download_and_extract() so run_esg_analysis() just works.
    """
    random.seed(seed)
    raw = {}
    for tk in tickers:
        sustain = {
            "E": round(random.uniform(0.30, 0.95), 3),
            "S": round(random.uniform(0.20, 0.90), 3),
            "G": round(random.uniform(0.25, 0.92), 3),
        }
        news = (
            round(random.uniform(0.20, 0.90), 3),
            round(random.uniform(0.20, 0.90), 3),
            round(random.uniform(0.20, 0.90), 3),
        )
        filing = (
            round(random.uniform(0.20, 0.90), 3),
            round(random.uniform(0.20, 0.90), 3),
            round(random.uniform(0.20, 0.90), 3),
        )
        raw[tk] = {"sustain": sustain, "news": news, "filing": filing}
    return raw
