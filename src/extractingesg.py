import os
from typing import Dict
import re
from transformers import pipeline

# Optional: Load FinBERT sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def download_and_extract(tickers: list, esg_dir: str = "data/raw/esg_reports") -> Dict[str, str]:
    """
    Loads ESG text reports for the given tickers from local files.

    Args:
        tickers (list): List of stock tickers
        esg_dir (str): Path to the ESG report directory

    Returns:
        Dict[str, str]: Dictionary of {ticker: ESG text}
    """
    texts = {}
    for ticker in tickers:
        path = os.path.join(esg_dir, f"{ticker}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                texts[ticker] = f.read()
        else:
            texts[ticker] = f"No ESG report found for {ticker}"
    return texts

def extract_esg_scores_from_texts(texts: Dict[str, str]) -> Dict[str, float]:
    """
    Extract ESG scores from the provided text using sentiment analysis.

    Args:
        texts (Dict[str, str]): Dictionary of {ticker: ESG report text}

    Returns:
        Dict[str, float]: Dictionary of {ticker: ESG score (0-1)}
    """
    scores = {}
    for ticker, text in texts.items():
        if "No ESG report found" in text:
            scores[ticker] = 0.0
            continue

        # Apply sentiment model (FinBERT)
        results = sentiment_pipeline(text[:1000])  # truncate for performance

        # Map sentiment to score
        positive = sum(1 for r in results if r['label'] == 'positive')
        total = len(results)
        score = positive / total if total else 0.0
        scores[ticker] = round(score, 2)
    return scores
