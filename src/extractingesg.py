import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK sentiment data if not already present
nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def download_and_extract(tickers):
    """
    Download ESG-related text for each ticker.
    Placeholder implementation — replace with real API/news.
    """
    texts = {}
    for ticker in tickers:
        texts[ticker] = f"{ticker} has initiatives for environmental sustainability, social equity, and strong governance."
    return texts

def run_esg_analysis(texts):
    """
    Use NLP to assign ESG scores based on text sentiment and keywords.
    Scores are normalized between 0 and 1.
    """
    esg_scores = {}

    for ticker, text in texts.items():
        lower_text = text.lower()

        # Base sentiment score
        sentiment = sia.polarity_scores(text)["compound"]
        sentiment = max(min((sentiment + 1) / 2, 1), 0)  # Normalize 0-1

        # Keyword detection for E, S, G
        e_keywords = ["environment", "climate", "sustainab", "carbon", "green"]
        s_keywords = ["social", "equity", "diversity", "community", "employee"]
        g_keywords = ["governance", "board", "transparency", "ethics", "compliance"]

        def keyword_score(keywords):
            return min(1.0, sum(1 for kw in keywords if kw in lower_text) / len(keywords) + 0.3)

        e_score = 0.5 * sentiment + 0.5 * keyword_score(e_keywords)
        s_score = 0.5 * sentiment + 0.5 * keyword_score(s_keywords)
        g_score = 0.5 * sentiment + 0.5 * keyword_score(g_keywords)

        esg_scores[ticker] = {"E": round(e_score, 2), "S": round(s_score, 2), "G": round(g_score, 2)}

    return esg_scores
