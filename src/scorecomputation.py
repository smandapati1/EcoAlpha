import json
import os

def extract_esg_scores_from_texts(texts):
    """
    Extracts basic ESG scores from input text per ticker.
    Scores are based on frequency of ESG-related keywords.
    """
    scores = {}

    for ticker, content in texts.items():
        content_lower = content.lower()

        # Simple keyword frequency-based scoring
        scores[ticker] = {
            "E": content_lower.count("environment") + content_lower.count("climate"),
            "S": content_lower.count("social") + content_lower.count("community") + content_lower.count("diversity"),
            "G": content_lower.count("governance") + content_lower.count("board") + content_lower.count("ethics")
        }

    return scores

def save_esg_scores(scores, output_path="data/processed/esg_scores.json"):
    """
    Saves ESG scores to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
