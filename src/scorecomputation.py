from collections import Counter
import os
import json


def compute_esg_score(sentiment_counts):
    
    pos = sentiment_counts.get("POSITIVE", 0)
    neu = sentiment_counts.get("NEUTRAL", 0)
    neg = sentiment_counts.get("NEGATIVE", 0)

    total = pos + neu + neg
   
    if total == 0:
        return 0.0 

    score = (1 * pos + 0.5 * neu + 0 * neg) / total
    return round(score, 4)


def process_sentiment_folder(sentiment_folder, output_file):
   
    company_scores = {}

    for filename in os.listdir(sentiment_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(sentiment_folder, filename)
            with open(filepath, "r") as f:
                sentiments = json.load(f)

            sentiment_labels = [entry["label"] for entry in sentiments]
            sentiment_counts = dict(Counter(sentiment_labels))
            score = compute_esg_score(sentiment_counts)

            company_name = filename.replace("_sentiment.json", "")
            company_scores[company_name] = {
                "score": score,
                "counts": sentiment_counts
            }

    with open(output_file, "w") as out:
        json.dump(company_scores, out, indent=4)

    print(f"[✓] ESG scores computed and saved to {output_file}")