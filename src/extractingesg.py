from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import os
import json

def load_finbert_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-esg", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-esg")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

def extract_sentences(text):
    return [sent.strip() for sent in text.split('.') if len(sent.split()) > 3]

def score_text_file(file_path, output_path, nlp_pipeline):
    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()
    sentences = extract_sentences(text)
    scored = []

    for sent in sentences:
        result = nlp_pipeline(sent)
        result[0]["sentence"] = sent
        scored.append(result[0])

    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2)