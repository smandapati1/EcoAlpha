import os
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def load_finbert_pipeline():

    model_name = "yiyanghkust/finbert-esg"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

def extract_esg_sentences(text, keywords):
   
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    return [s for s in sentences if any(k.lower() in s.lower() for k in keywords)]

def score_text_file(input_path, output_path, nlp_pipeline, keywords):
    
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    esg_sentences = extract_esg_sentences(text, keywords)
    if not esg_sentences:
        print(f"[Warning] No ESG sentences found in {input_path}")
        return

    results = []
    for sent in esg_sentences:
        prediction = nlp_pipeline(sent)[0]  
        prediction["sentence"] = sent
        results.append(prediction)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Info] Scored {len(results)} ESG sentences from {input_path}")