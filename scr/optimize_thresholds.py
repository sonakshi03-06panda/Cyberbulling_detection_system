import torch
import pandas as pd
import numpy as np
import re
import emoji
from sklearn.metrics import f1_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from scipy.optimize import minimize_scalar
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = emoji.demojize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_logits_and_labels(model_path, csv_path, sample_size=5000):
    """Get raw logits and true labels for threshold optimization."""
    df = pd.read_csv(csv_path)
    df = df[["comment_text"] + LABELS].dropna()
    df["comment_text"] = df["comment_text"].apply(clean_text)
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    texts = df["comment_text"].values
    true_labels = df[LABELS].astype(int).values
    
    print(f"Loading model from {model_path}...")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
    
    all_logits = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting logits"):
            batch = {k: torch.tensor(v[i:i+batch_size]).to(DEVICE) for k, v in encodings.items()}
            outputs = model(**batch)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
    
    all_logits = np.vstack(all_logits)
    return all_logits, true_labels


def optimize_thresholds(logits, true_labels, labels=LABELS):
    """Find optimal threshold per label to maximize F1-score."""
    optimal_thresholds = {}
    
    for i, label in enumerate(labels):
        y_true = true_labels[:, i]
        y_scores = logits[:, i]
        
        def neg_f1(threshold):
            y_pred = (y_scores > threshold).astype(int)
            return -f1_score(y_true, y_pred, zero_division=0)
        
        # Search over [0, 1]
        result = minimize_scalar(neg_f1, bounds=(0, 1), method="bounded")
        optimal_threshold = result.x
        best_f1 = -result.fun
        
        optimal_thresholds[label] = {
            "threshold": float(optimal_threshold),
            "f1": float(best_f1)
        }
        
        print(f"{label:15} - Optimal threshold: {optimal_threshold:.4f}, F1: {best_f1:.4f}")
    
    return optimal_thresholds


if __name__ == "__main__":
    # Get logits from validation set
    logits, labels = get_logits_and_labels("models/final_model", "data/train.csv", sample_size=5000)
    
    # Optimize thresholds
    print("\nOptimizing per-label thresholds...\n")
    thresholds = optimize_thresholds(logits, labels)
    
    # Save thresholds
    import json
    with open("models/optimal_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"\nThresholds saved to models/optimal_thresholds.json")
