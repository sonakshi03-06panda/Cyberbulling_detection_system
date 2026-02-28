import torch
import pandas as pd
import numpy as np
import re
import emoji
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = emoji.demojize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_performance():
    # Load data
    df = pd.read_csv("data/train.csv")
    df = df[["comment_text"] + LABELS].dropna()
    df["comment_text"] = df["comment_text"].apply(clean_text)
    
    # Sample 20k as in training
    df_small = df.sample(n=20000, random_state=42)
    
    # Split same way as training
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_small["comment_text"].values,
        df_small[LABELS].values,
        test_size=0.1,
        random_state=42
    )
    
    # Load trained model and tokenizer
    print("Loading model from models/final_model...")
    model = DistilBertForSequenceClassification.from_pretrained("models/final_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/final_model")
    model.to(DEVICE)
    model.eval()
    
    # Tokenize validation set (use full val set for analysis)
    print("Tokenizing validation set...")
    val_encodings = tokenizer(
        list(val_texts),
        truncation=True,
        padding=True,
        max_length=128
    )
    
    # Get predictions
    print("Running inference on validation set...")
    all_preds = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(val_texts), batch_size):
            batch_encodings = {
                key: torch.tensor(val_encodings[key][i:i+batch_size]).to(DEVICE)
                for key in val_encodings.keys()
            }
            outputs = model(**batch_encodings)
            logits = outputs.logits.cpu().numpy()
            preds = (logits > 0).astype(int)
            all_preds.append(preds)
    
    all_preds = np.vstack(all_preds)
    
    # Calculate per-label metrics
    print("\n" + "="*70)
    print("PER-LABEL PERFORMANCE ANALYSIS")
    print("="*70)
    
    for i, label in enumerate(LABELS):
        y_true = val_labels[:, i]
        y_pred = all_preds[:, i]
        
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        support = np.sum(y_true)
        
        print(f"\n{label.upper()}:")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Support:   {support} ({100*support/len(y_true):.1f}%)")
    
    # Weighted averages
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    f1_micro = f1_score(val_labels, all_preds, average="micro")
    f1_macro = f1_score(val_labels, all_preds, average="macro")
    f1_weighted = f1_score(val_labels, all_preds, average="weighted")
    
    print(f"F1 Micro (unweighted): {f1_micro:.4f}")
    print(f"F1 Macro (per-label):  {f1_macro:.4f}")
    print(f"F1 Weighted (by class support): {f1_weighted:.4f}")
    
    # Class imbalance analysis
    print("\n" + "="*70)
    print("CLASS IMBALANCE")
    print("="*70)
    for i, label in enumerate(LABELS):
        pos_rate = np.mean(val_labels[:, i])
        print(f"{label:15} - Positive rate: {100*pos_rate:5.2f}%")


if __name__ == "__main__":
    analyze_performance()
