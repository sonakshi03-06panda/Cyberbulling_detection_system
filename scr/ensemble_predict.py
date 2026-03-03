import torch
import pandas as pd
import numpy as np
import re
import emoji
import json
from sklearn.metrics import f1_score, classification_report
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    RobertaTokenizerFast, RobertaForSequenceClassification
)
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = emoji.demojize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EnsemblePredictor:
    """Ensemble of DistilBERT and RoBERTa models with threshold optimization."""
    
    def __init__(self, distilbert_path, roberta_path, thresholds_path=None):
        print("Loading DistilBERT model...")
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_path)
        self.distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_path)
        self.distilbert_model.to(DEVICE)
        self.distilbert_model.eval()
        
        print("Loading RoBERTa model...")
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_path)
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained(roberta_path)
        self.roberta_model.to(DEVICE)
        self.roberta_model.eval()
        
        # Load optimal thresholds if available
        self.thresholds = None
        if thresholds_path and os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                threshold_data = json.load(f)
                self.thresholds = np.array([threshold_data[label]["threshold"] for label in LABELS])
            print(f"Loaded optimized thresholds from {thresholds_path}")
        else:
            self.thresholds = np.array([0.5] * len(LABELS))
            print("Using default threshold of 0.5 for all labels")
    
    def predict(self, texts, batch_size=32, ensemble_method="average"):
        """
        Predict labels for texts using ensemble.
        
        Args:
            texts: list of strings
            batch_size: inference batch size
            ensemble_method: 'average' or 'voting'
        
        Returns:
            predictions: (len(texts), len(LABELS)) binary array
        """
        # Get logits from both models
        print("Getting DistilBERT logits...")
        distilbert_logits = self._get_logits(
            texts, self.distilbert_model, self.distilbert_tokenizer, batch_size
        )
        
        print("Getting RoBERTa logits...")
        roberta_logits = self._get_logits(
            texts, self.roberta_model, self.roberta_tokenizer, batch_size
        )
        
        # Ensemble: average logits
        ensemble_logits = (distilbert_logits + roberta_logits) / 2.0
        
        # Apply optimized thresholds
        preds = (ensemble_logits > self.thresholds).astype(int)
        
        return preds
    
    def _get_logits(self, texts, model, tokenizer, batch_size):
        """Get raw logits from a model."""
        encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
        
        all_logits = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Inferencing", unit="batch"):
                batch = {k: torch.tensor(v[i:i+batch_size]).to(DEVICE) for k, v in encodings.items()}
                outputs = model(**batch)
                logits = outputs.logits.cpu().numpy()
                all_logits.append(logits)
        
        return np.vstack(all_logits)


def evaluate_ensemble(test_csv, labels_csv, ensemble):
    """Evaluate ensemble on test set."""
    print("Loading test data...")
    test_df = pd.read_csv(test_csv)
    labels_df = pd.read_csv(labels_csv)
    
    df = test_df.merge(labels_df, on="id")
    df = df[["comment_text"] + LABELS]
    df["comment_text"] = df["comment_text"].apply(clean_text)
    
    # Filter unlabeled
    initial = len(df)
    mask = ~(df[LABELS] == -1).any(axis=1)
    df = df[mask].reset_index(drop=True)
    
    texts = df["comment_text"].values
    true_labels = df[LABELS].astype(int).values
    
    print(f"Evaluating on {len(texts)} fully-labelled test examples (filtered from {initial})")
    
    # Get ensemble predictions
    preds = ensemble.predict(texts)
    
    # Print report
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION REPORT")
    print("="*70)
    print(classification_report(true_labels, preds, target_names=LABELS, zero_division=0))
    
    f1_micro = f1_score(true_labels, preds, average="micro")
    f1_macro = f1_score(true_labels, preds, average="macro")
    
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")


if __name__ == "__main__":
    import os
    
    ensemble = EnsemblePredictor(
        "models/final_model",
        "models/roberta_focal_model",
        "models/optimal_thresholds.json"
    )
    
    evaluate_ensemble("data/test.csv", "data/test_labels.csv", ensemble)
