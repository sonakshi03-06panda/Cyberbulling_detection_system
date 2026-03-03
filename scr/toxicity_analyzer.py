"""
Classify comments for toxicity using a local DistilBERT model.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import re
import emoji
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    try:
        text = emoji.demojize(text)  # Convert emojis to text
    except:
        pass
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


class ToxicityAnalyzer:
    """Analyze comments for toxicity using a DistilBERT model."""
    
    def __init__(self, model_path: str = "models/final_model", threshold: float = 0.5):
        """
        Args:
            model_path: path where the HF model is saved
            threshold: sigmoid threshold for binary labels
        """
        print(f"Loading model from {model_path}...")
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model.to(DEVICE)
        self.model.eval()
        self.threshold = threshold
        self.labels = LABELS
    

    def classify_comment(self, text: str) -> Dict:
        """
        Classify a single comment.
        """
        # local HF inference
        clean = clean_text(text)
        inputs = self.tokenizer(
            clean, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
        is_toxic = (probs > self.threshold).astype(int)
        toxic_labels = [self.labels[i] for i in range(len(self.labels)) if is_toxic[i]]
        if not toxic_labels:
            severity = 0
        elif any(l in ["threat", "severe_toxic"] for l in toxic_labels):
            severity = 3
        elif any(l in ["insult", "identity_hate"] for l in toxic_labels):
            severity = 2
        else:
            severity = 1
        return {
            "text": text,
            "clean_text": clean,
            "is_toxic": len(toxic_labels) > 0,
            "toxic_labels": toxic_labels,
            "confidences": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "severity": severity,
            "max_confidence": float(probs.max())
        }
    
    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Classify multiple comments.
        
        Args:
            texts: List of comment texts
            batch_size: Batch size for inference
        
        Returns:
            List of classification dicts
        """
        results = []
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=128
        )
        all_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Classifying", unit="batch"):
                batch = {k: torch.tensor(v[i:i+batch_size]).to(DEVICE) for k, v in encodings.items()}
                outputs = self.model(**batch)
                logits = outputs.logits.cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                all_probs.append(probs)
        all_probs = np.vstack(all_probs)
        for i, text in enumerate(tqdm(texts, desc="Processing results")):
            probs = all_probs[i]
            is_toxic = (probs > self.threshold).astype(int)
            toxic_labels = [self.labels[j] for j in range(len(self.labels)) if is_toxic[j]]
            if not toxic_labels:
                severity = 0
            elif any(l in ["threat", "severe_toxic"] for l in toxic_labels):
                severity = 3
            elif any(l in ["insult", "identity_hate"] for l in toxic_labels):
                severity = 2
            else:
                severity = 1
            results.append({
                "text": text,
                "clean_text": clean_text(text),
                "is_toxic": len(toxic_labels) > 0,
                "toxic_labels": toxic_labels,
                "confidences": {self.labels[j]: float(probs[j]) for j in range(len(self.labels))},
                "severity": severity,
                "max_confidence": float(probs.max())
            })
        return results


def analyze_comments(comments: List[Dict]) -> pd.DataFrame:
    """
    Analyze a list of comments fetched from YouTube.
    
    Args:
        comments: List of comment dicts (from CommentFetcher)
    
    Returns:
        DataFrame with original data + toxicity analysis
    """
    analyzer = ToxicityAnalyzer()
    
    texts = [c["text"] for c in comments]
    classifications = analyzer.classify_batch(texts)
    
    # Merge results
    for i, comment in enumerate(comments):
        comment.update(classifications[i])
    
    df = pd.DataFrame(comments)
    return df


if __name__ == "__main__":
    # Example usage
    analyzer = ToxicityAnalyzer()
    
    test_comments = [
        "This video is amazing!",
        "You're a terrible person and I hate you",
        "I strongly disagree with your opinion",
        "You should die",
    ]
    
    results = analyzer.classify_batch(test_comments)
    for r in results:
        print(f"Text: {r['text'][:50]}...")
        print(f"  Toxic: {r['is_toxic']}, Labels: {r['toxic_labels']}, Severity: {r['severity']}\n")


