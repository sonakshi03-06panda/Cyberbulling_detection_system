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
        
        # Label-specific thresholds: serious toxicity is easier to trigger, mild offense requires higher confidence
        self.label_thresholds = {
            "threat": 0.30,           # Serious - lowest threshold
            "severe_toxic": 0.35,     # Serious - low threshold
            "identity_hate": 0.35,    # Serious - low threshold
            "toxic": 0.60,            # Higher threshold - base toxic label is unreliable on its own
            "insult": 0.70,           # Very high threshold - easily confused with assertive speech
            "obscene": 0.80           # Highest threshold - curse words in casual context must be very high confidence
        }
        
        # Label weights for severity calculation and confidence scoring
        self.label_weights = {
            "threat": 1.0,
            "severe_toxic": 1.0,
            "identity_hate": 0.9,
            "toxic": 0.7,
            "insult": 0.5,
            "obscene": 0.4
        }
    
    def _determine_toxicity(self, toxic_labels: List[str]) -> bool:
        """
        Determine if comment is truly toxic based on label combinations.
        Reduces false positives from curse words alone.
        """
        if not toxic_labels:
            return False
        
        # Serious labels trigger toxicity immediately
        serious = ["threat", "severe_toxic", "identity_hate"]
        if any(l in serious for l in toxic_labels):
            return True
        
        # Mild labels alone are not toxic
        mild = ["obscene", "insult"]
        if all(l in mild for l in toxic_labels):
            return False
        
        # "toxic" label without serious context is often unreliable
        # Only consider toxic if it comes with other signals or serious intent
        if toxic_labels == ["toxic"]:
            # Single "toxic" label requires higher confidence in our model
            # We'll use the max confidence threshold
            return False
        
        # "toxic" + mild offense = usually safe
        if "toxic" in toxic_labels and all(l in mild + ["toxic"] for l in toxic_labels):
            return False
        
        # Default: has multi-label toxicity signals
        return True
    
    def classify_comment(self, text: str) -> Dict:
        """
        Classify a single comment with smart thresholds and label weighting.
        """
        # Local HF inference
        clean = clean_text(text)
        inputs = self.tokenizer(
            clean, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
        
        # Apply label-specific thresholds
        toxic_labels = [
            self.labels[i] for i in range(len(self.labels))
            if probs[i] > self.label_thresholds[self.labels[i]]
        ]
        
        # Determine true toxicity using combination logic
        is_toxic = self._determine_toxicity(toxic_labels)
        
        # Calculate weighted confidence for true toxic labels only
        if is_toxic:
            weighted_probs = {self.labels[i]: float(probs[i]) * self.label_weights[self.labels[i]] 
                            for i in range(len(self.labels)) if self.labels[i] in toxic_labels}
            max_confidence = max(weighted_probs.values()) if weighted_probs else 0.0
        else:
            max_confidence = float(probs.max()) * 0.5  # Discount if not truly toxic
        
        # Calculate severity
        if not is_toxic:
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
            "is_toxic": is_toxic,
            "toxic_labels": toxic_labels,
            "confidences": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "severity": severity,
            "max_confidence": max_confidence
        }
    
    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Classify multiple comments with smart thresholds and label weighting.
        
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
            
            # Apply label-specific thresholds
            toxic_labels = [
                self.labels[j] for j in range(len(self.labels))
                if probs[j] > self.label_thresholds[self.labels[j]]
            ]
            
            # Determine true toxicity using combination logic
            is_toxic = self._determine_toxicity(toxic_labels)
            
            # Calculate weighted confidence for true toxic labels
            if is_toxic:
                weighted_probs = {self.labels[j]: float(probs[j]) * self.label_weights[self.labels[j]] 
                                for j in range(len(self.labels)) if self.labels[j] in toxic_labels}
                max_confidence = max(weighted_probs.values()) if weighted_probs else 0.0
            else:
                max_confidence = float(probs.max()) * 0.5  # Discount if not truly toxic
            
            # Calculate severity
            if not is_toxic:
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
                "is_toxic": is_toxic,
                "toxic_labels": toxic_labels,
                "confidences": {self.labels[j]: float(probs[j]) for j in range(len(self.labels))},
                "severity": severity,
                "max_confidence": max_confidence
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
        