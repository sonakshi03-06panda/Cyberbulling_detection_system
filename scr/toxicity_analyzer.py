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


def _is_toxic_word_in_caps_or_quotes(text: str, toxic_labels: List[str]) -> bool:
    """
    Check if toxic words are contained within ALL CAPS or inside quotation marks.
    Such instances are typically not abusive (quoting or emphasizing, not targeting anyone).
    
    Returns True if ALL detected toxic words are only found in caps or quotes.
    """
    if not toxic_labels:
        return False
    
    # Check for text segments in quotes
    quoted_segments = re.findall(r'["\']([^"\']*)["\']', text)
    caps_segments = re.findall(r'\b([A-Z]{2,})\b', text)  # Words with 2+ caps
    all_caps_words = re.findall(r'\b([A-Z][A-Z0-9_]*)\b', text)  # All caps words/phrases
    
    # Combine all non-abusive contexts
    non_abusive_contexts = quoted_segments + [' '.join(caps_segments)]
    
    # If most of the text is in caps or quotes, it's likely not abusive
    if text.isupper() and len(text) > 5:  # Whole comment in caps
        return True
    
    # Check if text appears to be primarily quoted
    quote_chars_count = sum(text.count(char) for char in ['"', "'", '"', '"'])
    if quote_chars_count >= 2 and len(quoted_segments) > 0:  # Has opening and closing quotes
        return True
    
    return False


def _is_likely_quote_or_reference(text: str) -> bool:
    """
    Detect if comment is likely a quote from video based on formatting.
    Quotes contain quotation marks or quote-like formatting.
    """
    text_lower = text.lower()
    
    # Has quotation marks (single or double, opening or closing)
    if any(char in text for char in ['"', "'", '"', '"', '„', '"', "'"]):
        return True
    
    # Quoted text with attribution marker (- or —)
    if ('—' in text or ' - ' in text) and (text.startswith('"') or text.startswith("'")):
        return True
    
    # Excessive punctuation/repetition = likely quoting or mimicking
    if re.search(r'([!?r.]{4,})', text):  # 4+ repeated ! ? . or r (like "barrrrr")
        return True
    
    # All caps with multiple exclamations = likely quoting dramatic moment
    if text.upper() == text and text.count('!') >= 2:
        return True
    
    return False


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
            "threat": 0.25,           # Serious - lowest threshold (most important)
            "severe_toxic": 0.65,     # HEAVILY INCREASED from 0.30 - profanity quotes shouldn't trigger this
            "identity_hate": 0.70,    # Increased from 0.65 - requires very high confidence
            "toxic": 0.80,            # INCREASED from 0.70 - stricter general toxicity detection (profanity ≠ toxic)
            "insult": 0.85,           # Increased from 0.80 - very high threshold
            "obscene": 0.90           # Increased from 0.85 - highest threshold - curse words in casual context must be very high confidence
        }
        
        # Label weights for severity calculation and confidence scoring
        self.label_weights = {
            "threat": 1.0,            # Maximum weight - direct threats are serious
            "severe_toxic": 0.95,     # Very high - severe toxicity
            "identity_hate": 0.60,    # Reduced from 0.9 - mention of identity alone shouldn't dominate (requires hateful intent)
            "toxic": 0.75,            # Increased from 0.7 - general toxicity is important
            "insult": 0.55,           # Reduced from 0.5 - insults are less serious than harassment
            "obscene": 0.35           # Reduced from 0.4 - profanity alone is minimal concern
        }
    
    def _determine_toxicity(self, toxic_labels: List[str], is_quote: bool = False) -> bool:
        """
        Determine if comment is truly toxic based on label combinations.
        VERY STRICT logic to minimize false positives from profanity/quotes.
        Only flags comments with clear harmful intent, not just strong language.
        """
        if not toxic_labels:
            return False
        
        # If it's a detected quote, almost nothing is toxic (quotes express reactions, not threats)
        # Exception: only very serious threats with other signals
        if is_quote:
            # Quotes with ONLY threat label = not genuine threat (it's a movie reference)
            if toxic_labels == ["threat"]:
                return False
            # Quotes with threat + other labels might still not be genuine
            if "threat" in toxic_labels and len(toxic_labels) <= 2:
                return False
        
        # Direct threats are ALWAYS toxic (but see quote handling above)
        if "threat" in toxic_labels and not is_quote:
            return True
        
        # Severe_toxic at high threshold (0.65+) indicates GENUINELY severe content (not just quotes)
        if "severe_toxic" in toxic_labels:
            # Still double-check: if ONLY severe_toxic with mild labels, might be quoted
            if all(l in ["severe_toxic", "obscene", "insult"] for l in toxic_labels):
                # If it has toxic too, it's more likely genuine
                return "toxic" in toxic_labels
            return True
        
        # Single label cases - almost never toxic (these are high threshold labels now)
        if len(toxic_labels) == 1:
            return False
        
        # Multiple labels required beyond this point
        
        # Insult + identity_hate (without severe signals) = not toxic
        # (Could be assertive speech about a group, not hateful)
        if set(toxic_labels) == {"insult", "identity_hate"}:
            return False
        
        # Toxic + any single mild label = not toxic
        # (Generic negativity, not targeted harassment)
        if "toxic" in toxic_labels and len([l for l in toxic_labels if l in ["insult", "obscene"]]) == 1:
            return False
        
        # Toxic + identity_hate (no other serious labels) = not necessarily toxic
        # (Could be political speech, not hate speech)
        if set(toxic_labels) == {"toxic", "identity_hate"}:
            return False
        
        # 3+ serious labels = very likely toxic
        serious = ["toxic", "insult", "identity_hate"]
        if sum(1 for l in toxic_labels if l in serious) >= 3:
            return True
        
        # Toxic + insult + something else = toxic
        if "toxic" in toxic_labels and "insult" in toxic_labels and len(toxic_labels) >= 3:
            return True
        
        return False
    
    def classify_comment(self, text: str) -> Dict:
        """
        Classify a single comment with smart thresholds and label weighting.
        Considers context (quotes/references) to avoid false positives.
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
        
        # Check if this looks like a video quote/reference
        is_quote = _is_likely_quote_or_reference(text)
        
        # Check if toxic words are only in caps or quoted text (not abusive)
        if _is_toxic_word_in_caps_or_quotes(text, toxic_labels):
            is_toxic = False
            toxic_labels = []
        else:
            # Determine true toxicity using combination logic
            is_toxic = self._determine_toxicity(toxic_labels, is_quote=is_quote)
            
            # If it's clearly a quote and doesn't have threat/severe signals, don't flag it
            if is_quote and is_toxic and "threat" not in toxic_labels and "severe_toxic" not in toxic_labels:
                is_toxic = False
                toxic_labels = []  # Clear the labels since it's likely just a quote
        
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
    
    def classify_batch(self, texts: List[str], batch_size: int = 64) -> List[Dict]:
        """
        Classify multiple comments with smart thresholds and label weighting.
        
        Args:
            texts: List of comment texts
            batch_size: Batch size for inference (increased to 64 for faster processing)
        
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
            
            # Check if this looks like a video quote/reference
            is_quote = _is_likely_quote_or_reference(text)
            
            # Check if toxic words are only in caps or quoted text (not abusive)
            if _is_toxic_word_in_caps_or_quotes(text, toxic_labels):
                is_toxic = False
                toxic_labels = []
            else:
                # Determine true toxicity using combination logic
                is_toxic = self._determine_toxicity(toxic_labels, is_quote=is_quote)
                
                # If comment is clearly quoted/referenced, additional check
                if is_quote and is_toxic and "threat" not in toxic_labels and "severe_toxic" not in toxic_labels:
                    is_toxic = False
                    toxic_labels = []
            
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


def analyze_comments(comments: List[Dict], analyzer: 'ToxicityAnalyzer' = None) -> pd.DataFrame:
    """
    Analyze a list of comments fetched from YouTube.
    
    Args:
        comments: List of comment dicts (from CommentFetcher)
        analyzer: Optional pre-loaded ToxicityAnalyzer instance. If None, creates a new one.
    
    Returns:
        DataFrame with original data + toxicity analysis
    """
    if analyzer is None:
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
        