"""
Classify comments for toxicity using a local DistilBERT model.
Integrates advanced text preprocessing pipeline for improved accuracy.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
import emoji
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm
from preprocessing import TextPreprocessor
from IMPROVED_CLASSIFIER_IMPLEMENTATION import ImprovedToxicityClassifier
try:
    from emoji_sarcasm_detector import detect_emoji_sarcasm
except ImportError:
    # Fallback if module not found
    def detect_emoji_sarcasm(text, base_toxicity):
        return {"is_sarcastic": False, "sarcasm_score": 0.0, "adjusted_toxicity_score": base_toxicity}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Initialize global preprocessor instance
_preprocessor = TextPreprocessor()


def clean_text(text: str, use_advanced_preprocessing: bool = True) -> str:
    """
    Clean and normalize text using advanced preprocessing pipeline.
    
    Args:
        text: Input text to cleaning 
        use_advanced_preprocessing: If True, uses advanced preprocessing; if False, uses basic cleaning
    
    Returns:
        Cleaned text
    """
    if use_advanced_preprocessing:
        # Use advanced preprocessing pipeline
        result = _preprocessor.preprocess(text, return_features=False)
        return result['cleaned_text']
    else:
        # Fallback to basic preprocessing
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
    """Analyze comments for toxicity using a DistilBERT model with advanced preprocessing."""
    
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
        self.preprocessor = TextPreprocessor()  # Initialize preprocessor
        self.improved_classifier = ImprovedToxicityClassifier()  # Initialize improved classifier
        
        # IMPROVED THRESHOLDS: Better for distinguishing criticism from toxicity
        self.label_thresholds = {
            "threat": 0.25,           # Lower - threats are unambiguous
            "severe_toxic": 0.55,     # Reduced - better recall on severe abuse
            "identity_hate": 0.65,    # Refined - catches hate speech better
            "toxic": 0.70,            # Lowered - allows more true positives
            "insult": 0.75,           # Lowered - better insult detection
            "obscene": 0.85           # Unchanged - profanity alone isn't always toxic
        }
        
        # IMPROVED WEIGHTS: Better severity calibration
        self.label_weights = {
            "threat": 1.0,            # Critical
            "severe_toxic": 0.95,    # Almost as critical
            "identity_hate": 0.80,    # Serious
            "toxic": 0.60,            # Moderate
            "insult": 0.70,           # Serious-Moderate
            "obscene": 0.25           # Low priority
        }
    
    def classify_comment(self, text: str) -> Dict:
        """
        Classify a single comment with IMPROVED logic.
        Uses ImprovedToxicityClassifier for better distinction between
        criticism and personal attacks. Includes 5-tier categorization.
        """
        # Use advanced preprocessing pipeline with auxiliary features
        preprocessing_result = self.preprocessor.preprocess(text, return_features=True)
        clean = preprocessing_result['cleaned_text']
        
        # Local HF inference
        inputs = self.tokenizer(
            clean, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
        
        # Create probability dict
        label_probs = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        
        # Apply improved thresholds
        toxic_labels = [
            self.labels[i] for i in range(len(self.labels))
            if probs[i] > self.label_thresholds[self.labels[i]]
        ]
        
        # USE IMPROVED CLASSIFIER for better categorization
        improved_result = self.improved_classifier.classify(
            text,
            label_probs,
            toxic_labels,
            self.labels
        )
        
        # EMOJI-AWARE SARCASM ANALYSIS
        # Detect if emojis amplify toxicity through sarcasm
        sarcasm_analysis = detect_emoji_sarcasm(
            text,
            improved_result.get("toxicity_score", 0.0)
        )
        
        # If strong sarcasm signal detected
        sarcasm_detected = False
        if sarcasm_analysis.get("is_sarcastic", False) and sarcasm_analysis.get("sarcasm_score", 0) > 0.5:
            sarcasm_detected = True
            adjusted_score = sarcasm_analysis.get("adjusted_toxicity_score", improved_result.get("toxicity_score", 0.0))
            
            # Check if we should upgrade category to sarcastic_toxic
            if improved_result["category"] in ["non_toxic", "mild_negative"]:
                if adjusted_score > 0.45:
                    improved_result["category"] = "sarcastic_toxic"
                    improved_result["is_toxic"] = True
                    improved_result["toxicity_score"] = adjusted_score
            elif improved_result["category"] in ["toxic_insult"]:
                # Boost existing toxicity with emoji context
                improved_result["toxicity_score"] = adjusted_score
        
        # Build result dict
        result = {
            "text": text,
            "clean_text": clean,
            # NEW: 5-tier categorization (with sarcastic_toxic)
            "category": improved_result["category"],
            # Core toxicity
            "is_toxic": improved_result["is_toxic"],
            "toxicity_score": improved_result["toxicity_score"],
            "severity": improved_result["severity"],
            # Model outputs
            "toxic_labels": improved_result["labels"],
            "confidences": label_probs,
            "max_confidence": max(label_probs.values()) if label_probs else 0.0,
            # Explanation
            "reasoning": improved_result["reasoning"],
            # Auxiliary features
            "is_valid_length": preprocessing_result['is_valid'],
            "has_profanity": preprocessing_result['has_profanity'],
            "profanity_list": preprocessing_result['profanity_list'],
            "profanity_intensity": preprocessing_result['profanity_intensity'],
            "has_sarcasm_indicators": preprocessing_result['has_sarcasm_indicators'],
            "sarcasm_indicators": preprocessing_result['sarcasm_indicators'],
        }
        
        # Add emoji sarcasm info if detected
        if sarcasm_detected:
            result["sarcasm_info"] = sarcasm_analysis
        
        return result
    
    def classify_batch(self, texts: List[str], batch_size: int = 128) -> List[Dict]:
        """
        Classify multiple comments with IMPROVED logic.
        Uses ImprovedToxicityClassifier for better distinction between
        criticism and personal attacks. Includes 5-tier categorization.
        
        Args:
            texts: List of comment texts
            batch_size: Batch size for inference (default 128 for optimal throughput)
        
        Returns:
            List of classification dicts with 5-tier category and explanation
        """
        # Preprocess all texts first
        preprocessing_results = self.preprocessor.batch_preprocess(texts, return_features=True)
        clean_texts = [r['cleaned_text'] for r in preprocessing_results]
        
        # Tokenize cleaned texts
        encodings = self.tokenizer(
            clean_texts, truncation=True, padding=True, max_length=128
        )
        all_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(clean_texts), batch_size), desc="Model Inference", unit="batch"):
                batch = {k: torch.tensor(v[i:i+batch_size]).to(DEVICE) for k, v in encodings.items()}
                outputs = self.model(**batch)
                logits = outputs.logits.cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                all_probs.append(probs)
        all_probs = np.vstack(all_probs)
        
        results = []
        for i, text in enumerate(tqdm(texts, desc="Classification")):
            probs = all_probs[i]
            preprocessing_result = preprocessing_results[i]
            
            # Create probability dict
            label_probs = {self.labels[j]: float(probs[j]) for j in range(len(self.labels))}
            
            # Apply improved thresholds
            toxic_labels = [
                self.labels[j] for j in range(len(self.labels))
                if probs[j] > self.label_thresholds[self.labels[j]]
            ]
            
            # USE IMPROVED CLASSIFIER for better categorization
            improved_result = self.improved_classifier.classify(
                text,
                label_probs,
                toxic_labels,
                self.labels
            )
            
            # EMOJI-AWARE SARCASM ANALYSIS
            sarcasm_analysis = detect_emoji_sarcasm(
                text,
                improved_result.get("toxicity_score", 0.0)
            )
            
            # Update category if sarcasm is strong
            sarcasm_detected = False
            if sarcasm_analysis.get("is_sarcastic", False) and sarcasm_analysis.get("sarcasm_score", 0) > 0.5:
                sarcasm_detected = True
                adjusted_score = sarcasm_analysis.get("adjusted_toxicity_score", improved_result.get("toxicity_score", 0.0))
                
                if improved_result["category"] in ["non_toxic", "mild_negative"]:
                    if adjusted_score > 0.45:
                        improved_result["category"] = "sarcastic_toxic"
                        improved_result["is_toxic"] = True
                        improved_result["toxicity_score"] = adjusted_score
                elif improved_result["category"] in ["toxic_insult"]:
                    improved_result["toxicity_score"] = adjusted_score
            
            result_item = {
                "text": text,
                "clean_text": preprocessing_result['cleaned_text'],
                # NEW: 5-tier categorization (including sarcastic_toxic)
                "category": improved_result["category"],
                # Core toxicity
                "is_toxic": improved_result["is_toxic"],
                "toxicity_score": improved_result["toxicity_score"],
                "severity": improved_result["severity"],
                # Model outputs
                "toxic_labels": improved_result["labels"],
                "confidences": label_probs,
                "max_confidence": max(label_probs.values()) if label_probs else 0.0,
                # Explanation
                "reasoning": improved_result["reasoning"],
                # Auxiliary features
                "is_valid_length": preprocessing_result['is_valid'],
                "has_profanity": preprocessing_result['has_profanity'],
                "profanity_list": preprocessing_result['profanity_list'],
                "profanity_intensity": preprocessing_result['profanity_intensity'],
                "has_sarcasm_indicators": preprocessing_result['has_sarcasm_indicators'],
                "sarcasm_indicators": preprocessing_result['sarcasm_indicators'],
            }
            
            # Add emoji sarcasm info if detected
            if sarcasm_detected:
                result_item["sarcasm_info"] = sarcasm_analysis
            
            results.append(result_item)
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
        