"""
INTEGRATION GUIDE: Improved Toxicity Classification
====================================================

This guide explains how to integrate the improved toxicity classification
system into your existing VibeCheck application.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from IMPROVED_CLASSIFIER_IMPLEMENTATION import (
    ImprovedToxicityClassifier,
    VALIDATION_TESTS
)
import numpy as np


# ============================================================================
# STEP 1: UPDATE THRESHOLDS IN TOXICITY_ANALYZER.PY
# ============================================================================

UPDATED_THRESHOLDS_CODE = '''
# In scr/toxicity_analyzer.py, update the __init__ method:

def __init__(self, model_path: str = "models/final_model", threshold: float = 0.5):
    """Initialize Toxicity Analyzer with improved thresholds."""
    print(f"Loading model from {model_path}...")
    self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
    self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    self.model.to(DEVICE)
    self.model.eval()
    self.threshold = threshold
    self.labels = LABELS
    self.preprocessor = TextPreprocessor()
    
    # IMPROVED THRESHOLDS - Better for distinguishing criticism vs toxicity
    self.label_thresholds = {
        "threat": 0.25,           # Lower - threats are clear
        "severe_toxic": 0.55,     # Reduced - better recall
        "identity_hate": 0.65,    # Refined - better detection
        "toxic": 0.70,            # Lowered - more balanced
        "insult": 0.75,           # Lowered - better accuracy
        "obscene": 0.85           # Balanced
    }
    
    # IMPROVED WEIGHTS - Severity calibration
    self.label_weights = {
        "threat": 1.0,            # Critical
        "severe_toxic": 0.95,     # Almost critical
        "identity_hate": 0.80,    # Serious
        "toxic": 0.60,            # Moderate
        "insult": 0.70,           # Serious
        "obscene": 0.25           # Low
    }
    
    # Initialize improved classifier
    from IMPROVED_CLASSIFIER_IMPLEMENTATION import ImprovedToxicityClassifier
    self.improved_classifier = ImprovedToxicityClassifier()
'''

# ============================================================================
# STEP 2: UPDATE CLASSIFY_COMMENT METHOD
# ============================================================================

UPDATED_CLASSIFY_CODE = '''
# In scr/toxicity_analyzer.py, update classify_comment() method:

def classify_comment(self, text: str) -> Dict:
    """
    Classify a single comment with IMPROVED logic.
    
    Key improvements:
    - Distinguishes between criticism and personal attacks
    - Better threshold tuning
    - Improved category classification
    """
    # Use advanced preprocessing
    preprocessing_result = self.preprocessor.preprocess(text, return_features=True)
    clean = preprocessing_result['cleaned_text']
    
    # Get model probabilities
    inputs = self.tokenizer(
        clean, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Get labels that pass improved thresholds
    toxic_labels = [
        self.labels[i] for i in range(len(self.labels))
        if probs[i] > self.label_thresholds[self.labels[i]]
    ]
    
    # Create probability dict
    label_probs = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
    
    # USE IMPROVED CLASSIFIER
    improved_result = self.improved_classifier.classify(
        text,
        label_probs,
        toxic_labels,
        self.labels
    )
    
    return {
        "text": text,
        "clean_text": clean,
        
        # NEW: Category classification (non_toxic, mild_negative, toxic_insult, severe_abuse, threat_language)
        "category": improved_result["category"],
        
        # Core toxicity info
        "is_toxic": improved_result["is_toxic"],
        "toxicity_score": improved_result["toxicity_score"],  # 0.0-1.0
        "severity": improved_result["severity"],             # 0-3
        
        # Model labels and confidences
        "toxic_labels": improved_result["labels"],
        "confidences": label_probs,
        "max_confidence": max(improved_result["toxicity_score"] * c for c in label_probs.values()) if label_probs else 0.0,
        
        # Reasoning
        "reasoning": improved_result["reasoning"],
        
        # Auxiliary features (from preprocessing)
        "is_valid_length": preprocessing_result['is_valid'],
        "has_profanity": preprocessing_result['has_profanity'],
        "profanity_list": preprocessing_result['profanity_list'],
        "has_sarcasm_indicators": preprocessing_result['has_sarcasm_indicators'],
    }
'''

# ============================================================================
# STEP 3: UPDATE CLASSIFY_BATCH METHOD
# ============================================================================

UPDATED_BATCH_CODE = '''
# In scr/toxicity_analyzer.py, update classify_batch() method:

def classify_batch(self, texts: List[str], batch_size: int = 128) -> List[Dict]:
    """
    Classify multiple comments with IMPROVED logic.
    """
    # Preprocess all texts
    preprocessing_results = self.preprocessor.batch_preprocess(texts, return_features=True)
    clean_texts = [r['cleaned_text'] for r in preprocessing_results]
    
    # Tokenize
    encodings = self.tokenizer(
        clean_texts, truncation=True, padding=True, max_length=128
    )
    
    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(clean_texts), batch_size), desc="Model Inference"):
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
        
        # Get labels passing improved thresholds
        toxic_labels = [
            self.labels[j] for j in range(len(self.labels))
            if probs[j] > self.label_thresholds[self.labels[j]]
        ]
        
        # Create probability dict
        label_probs = {self.labels[j]: float(probs[j]) for j in range(len(self.labels))}
        
        # USE IMPROVED CLASSIFIER
        improved_result = self.improved_classifier.classify(
            text,
            label_probs,
            toxic_labels,
            self.labels
        )
        
        results.append({
            "text": text,
            "clean_text": preprocessing_result['cleaned_text'],
            "category": improved_result["category"],
            "is_toxic": improved_result["is_toxic"],
            "toxicity_score": improved_result["toxicity_score"],
            "severity": improved_result["severity"],
            "toxic_labels": improved_result["labels"],
            "confidences": label_probs,
            "reasoning": improved_result["reasoning"],
            "is_valid_length": preprocessing_result['is_valid'],
            "has_profanity": preprocessing_result['has_profanity'],
            "profanity_list": preprocessing_result['profanity_list'],
            "has_sarcasm_indicators": preprocessing_result['has_sarcasm_indicators'],
        })
    
    return results
'''

# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def run_validation_tests():
    """
    Run validation tests against provided examples.
    
    This tests the classification logic without the full model,
    using mock probabilities.
    """
    print("=" * 80)
    print("VALIDATION TEST SUITE")
    print("=" * 80)
    
    classifier = ImprovedToxicityClassifier()
    
    # Mock labels for testing
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    test_cases = [
        # NON-TOXIC
        {
            "text": "This video was really helpful, thanks for explaining everything clearly.",
            "probs": {"toxic": 0.1, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.0, "identity_hate": 0.0},
            "expected": {"is_toxic": False, "category": "non_toxic"}
        },
        {
            "text": "I disagree with your opinion, but I appreciate the effort you put into the video.",
            "probs": {"toxic": 0.3, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.1, "identity_hate": 0.0},
            "expected": {"is_toxic": False, "category": "non_toxic"}
        },
        
        # MILD NEGATIVE (Criticism)
        {
            "text": "This explanation is confusing and could have been done better.",
            "probs": {"toxic": 0.65, "severe_toxic": 0.1, "obscene": 0.0, "threat": 0.0, "insult": 0.3, "identity_hate": 0.0},
            "expected": {"is_toxic": False, "category": "mild_negative"}
        },
        {
            "text": "I don't think this video is very useful.",
            "probs": {"toxic": 0.55, "severe_toxic": 0.0, "obscene": 0.0, "threat": 0.0, "insult": 0.1, "identity_hate": 0.0},
            "expected": {"is_toxic": False, "category": "mild_negative"}
        },
        
        # TOXIC INSULT
        {
            "text": "You are such an idiot.",
            "probs": {"toxic": 0.8, "severe_toxic": 0.5, "obscene": 0.2, "threat": 0.1, "insult": 0.9, "identity_hate": 0.0},
            "expected": {"is_toxic": True, "category": "toxic_insult"}
        },
        {
            "text": "Only a complete fool would believe something like this.",
            "probs": {"toxic": 0.75, "severe_toxic": 0.4, "obscene": 0.1, "threat": 0.0, "insult": 0.85, "identity_hate": 0.0},
            "expected": {"is_toxic": True, "category": "toxic_insult"}
        },
        
        # SEVERE ABUSE
        {
            "text": "You're a pathetic loser and nobody cares about your opinion.",
            "probs": {"toxic": 0.9, "severe_toxic": 0.85, "obscene": 0.6, "threat": 0.3, "insult": 0.95, "identity_hate": 0.1},
            "expected": {"is_toxic": True, "category": "severe_abuse"}
        },
        
        # THREAT
        {
            "text": "Someone should seriously shut you up for saying things like this.",
            "probs": {"toxic": 0.85, "severe_toxic": 0.7, "obscene": 0.4, "threat": 0.8, "insult": 0.8, "identity_hate": 0.1},
            "expected": {"is_toxic": True, "category": "threat_language"}
        },
        {
            "text": "You should die in a hole.",
            "probs": {"toxic": 0.95, "severe_toxic": 0.9, "obscene": 0.7, "threat": 0.95, "insult": 0.9, "identity_hate": 0.2},
            "expected": {"is_toxic": True, "category": "threat_language"}
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        text = test["text"]
        probs = test["probs"]
        expected = test["expected"]
        
        # Get triggered labels
        triggered = [label for label in labels if probs[label] > classifier.label_thresholds[label]]
        
        # Classify
        result = classifier.classify(text, probs, triggered, labels)
        
        # Check results
        is_toxic_match = result["is_toxic"] == expected["is_toxic"]
        category_match = result["category"] == expected["category"]
        
        status = "✓ PASS" if (is_toxic_match and category_match) else "✗ FAIL"
        
        print(f"\n{status}")
        print(f"Text: {text[:60]}...")
        print(f"  Expected: is_toxic={expected['is_toxic']}, category={expected['category']}")
        print(f"  Got:      is_toxic={result['is_toxic']}, category={result['category']}")
        print(f"  Score: {result['toxicity_score']:.3f}, Severity: {result['severity']}")
        
        if is_toxic_match and category_match:
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"Success Rate: {100 * passed / (passed + failed):.1f}%")
    print(f"{'=' * 80}")
    
    return passed, failed


# ============================================================================
# INTEGRATION CHECKLIST
# ============================================================================

INTEGRATION_CHECKLIST = """
INTEGRATION CHECKLIST
====================

Before deploying the improved classifier:

[ ] 1. Add ImprovedToxicityClassifier to project
    - Copy IMPROVED_CLASSIFIER_IMPLEMENTATION.py to scr/
    
[ ] 2. Update toxicity_analyzer.py
    - Import ImprovedToxicityClassifier
    - Update label thresholds in __init__()
    - Update classify_comment() method
    - Update classify_batch() method
    - Add improved_classifier instance variable

[ ] 3. Run validation tests
    - Execute run_validation_tests()
    - Verify all provided examples pass
    - Target: 100% pass rate on validation set

[ ] 4. Test with existing app
    - python scr/app.py
    - Submit test comments through web interface
    - Verify classification accuracy

[ ] 5. Update API responses
    - Add "category" field to prediction responses
    - Add "reasoning" field for user clarity
    - Update severity calculation
    
[ ] 6. Update dashboard display
    - Show category labels (not just is_toxic)
    - Display reasoning for transparency
    - Update color coding if needed

[ ] 7. Update documentation
    - Document new category system
    - Explain classification logic
    - Provide example responses

[ ] 8. Performance validation
    - Benchmark speed impact: target <1% increase
    - Verify accuracy on held-out test set
    - Compare against baseline

[ ] 9. User testing
    - Get feedback from moderators
    - Adjust thresholds if needed
    - Monitor false positive/negative rates

[ ] 10. Production deployment
    - Create backup of current model
    - Deploy to staging first
    - Monitor error rates for 48 hours
    - If all good, deploy to production
"""

# ============================================================================
# EXPECTED OUTPUT CHANGES
# ============================================================================

EXPECTED_OUTPUT_CHANGES = """
OLD OUTPUT STRUCTURE:
{
    "text": "You are such an idiot.",
    "is_toxic": True,
    "toxic_labels": ["insult", "toxic"],
    "confidences": {"toxic": 0.8, ...},
    "severity": 2,
    "max_confidence": 0.65
}

NEW OUTPUT STRUCTURE (with improvements):
{
    "text": "You are such an idiot.",
    
    # NEW: Category classification
    "category": "toxic_insult",                    # 5-class categorization
    
    # Updated fields
    "is_toxic": True,
    "toxicity_score": 0.72,                       # 0.0-1.0 score using improved logic
    "severity": 2,                                # 0-3
    "toxic_labels": ["insult", "toxic"],          # Raw model labels
    "confidences": {...},                         # Raw model confidences
    
    # NEW: Explanation
    "reasoning": "Personal attacks or insults detected.",
    
    # Preserved fields
    "clean_text": "you are such an idiot .",
    "is_valid_length": True,
    "has_profanity": True,
    "profanity_list": ["idiot"],
    "has_sarcasm_indicators": False
}
"""


if __name__ == "__main__":
    print("Improved Toxicity Classification System")
    print("=" * 80)
    print()
    
    # Run validation tests
    passed, failed = run_validation_tests()
    
    # Print guidance
    print("\n" + "=" * 80)
    print("INTEGRATION GUIDE")
    print("=" * 80)
    print(INTEGRATION_CHECKLIST)
    print("\n" + "=" * 80)
    print("EXPECTED OUTPUT CHANGES")
    print("=" * 80)
    print(EXPECTED_OUTPUT_CHANGES)
