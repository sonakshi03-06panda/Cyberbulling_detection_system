"""
Emoji-Aware Sarcasm Detection Module
=====================================

Detects sarcastic toxicity by analyzing emoji usage patterns combined with text sentiment.

Key Features:
- Identifies mocking/sarcastic emoji patterns
- Detects contrast between positive text and negative emoji sentiment
- Extracts emoji sentiment scores
- Implements sarcasm context detection (sarcastic praise with mocking emojis)
"""

import re
import emoji
from typing import Dict, List, Tuple, Optional


# Emoji sentiment and context mapping
EMOJI_SENTIMENT_MAP = {
    # MOCKING/SARCASTIC EMOJIS (HIGH TOXICITY INDICATORS)
    "🤡": {"sentiment": -0.95, "context": "mockery", "emotion": "clown", "toxicity_boost": 0.9},
    "🙃": {"sentiment": -0.85, "context": "sarcasm", "emotion": "sarcasm", "toxicity_boost": 0.8},
    "🤦": {"sentiment": -0.90, "context": "ridicule", "emotion": "disappointment", "toxicity_boost": 0.85},
    "🤦‍♂️": {"sentiment": -0.90, "context": "ridicule", "emotion": "disappointment", "toxicity_boost": 0.85},
    "🤦‍♀️": {"sentiment": -0.90, "context": "ridicule", "emotion": "disappointment", "toxicity_boost": 0.85},
    "🤡": {"sentiment": -0.95, "context": "mockery", "emotion": "clown", "toxicity_boost": 0.9},
    
    # SARCASTIC LAUGHTER (CAN INDICATE MOCKING)
    "😂": {"sentiment": 0.2, "context": "sarcasm", "emotion": "laugh_tears", "toxicity_boost": 0.7},
    "🤣": {"sentiment": 0.1, "context": "sarcasm", "emotion": "loud_laugh", "toxicity_boost": 0.65},
    "😈": {"sentiment": -0.8, "context": "mockery", "emotion": "evil", "toxicity_boost": 0.75},
    
    # SARCASTIC APPLAUSE
    "👏": {"sentiment": -0.8, "context": "sarcasm", "emotion": "clapping", "toxicity_boost": 0.8},
    "⚰️": {"sentiment": -0.9, "context": "death", "emotion": "coffin", "toxicity_boost": 0.85},
    
    # EYE-ROLL AND DISMISSAL
    "🙄": {"sentiment": -0.75, "context": "dismissal", "emotion": "eye_roll", "toxicity_boost": 0.7},
    "😒": {"sentiment": -0.7, "context": "dismissal", "emotion": "unamused", "toxicity_boost": 0.65},
    "😑": {"sentiment": -0.6, "context": "dismissal", "emotion": "unamused", "toxicity_boost": 0.6},
    "🤨": {"sentiment": -0.5, "context": "skeptical", "emotion": "skeptic", "toxicity_boost": 0.5},
    "😏": {"sentiment": -0.65, "context": "dismissal", "emotion": "smirk", "toxicity_boost": 0.65},
    
    # NEGATIVE/DEATH RELATED
    "💀": {"sentiment": -0.85, "context": "death", "emotion": "skull", "toxicity_boost": 0.8},
    "☠️": {"sentiment": -0.9, "context": "danger", "emotion": "skull_crossbones", "toxicity_boost": 0.85},
    "🤮": {"sentiment": -0.8, "context": "disgust", "emotion": "vomit", "toxicity_boost": 0.75},
    "🔥": {"sentiment": -0.6, "context": "harsh_criticism", "emotion": "flame", "toxicity_boost": 0.6},
    
    # NEUTRAL/SLIGHTLY POSITIVE (BUT CONTEXT DEPENDENT)
    "😐": {"sentiment": -0.1, "context": "neutral", "emotion": "neutral_face", "toxicity_boost": 0.2},
    "😶": {"sentiment": 0.0, "context": "silence", "emotion": "no_mouth", "toxicity_boost": 0.3},
    
    # POSITIVE EMOJIS (BUT CAN BE SARCASTIC WITH CERTAIN TEXT)
    "😄": {"sentiment": 0.7, "context": "positive", "emotion": "happy", "toxicity_boost": 0.0},
    "😁": {"sentiment": 0.8, "context": "positive", "emotion": "grinning", "toxicity_boost": 0.0},
    "😃": {"sentiment": 0.7, "context": "positive", "emotion": "happy", "toxicity_boost": 0.0},
    "👍": {"sentiment": 0.8, "context": "positive", "emotion": "thumbs_up", "toxicity_boost": 0.0},
    "❤️": {"sentiment": 0.9, "context": "positive", "emotion": "love", "toxicity_boost": 0.0},
}

# Extended emojis mapping for common sarcastic patterns
SARCASTIC_EMOJI_PATTERNS = {
    "🤡": "clown_mockery",
    "🙃": "fake_smile",
    "🤦": "face_palm",
    "👏": "sarcastic_clap",
    "😂": "mocking_laughter",
    "🙄": "eye_roll",
    "😒": "unamused",
}

# Positive/approval words that become sarcastic with mocking emoji
SARCASM_WORD_PATTERNS = [
    ("brilliant", "genius", "smart", "clever"), 
    ("wow", "amazing", "great", "wonderful", "excellent"),
    ("fantastic", "awesome", "incredible", "perfect"),
    ("beautiful", "lovely", "nice", "good"),
    ("yes", "correct", "right", "exactly"),
    ("best", "greatest", "top"),
    ("masterpiece", "work of art", "chef's kiss"),
]

# Implicit sarcasm phrases (often paired with mocking emoji indirectly)
SARCASM_PHRASES = [
    "yeah right",
    "oh sure",
    "that's brilliant",
    "wow so smart",
    "great job",
    "nice one",
    "yeah because",
    "sure",
    "whatever",
]


class EmojiSarcasmDetector:
    """
    Detects sarcastic toxicity using emoji analysis combined with text patterns.
    """
    
    def __init__(self):
        """Initialize the detector."""
        self.emoji_sentiment = EMOJI_SENTIMENT_MAP
        self.sarcastic_patterns = SARCASTIC_EMOJI_PATTERNS
        self.positive_words = set()
        for word_group in SARCASM_WORD_PATTERNS:
            self.positive_words.update(word_group)
        self.sarcasm_phrases = SARCASM_PHRASES
    
    def extract_emojis(self, text: str) -> List[Dict]:
        """
        Extract all emojis from text with their sentiment information.
        
        Args:
            text: Input text containing emojis
        
        Returns:
            List of dicts with emoji, position, and sentiment data
        """
        emojis_found = []
        
        # Use emoji library to find all emojis
        data = emoji.emoji_list(text)
        
        for item in data:
            emoji_char = item['emoji']
            position = item['match_start']
            
            # Get sentiment info
            sentiment_info = self.emoji_sentiment.get(emoji_char, {
                "sentiment": 0.0,
                "context": "unknown",
                "emotion": "unknown",
                "toxicity_boost": 0.0
            })
            
            emojis_found.append({
                "emoji": emoji_char,
                "position": position,
                "sentiment": sentiment_info["sentiment"],
                "context": sentiment_info["context"],
                "emotion": sentiment_info["emotion"],
                "toxicity_boost": sentiment_info["toxicity_boost"]
            })
        
        return emojis_found
    
    def detect_sarcasm_pattern(self, text: str, emojis: List[Dict]) -> Dict:
        """
        Detect if text exhibits sarcasm pattern: positive words with mocking emoji.
        
        Args:
            text: Input text
            emojis: List of emoji dicts from extract_emojis
        
        Returns:
            Dict with sarcasm detection results
        """
        text_lower = text.lower()
        
        result = {
            "is_sarcastic": False,
            "sarcasm_score": 0.0,
            "sarcasm_type": None,
            "positive_text_with_negative_emoji": False,
            "mocking_emoji_count": 0,
            "emoji_toxicity_total": 0.0,
            "detected_mocking_emojis": [],
        }
        
        # Count mocking emojis
        mocking_emojis = [e for e in emojis if e["context"] in ["mockery", "sarcasm", "ridicule", "death", "disgust"]]
        result["mocking_emoji_count"] = len(mocking_emojis)
        result["detected_mocking_emojis"] = [e["emoji"] for e in mocking_emojis]
        
        if not mocking_emojis:
            return result
        
        # Check for positive text with negative/mocking emojis
        positive_word_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_emoji_toxicity = sum(e["toxicity_boost"] for e in mocking_emojis) / len(mocking_emojis)
        
        result["emoji_toxicity_total"] = negative_emoji_toxicity
        
        # PATTERN 1: Positive words + mocking emoji = sarcasm
        if positive_word_count > 0 and negative_emoji_toxicity > 0.6:
            result["is_sarcastic"] = True
            result["sarcasm_type"] = "positive_text_with_mocking_emoji"
            result["positive_text_with_negative_emoji"] = True
            # Score: how many positive words × emoji toxicity
            result["sarcasm_score"] = min(0.95, (positive_word_count * 0.3) + negative_emoji_toxicity)
            return result
        
        # PATTERN 2: Explicit sarcasm phrases detected
        for phrase in self.sarcasm_phrases:
            if phrase in text_lower:
                result["is_sarcastic"] = True
                result["sarcasm_type"] = "explicit_sarcasm_phrase"
                result["sarcasm_score"] = 0.7 + (negative_emoji_toxicity * 0.25)
                return result
        
        # PATTERN 3: Just negative/mocking emojis (even without positive text)
        if negative_emoji_toxicity > 0.75:
            result["is_sarcastic"] = True
            result["sarcasm_type"] = "heavy_negative_emoji"
            result["sarcasm_score"] = negative_emoji_toxicity
            return result
        
        return result
    
    def get_emoji_sentiment_sum(self, emojis: List[Dict]) -> Tuple[float, float, List[str]]:
        """
        Calculate overall emoji sentiment.
        
        Args:
            emojis: List of emoji dicts
        
        Returns:
            Tuple: (total_sentiment, toxicity_boost, emoji_list)
        """
        if not emojis:
            return 0.0, 0.0, []
        
        total_sentiment = sum(e["sentiment"] for e in emojis)
        total_toxicity = sum(e["toxicity_boost"] for e in emojis)
        emoji_chars = [e["emoji"] for e in emojis]
        
        # Normalize by count
        avg_sentiment = total_sentiment / len(emojis) if emojis else 0.0
        avg_toxicity = total_toxicity / len(emojis) if emojis else 0.0
        
        return avg_sentiment, avg_toxicity, emoji_chars
    
    def analyze_sarcastic_toxicity(
        self,
        text: str,
        toxicity_score_from_model: float
    ) -> Dict:
        """
        Complete sarcasm-aware toxicity analysis.
        
        Combines emoji sarcasm detection with model toxicity score.
        
        Args:
            text: Original text with emojis
            toxicity_score_from_model: Already-computed toxicity score from model
        
        Returns:
            Dict with enhanced toxicity scoring and sarcasm info
        """
        # Extract emojis
        emojis = self.extract_emojis(text)
        
        # Detect sarcasm pattern
        sarcasm_result = self.detect_sarcasm_pattern(text, emojis)
        
        # Get emoji sentiment
        emoji_sentiment, emoji_toxicity, emoji_list = self.get_emoji_sentiment_sum(emojis)
        
        # Adjust toxicity based on sarcasm
        adjusted_toxicity = toxicity_score_from_model
        
        if sarcasm_result["is_sarcastic"]:
            # Boost toxicity score if sarcasm detected
            sarcasm_boost = sarcasm_result["sarcasm_score"] * 0.4  # Up to 40% boost
            adjusted_toxicity = min(1.0, toxicity_score_from_model + sarcasm_boost)
        
        # Additional boost if very negative emoji sentiment
        if emoji_sentiment < -0.5:
            emoji_boost = abs(emoji_sentiment) * 0.2  # Up to 20% boost
            adjusted_toxicity = min(1.0, adjusted_toxicity + emoji_boost)
        
        return {
            "is_sarcastic": sarcasm_result["is_sarcastic"],
            "sarcasm_type": sarcasm_result["sarcasm_type"],
            "sarcasm_score": sarcasm_result["sarcasm_score"],
            "mocking_emoji_count": sarcasm_result["mocking_emoji_count"],
            "detected_mocking_emojis": sarcasm_result["detected_mocking_emojis"],
            "emoji_sentiment_avg": emoji_sentiment,
            "emoji_toxicity_avg": emoji_toxicity,
            "all_emojis": emoji_list,
            "original_toxicity_score": toxicity_score_from_model,
            "adjusted_toxicity_score": adjusted_toxicity,
            "sarcasm_adjustment": adjusted_toxicity - toxicity_score_from_model,
        }


# Global instance
emoji_sarcasm_detector = EmojiSarcasmDetector()


def detect_emoji_sarcasm(text: str, base_toxicity: float) -> Dict:
    """
    Convenience function for emoji-aware sarcasm detection.
    
    Args:
        text: Comment text with emojis
        base_toxicity: Toxicity score from model
    
    Returns:
        Sarcasm analysis dict
    """
    return emoji_sarcasm_detector.analyze_sarcastic_toxicity(text, base_toxicity)


if __name__ == "__main__":
    # Test examples
    test_cases = [
        ("Wow you're clearly a genius 🤡", 0.3),
        ("Yeah because that idea worked so well 🤦‍♂️", 0.25),
        ("Great job ruining the whole argument 😂", 0.4),
        ("Brilliant thinking as always 🙃", 0.35),
        ("Wow, another masterpiece of bad ideas 👏", 0.32),
        ("This is actually great content", 0.1),
    ]
    
    print("=" * 80)
    print("EMOJI SARCASM DETECTION TEST")
    print("=" * 80)
    
    for text, base_score in test_cases:
        result = detect_emoji_sarcasm(text, base_score)
        print(f"\nText: {text}")
        print(f"  Base toxicity score: {base_score:.2f}")
        print(f"  Is sarcastic: {result['is_sarcastic']}")
        print(f"  Sarcasm type: {result['sarcasm_type']}")
        print(f"  Adjusted toxicity: {result['adjusted_toxicity_score']:.2f}")
        print(f"  Mocking emojis: {result['detected_mocking_emojis']}")
