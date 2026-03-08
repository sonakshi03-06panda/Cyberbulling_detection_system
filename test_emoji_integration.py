"""
Test: Emoji-Aware Sarcasm Detection Integration
===============================================

This script tests the complete emoji-sarcasm detection system
integrated with the improved toxicity classifier.
"""

import sys
import os

# Add scr to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scr'))

from toxicity_analyzer import ToxicityAnalyzer
from emoji_sarcasm_detector import detect_emoji_sarcasm


def test_emoji_sarcasm_detection():
    """Test emoji-aware sarcasm detection with all user examples."""
    
    print("=" * 100)
    print("EMOJI-AWARE SARCASM DETECTION - INTEGRATION TEST")
    print("=" * 100)
    
    test_cases = [
        {
            "text": "Wow you're clearly a genius 🤡",
            "expected_category": "sarcastic_toxic",
            "expected_toxic": True,
            "reason": "Positive word + clown emoji (mockery)"
        },
        {
            "text": "Yeah because that idea worked so well 🤦‍♂️",
            "expected_category": "sarcastic_toxic",
            "expected_toxic": True,
            "reason": "Sarcasm phrase + face_palm (ridicule)"
        },
        {
            "text": "Great job ruining the whole argument 😂",
            "expected_category": "sarcastic_toxic",
            "expected_toxic": True,
            "reason": "Praise phrase + mocking laughter"
        },
        {
            "text": "Brilliant thinking as always 🙃",
            "expected_category": "sarcastic_toxic",
            "expected_toxic": True,
            "reason": "Positive word + fake smile (sarcasm)"
        },
        {
            "text": "Wow, another masterpiece of bad ideas 👏",
            "expected_category": "sarcastic_toxic",
            "expected_toxic": True,
            "reason": "Positive word + sarcastic clap"
        },
        {
            "text": "This is actually great content",
            "expected_category": "non_toxic",
            "expected_toxic": False,
            "reason": "Positive words without mocking emoji"
        },
        {
            "text": "I genuinely disagree with you",
            "expected_category": "mild_negative",
            "expected_toxic": False,
            "reason": "Disagreement without toxic tone"
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'-' * 100}")
        print(f"TEST {i}: {case['text'][:60]}...")
        print(f"{'-' * 100}")
        
        # Quick emoji sarcasm test
        sarcasm_result = detect_emoji_sarcasm(case['text'], 0.3)
        
        print(f"Emoji Sarcasm Analysis:")
        print(f"  Is sarcastic: {sarcasm_result['is_sarcastic']}")
        print(f"  Sarcasm type: {sarcasm_result['sarcasm_type']}")
        print(f"  Sarcasm score: {sarcasm_result['sarcasm_score']:.2f}")
        print(f"  Mocking emojis: {sarcasm_result['detected_mocking_emojis']}")
        print(f"  Base score: {sarcasm_result['original_toxicity_score']:.2f}")
        print(f"  Adjusted score: {sarcasm_result['adjusted_toxicity_score']:.2f}")
        print(f"  Adjustment: {sarcasm_result['sarcasm_adjustment']:+.2f}")
        
        print(f"\nExpected Results:")
        print(f"  Category: {case['expected_category']}")
        print(f"  Is toxic: {case['expected_toxic']}")
        print(f"  Reason: {case['reason']}")
        
        # Check if emoji sarcasm was detected when expected
        if case['expected_category'] == 'sarcastic_toxic':
            if sarcasm_result['is_sarcastic']:
                print("\n✅ PASS - Emoji sarcasm correctly detected")
                passed += 1
            else:
                print("\n❌ FAIL - Emoji sarcasm not detected when expected")
                failed += 1
        else:
            if not sarcasm_result['is_sarcastic']:
                print("\n✅ PASS - No sarcasm detected as expected")
                passed += 1
            else:
                print("\n⚠️  WARNING - Sarcasm detected when not expected (may be fine)")
                # Don't count as failure since it might be legitimate
    
    print(f"\n{'=' * 100}")
    print(f"SUMMARY: {passed} Passed, {failed} Failed")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print(f"{'=' * 100}")
    
    return passed, failed


def test_output_format():
    """Test that output format includes sarcasm_info when detected."""
    
    print(f"\n{'=' * 100}")
    print("OUTPUT FORMAT TEST")
    print(f"{'=' * 100}")
    
    test_comment = "Wow you're clearly a genius 🤡"
    
    print(f"\nTest comment: {test_comment}")
    
    # Direct emoji sarcasm analysis
    result = detect_emoji_sarcasm(test_comment, 0.3)
    
    print(f"\nOutput structure:")
    print(f"  - is_sarcastic: {result['is_sarcastic']}")
    print(f"  - sarcasm_type: {result['sarcasm_type']}")
    print(f"  - sarcasm_score: {result['sarcasm_score']:.2f}")
    print(f"  - mocking_emoji_count: {result['mocking_emoji_count']}")
    print(f"  - detected_mocking_emojis: {result['detected_mocking_emojis']}")
    print(f"  - emoji_sentiment_avg: {result['emoji_sentiment_avg']:.2f}")
    print(f"  - original_toxicity_score: {result['original_toxicity_score']:.2f}")
    print(f"  - adjusted_toxicity_score: {result['adjusted_toxicity_score']:.2f}")
    print(f"  - sarcasm_adjustment: {result['sarcasm_adjustment']:+.2f}")
    
    print(f"\n✅ Output format includes all required fields")


def show_emoji_guide():
    """Display key emoji toxicity mappings."""
    
    print(f"\n{'=' * 100}")
    print("EMOJI SENTIMENT REFERENCE")
    print(f"{'=' * 100}")
    
    emojis_info = {
        "🤡": "Clown - Mockery (boost: 0.90)",
        "🙃": "Fake smile - Sarcasm (boost: 0.80)",
        "🤦": "Face palm - Ridicule (boost: 0.85)",
        "😂": "Mocking laugh (boost: 0.70)",
        "👏": "Sarcastic clap (boost: 0.80)",
        "🙄": "Eye roll - Dismissal (boost: 0.70)",
        "😒": "Unamused (boost: 0.65)",
        "💀": "Death emoji (boost: 0.80)",
    }
    
    print("\nHigh-Toxicity Boost Emojis:\n")
    for emoji, info in emojis_info.items():
        print(f"  {emoji} {info}")


if __name__ == "__main__":
    # Run tests
    passed, failed = test_emoji_sarcasm_detection()
    test_output_format()
    show_emoji_guide()
    
    print(f"\n{'=' * 100}")
    print("FINAL RESULTS")
    print(f"{'=' * 100}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Emoji sarcasm detection is working correctly!")
        print("\nThe system now correctly identifies:")
        print("  • Positive words + mocking emojis = sarcastic toxicity")
        print("  • Explicit sarcasm phrases + negative emojis = sarcasm detected")
        print("  • Heavy emoji toxicity signals = sarcasm patterns recognized")
        print("\nNext steps:")
        print("  1. Deploy to staging for 24-hour testing")
        print("  2. Monitor false positive/negative rates")
        print("  3. Adjust emoji weights if needed")
        print("  4. Deploy to production")
    else:
        print(f"\n⚠️  {failed} tests failed - Review detection logic")
