"""
Quick Emoji Sarcasm Detection Test (Unicode-safe)
"""

import sys
sys.path.insert(0, 'scr')

from emoji_sarcasm_detector import detect_emoji_sarcasm

print("=" * 80)
print("EMOJI-AWARE SARCASM DETECTION - QUICK TEST")
print("=" * 80)

test_cases = [
    ("Wow you're clearly a genius [clown]", "sarcastic_toxic"),
    ("Yeah because that idea worked so well [facepalm]", "sarcastic_toxic"),
    ("Great job ruining the whole argument [laugh]", "sarcastic_toxic"),
    ("Brilliant thinking as always [smile]", "sarcastic_toxic"),
    ("Wow, another masterpiece of bad ideas [clap]", "sarcastic_toxic"),
    ("This is actually great content", "non_toxic"),
    ("I genuinely disagree with you", "non_toxic"),
]

actual_cases = [
    "Wow you're clearly a genius 🤡",
    "Yeah because that idea worked so well 🤦‍♂️",
    "Great job ruining the whole argument 😂",
    "Brilliant thinking as always 🙃",
    "Wow, another masterpiece of bad ideas 👏",
    "This is actually great content",
    "I genuinely disagree with you",
]

passed = 0
failed = 0

for i, (display_text, expected) in enumerate(test_cases):
    actual_text = actual_cases[i]
    result = detect_emoji_sarcasm(actual_text, 0.3)
    
    print(f"\nTest {i+1}: {display_text}")
    
    if expected == "sarcastic_toxic":
        if result['is_sarcastic']:
            print(f"  ✅ PASS - Sarcasm detected (score: {result['sarcasm_score']:.2f}, boost: {result['sarcasm_adjustment']:+.2f})")
            passed += 1
        else:
            print(f"  ❌ FAIL - Expected sarcasm but not detected")
            failed += 1
    else:
        if not result['is_sarcastic']:
            print(f"  ✅ PASS - No sarcasm detected (as expected)")
            passed += 1
        else:
            print(f"  ❌ FAIL - Unexpected sarcasm detection")
            failed += 1

print(f"\n{'=' * 80}")
print(f"RESULTS: {passed} Passed, {failed} Failed ({(passed/(passed+failed)*100):.1f}% success)")
print(f"{'=' * 80}")

if failed == 0:
    print("\n✅ ALL TESTS PASSED!")
    print("\nEmoji-aware sarcasm detection is working correctly:")
    print("  • Positive words + mocking emojis = Sarcasm detected")
    print("  • Base toxicity scores boosted by emoji context")
    print("  • Non-sarcastic content correctly classified")
else:
    print(f"\n❌ {failed} test(s) failed")
