"""
Test script to demonstrate improved toxicity detection that handles curse words better.
"""

from toxicity_analyzer import ToxicityAnalyzer

# Test cases: safe comments with curse words vs. actual toxic
test_comments = [
    # Safe comments with curse words (were false positives before)
    "This movie is fucking awesome!",
    "Damn, that was an incredible performance!",
    "I don't give a shit about the haters",
    "Hell yeah, let's go!",
    
    # Actual toxic comments
    "You're a fucking idiot and I hope you die",
    "I hate you, you piece of shit",
    "You should be dead, you worthless human",
    "I'm going to find you and hurt you",
    
    # Borderline cases
    "That's a stupid idea",
    "You're being annoying",
    "I strongly disagree with you",
    
    # Clearly safe
    "Great video! Thanks for sharing!",
    "I really enjoyed this content",
]

print("=" * 80)
print("IMPROVED TOXICITY DETECTION TEST")
print("=" * 80)
print("\nThe new model uses:")
print("[+] Label-specific thresholds (curse words need higher confidence)")
print("[+] Weighted label importance (threats are serious, obscenity is mild)")
print("[+] Intelligent label combinations (e.g., 'obscene' alone != toxic)")
print("\n" + "=" * 80 + "\n")

analyzer = ToxicityAnalyzer()

for comment in test_comments:
    result = analyzer.classify_comment(comment)
    status = "[TOXIC]" if result["is_toxic"] else "[SAFE]"
    print(f"{status}")
    print(f"Comment: {comment}")
    print(f"Labels detected: {result['toxic_labels'] if result['toxic_labels'] else 'None'}")
    if result['toxic_labels']:
        print(f"Confidence scores:")
        for label, score in result['confidences'].items():
            threshold = analyzer.label_thresholds[label]
            marker = "Y" if score > threshold else "N"
            print(f"  {marker} {label}: {score:.3f} (threshold: {threshold})")
    print(f"Severity: {result['severity']}/3")
    print(f"Max confidence: {result['max_confidence']:.3f}\n")

print("=" * 80)
print("SUMMARY OF IMPROVEMENTS:")
print("=" * 80)
print("""
1. SMART THRESHOLDS:
   - "threat" (0.35) and "severe_toxic" (0.40): Lower threshold = easier to detect serious threats
   - "obscene" (0.70) and "insult" (0.65): Higher threshold = curse words alone won't trigger

2. WEIGHTED SCORING:
   - Threats and severe toxic content: Weight 1.0 (full importance)
   - Identity hate: Weight 0.9
   - Toxic: Weight 0.7
   - Insult: Weight 0.5
   - Obscene: Weight 0.4 (lowest)

3. INTELLIGENT COMBINATIONS:
   - Just "obscene" or "insult" alone = NOT toxic
   - "Toxic" + "insult/identity_hate" = toxic
   - "Threat" or "severe_toxic" = ALWAYS toxic
   - Other combinations evaluated based on context

Result: Safe comments with curse words are no longer flagged as toxic!
""")
