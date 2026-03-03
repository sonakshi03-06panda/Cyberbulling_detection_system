"""
Demo: Generate a toxicity report without API credentials.
Tests the analysis and report generation with sample data.
"""

import pandas as pd
from toxicity_analyzer import ToxicityAnalyzer
from report_generator import ToxicityReportGenerator

# Sample comments (simulating fetched comments)
SAMPLE_COMMENTS = [
    {"text": "Great video! Really enjoyed it.", "platform": "YouTube", "author": "User1"},
    {"text": "This is amazing content!", "platform": "YouTube", "author": "User2"},
    {"text": "I love this!", "platform": "YouTube", "author": "User3"},
    {"text": "You're an idiot", "platform": "YouTube", "author": "User4"},
    {"text": "This is stupid", "platform": "YouTube", "author": "User5"},
    {"text": "I hate you", "platform": "YouTube", "author": "User6"},
    {"text": "You should die", "platform": "YouTube", "author": "User7"},
    {"text": "You're disgusting", "platform": "YouTube", "author": "User8"},
    {"text": "I'll hurt you for this", "platform": "YouTube", "author": "User9"},
    {"text": "You're a [slur]", "platform": "YouTube", "author": "User10"},
]


def run_demo():
    """Run the demo."""
    print("🚨 Toxicity Report System - Demo\n")
    
    # Initialize analyzer
    print("1. Loading toxicity model...")
    analyzer = ToxicityAnalyzer("models/final_model")
    print("   ✓ Model loaded\n")
    
    # Analyze comments
    print("2. Analyzing comments...")
    texts = [c["text"] for c in SAMPLE_COMMENTS]
    classifications = analyzer.classify_batch(texts)
    print(f"   ✓ Analyzed {len(texts)} comments\n")
    
    # Merge results
    for i, comment in enumerate(SAMPLE_COMMENTS):
        comment.update(classifications[i])
    
    # Create DataFrame
    df = pd.DataFrame(SAMPLE_COMMENTS)
    
    # Display results
    print("3. Results Summary:")
    print(f"   Total comments: {len(df)}")
    print(f"   Toxic comments: {df['is_toxic'].sum()}")
    print(f"   Toxicity rate: {100 * df['is_toxic'].sum() / len(df):.1f}%\n")
    
    print("   Per-label breakdown:")
    all_labels = []
    for labels_list in df["toxic_labels"]:
        all_labels.extend(labels_list)
    
    from collections import Counter
    label_counts = Counter(all_labels)
    for label, count in label_counts.most_common():
        print(f"     • {label}: {count}")
    
    print("\n   Top toxic comments:")
    toxic_df = df[df["is_toxic"]].sort_values("max_confidence", ascending=False)
    for i, row in toxic_df.head(5).iterrows():
        print(f"     [{row['severity']}] {row['text'][:50]}... (confidence: {row['max_confidence']:.1%})")
    
    # Generate report
    print("\n4. Generating HTML report...")
    report_gen = ToxicityReportGenerator("reports")
    report_path = report_gen.generate_report(
        df, 
        "https://youtube.com/watch?v=demo",
        "Demo Report - Sample Comments"
    )
    print(f"   ✓ Report saved to {report_path}\n")
    
    print("✅ Demo completed! Open the HTML report in your browser.")
    

if __name__ == "__main__":
    run_demo()
