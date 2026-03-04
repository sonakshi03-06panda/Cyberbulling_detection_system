"""
Batch analyze multiple YouTube URLs.
Usage: python batch_analyze.py urls.txt
"""

import sys
import os
from comment_fetcher import CommentFetcher
from toxicity_analyzer import analyze_comments, ToxicityAnalyzer
from report_generator import ToxicityReportGenerator
import json

# Load API keys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

YOUTUBE_KEY = os.getenv("YOUTUBE_API_KEY")
fetcher = CommentFetcher(YOUTUBE_KEY)
report_gen = ToxicityReportGenerator("reports")

# Initialize analyzer once
analyzer = ToxicityAnalyzer("models/final_model")


def batch_analyze(file_path: str, max_comments: int = 500):
    """Analyze multiple URLs from a file."""
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
    
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(urls)}] Analyzing: {url}")
        print('='*70)
        
        try:
            # Fetch
            print(f"  Fetching comments...")
            comments = fetcher.fetch_comments(url, max_comments)
            
            if not comments:
                print(f"  ❌ No comments found")
                results.append({
                    "url": url,
                    "status": "failed",
                    "error": "No comments found"
                })
                continue
            
            print(f"  ✓ Fetched {len(comments)} comments")
            
            # Analyze
            print(f"  Analyzing toxicity...")
            df = analyze_comments(comments, analyzer)
            
            # Generate report
            print(f"  Generating report...")
            report_path = report_gen.generate_report(df, url)
            
            # Stats
            toxic_count = df["is_toxic"].sum()
            toxic_rate = 100 * toxic_count / len(df)
            
            result = {
                "url": url,
                "status": "success",
                "total_comments": len(df),
                "toxic_comments": int(toxic_count),
                "toxic_rate": float(toxic_rate),
                "avg_confidence": float(df["max_confidence"].mean()),
                "report": report_path
            }
            results.append(result)
            
            print(f"  ✓ Report: {os.path.basename(report_path)}")
            print(f"    Toxic: {toxic_count}/{len(df)} ({toxic_rate:.1f}%)")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "url": url,
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH ANALYSIS SUMMARY")
    print('='*70)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n✓ Completed:")
        for r in successful:
            print(f"  {r['url']}")
            print(f"    → {r['toxic_comments']}/{r['total_comments']} toxic ({r['toxic_rate']:.1f}%)")
    
    if failed:
        print("\n✗ Failed:")
        for r in failed:
            print(f"  {r['url']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    summary_path = "reports/batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_analyze.py <urls_file> [max_comments]")
        print("\nExample urls.txt:")
        print("https://youtube.com/watch?v=...")
        print("# (Only include YouTube URLs in your urls.txt)")
        sys.exit(1)
    
    file_path = sys.argv[1]
    max_comments = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    batch_analyze(file_path, max_comments)
