"""
Diagnostic script to test YouTube API connectivity and comment fetching.
Helps troubleshoot "No comments found" errors.
"""

import os
from dotenv import load_dotenv
from comment_fetcher import YouTubeCommentFetcher

def diagnose():
    print("=" * 80)
    print("YouTube API Diagnostic Tool")
    print("=" * 80)
    
    # Check 1: API Key is set
    print("\n[CHECK 1] YOUTUBE_API_KEY Configuration")
    print("-" * 80)
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        print("ERROR: YOUTUBE_API_KEY not found in environment or .env file")
        print("\nSolution:")
        print("  1. Create or edit .env file in the project root:")
        print("     YOUTUBE_API_KEY=your_actual_api_key_here")
        print("  2. Make sure you have a valid YouTube Data API v3 key from Google Cloud Console")
        print("  3. Restart the Flask app after adding the key")
        return False
    else:
        print("OK: API key is set")
        print(f"    Key (first 20 chars): {api_key[:20]}...")
    
    # Check 2: Can we initialize the YouTube API?
    print("\n[CHECK 2] YouTube API Initialization")
    print("-" * 80)
    try:
        fetcher = YouTubeCommentFetcher(api_key)
        print("OK: YouTube API client initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize YouTube API client: {e}")
        return False
    
    # Check 3: Can we connect to YouTube API?
    print("\n[CHECK 3] YouTube API Connectivity")
    print("-" * 80)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - always has comments
    print(f"Testing with URL: {test_url}")
    
    try:
        comments = fetcher.fetch_comments(test_url, max_results=5)
        if comments:
            print(f"OK: Successfully fetched {len(comments)} comments from test video")
            print(f"    Sample comment: {comments[0]['text'][:60]}...")
        else:
            print("WARNING: No comments returned (video might have comments disabled)")
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: {error_msg}")
        
        if "403" in error_msg:
            print("\nLikely causes:")
            print("  - API quota exceeded (limit: 10,000 units/day)")
            print("  - API key permissions not set correctly in Google Cloud Console")
            print("  - Video is from a channel that restricts comment access")
        elif "404" in error_msg or "not found" in error_msg.lower():
            print("\nLikely cause: Video does not exist or is unavailable")
        elif "disabled" in error_msg.lower():
            print("\nLikely cause: Comments are disabled on this video")
        return False
    
    # Check 4: Test with user's URL
    print("\n[CHECK 4] Test Your YouTube Video")
    print("-" * 80)
    user_url = input("Enter a YouTube video URL to test (or press Enter to skip): ").strip()
    
    if user_url:
        try:
            comments = fetcher.fetch_comments(user_url, max_results=10)
            print(f"\nSUCCESS: Fetched {len(comments)} comments!")
            if comments:
                print("\nSample comments:")
                for i, comment in enumerate(comments[:3], 1):
                    print(f"  {i}. {comment['text'][:70]}...")
            else:
                print("\nWARNING: No comments found. Possible reasons:")
                print("  - Comments are disabled on this video")
                print("  - Video has very few or no comments")
                print("  - API access is restricted for this channel")
        except Exception as e:
            print(f"\nERROR: {e}")
    
    print("\n" + "=" * 80)
    print("Diagnostic Results")
    print("=" * 80)
    print("""
SUMMARY:
  If all checks passed, the YouTube API configuration is working.
  
TROUBLESHOOTING STEPS:
  1. Ensure YOUTUBE_API_KEY is set correctly in your .env file
  2. Check your API quota at https://console.cloud.google.com/
  3. Try testing with a different video (some disable comments)
  4. Verify the video URL format is correct:
     - https://www.youtube.com/watch?v=VIDEO_ID
     - https://youtu.be/VIDEO_ID
  5. Restart the Flask app after any changes to .env
  
API LIMITS:
  - YouTube API v3: 10,000 units/day (shared across all requests)
  - commentThreads.list: ~300 units per call
  - You can fetch comments from ~30 videos per day
  
STILL HAVING ISSUES?
  Check Google Cloud Console for error logs and quota usage
    """)

if __name__ == "__main__":
    success = diagnose()
    exit(0 if success else 1)
