"""
Fetch comments from YouTube.
Requires: google-api-python-client
"""

import os
import time
from typing import List, Dict


class YouTubeCommentFetcher:
    """Fetch comments from YouTube videos."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from googleapiclient.discovery import build
            self.youtube = build("youtube", "v3", developerKey=api_key)
        except ImportError:
            raise ImportError("Install google-api-python-client: pip install google-api-python-client")
    
    def get_video_id_from_url(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        if "youtube.com/watch?v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError(f"Invalid YouTube URL: {url}")
    
    def fetch_comments(self, video_url: str, max_results: int = 1000) -> List[Dict]:
        """
        Fetch comments from a YouTube video.
        Returns list of dicts with keys: text, author, timestamp, likes, replies
        Fetches ALL available comments if max_results is very high (e.g., 999999)
        """
        video_id = self.get_video_id_from_url(video_url)
        comments = []
        max_retries = 5
        
        try:
            # Try with different sort orders to maximize comment retrieval
            # When ordered by relevance hits a wall, try by time
            sort_orders = ["time", "relevance"]
            
            for sort_order in sort_orders:
                if len(comments) > 0:
                    print(f"[INFO] Switching to sort order: {sort_order} to try fetching more comments...")
                
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText",
                    order=sort_order
                )
                
                page_count = 0
                consecutive_empty_pages = 0
                max_consecutive_empty = 10  # More lenient empty page tolerance
                pages_without_new_comments = 0
                
                while request and consecutive_empty_pages < max_consecutive_empty:
                    retry_count = 0
                    parent_request = None
                    while retry_count < max_retries:
                        try:
                            response = request.execute()
                            parent_request = None  # Clear error state on success
                            break
                        except Exception as api_error:
                            error_msg = str(api_error)
                            retry_count += 1
                            parent_request = api_error
                            
                            # Check if it's a network/SSL error
                            if "EOF occurred" in error_msg or "SSL" in error_msg or "connection" in error_msg.lower():
                                if retry_count < max_retries:
                                    wait_time = 2 ** retry_count
                                    print(f"[WARNING] Network error, retrying in {wait_time}s... (attempt {retry_count}/{max_retries})")
                                    time.sleep(wait_time)
                                    continue
                                else:
                                    raise Exception(f"Network error after {max_retries} retries: {error_msg}")
                            
                            # For other API errors, handle normally
                            if len(comments) > 0 and ("400" in error_msg or "processingFailure" in error_msg):
                                print(f"[INFO] Pagination stopped after {page_count} pages ({len(comments)} comments fetched). API error: {error_msg}")
                                request = None
                                break
                            elif "403" in error_msg:
                                raise Exception("YouTube API quota exceeded or access denied. Try again later or check your API key.")
                            elif "404" in error_msg:
                                raise Exception(f"Video not found. Check if the URL is correct: {video_url}")
                            elif "commentsDisabled" in error_msg or "disabled" in error_msg.lower():
                                raise Exception("Comments are disabled on this video.")
                            else:
                                raise Exception(f"YouTube API error: {error_msg}")
                
                    if parent_request:
                        break  # Exit if we had an unrecoverable error
                    
                    page_count += 1
                    items = response.get("items", [])
                    
                    if not items:
                        consecutive_empty_pages += 1
                        pages_without_new_comments += 1
                        if consecutive_empty_pages % 3 == 0:
                            print(f"[WARNING] Empty page {page_count} ({pages_without_new_comments} consecutive empty). Trying to continue...")
                    else:
                        consecutive_empty_pages = 0
                        pages_without_new_comments = 0
                        
                        initial_count = len(comments)
                        for item in items:
                            snippet = item["snippet"]["topLevelComment"]["snippet"]
                            comment_text = snippet.get("textDisplay", "")
                            
                            # Check for duplicates (sometimes YouTube API returns duplicates during pagination)
                            if not any(c["text"] == comment_text for c in comments):
                                comments.append({
                                    "platform": "YouTube",
                                    "video_url": video_url,
                                    "text": comment_text,
                                    "author": snippet.get("authorDisplayName", ""),
                                    "timestamp": snippet.get("publishedAt", ""),
                                    "likes": snippet.get("likeCount", 0),
                                    "replies": snippet.get("replyCount", 0)
                                })
                        
                        new_count = len(comments) - initial_count
                        if new_count == 0:
                            pages_without_new_comments += 1
                        
                        if page_count % 5 == 0:
                            print(f"[INFO] Fetched {len(comments)} comments so far (page {page_count}, sort: {sort_order})...")
                        
                        # Stop if we've reached max_results
                        if len(comments) >= max_results:
                            print(f"[INFO] Reached max_results limit of {max_results} comments")
                            return comments[:max_results]
                    
                    # Continue to next page if available
                    if "nextPageToken" in response and pages_without_new_comments < 5:
                        next_token_retry = 0
                        max_token_retries = 3
                        while next_token_retry < max_token_retries:
                            try:
                                time.sleep(0.1)
                                request = self.youtube.commentThreads().list(
                                    part="snippet",
                                    videoId=video_id,
                                    pageToken=response["nextPageToken"],
                                    maxResults=100,
                                    textFormat="plainText",
                                    order=sort_order
                                )
                                break
                            except Exception as e:
                                next_token_retry += 1
                                if next_token_retry < max_token_retries:
                                    wait_time = 1 + next_token_retry
                                    print(f"[WARNING] Pagination retry {next_token_retry}/{max_token_retries}: {str(e)[:80]}")
                                    time.sleep(wait_time)
                                else:
                                    print(f"[WARNING] Failed pagination after retries. Stopping this sort order.")
                                    request = None
                                    break
                    else:
                        if pages_without_new_comments >= 5:
                            print(f"[INFO] No new comments for {pages_without_new_comments} consecutive pages. Pagination hit.")
                        print(f"[INFO] ✓ Reached end of {sort_order} pagination. Total so far: {len(comments)} comments.")
                        break
        
        except Exception as e:
            raise Exception(f"Failed to fetch comments: {str(e)}")
        
        print(f"[INFO] === FINAL RESULT === Fetched {len(comments)} total comments across all sort orders")
        return comments[:max_results]


class CommentFetcher:
    """Simple interface for fetching YouTube comments."""

    def __init__(self, youtube_api_key: str = None):
        self.youtube_fetcher = None
        if youtube_api_key:
            self.youtube_fetcher = YouTubeCommentFetcher(youtube_api_key)

    def fetch_comments(self, url: str, max_results: int = 1000) -> List[Dict]:
        """Fetch comments from a YouTube URL."""
        if "youtube.com" in url or "youtu.be" in url:
            if not self.youtube_fetcher:
                raise ValueError("YouTube API key not configured")
            return self.youtube_fetcher.fetch_comments(url, max_results)
        else:
            raise ValueError(f"Unsupported platform for URL: {url}")


if __name__ == "__main__":
    # Example usage (requires credentials)
    import os
    from dotenv import load_dotenv
    
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    yt_key = os.getenv("YOUTUBE_API_KEY")
    fetcher = CommentFetcher(yt_key)
    
    # Example
    # comments = fetcher.fetch_comments("https://www.youtube.com/watch?v=...")
    # print(f"Fetched {len(comments)} comments")
