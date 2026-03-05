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
    
    def fetch_replies(self, parent_id: str, video_url: str, max_replies: int = 1000) -> List[Dict]:
        """
        Fetch all reply comments for a given parent comment.
        
        Args:
            parent_id: The ID of the parent comment thread
            video_url: The video URL (for metadata)
            max_replies: Maximum replies to fetch per parent
            
        Returns:
            List of reply comment dicts
        """
        replies = []
        max_retries = 3
        
        try:
            request = self.youtube.comments().list(
                part="snippet",
                parentId=parent_id,
                maxResults=100,
                textFormat="plainText"
            )
            
            page_count = 0
            while request:
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        response = request.execute()
                        break
                    except Exception as api_error:
                        error_msg = str(api_error)
                        retry_count += 1
                        
                        if retry_count < max_retries:
                            wait_time = 1 + retry_count
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"[WARNING] Could not fetch replies for parent {parent_id[:20]}...")
                            return replies
                
                page_count += 1
                items = response.get("items", [])
                
                if not items:
                    break
                
                for item in items:
                    snippet = item["snippet"]
                    reply_text = snippet.get("textDisplay", "")
                    
                    # Check for duplicates
                    if not any(r["text"] == reply_text for r in replies):
                        replies.append({
                            "platform": "YouTube",
                            "video_url": video_url,
                            "text": reply_text,
                            "author": snippet.get("authorDisplayName", ""),
                            "timestamp": snippet.get("publishedAt", ""),
                            "likes": snippet.get("likeCount", 0),
                            "replies": 0,
                            "is_reply": True,
                            "parent_id": parent_id
                        })
                
                if len(replies) >= max_replies:
                    break
                
                # Continue to next page if available
                if "nextPageToken" in response:
                    request = self.youtube.comments().list(
                        part="snippet",
                        parentId=parent_id,
                        pageToken=response["nextPageToken"],
                        maxResults=100,
                        textFormat="plainText"
                    )
                    time.sleep(0.1)
                else:
                    break
            
        except Exception as e:
            print(f"[WARNING] Error fetching replies: {str(e)[:100]}")
        
        return replies
    
    def fetch_comments(self, video_url: str, max_results: int = 10000, fetch_replies: bool = True, max_replies_per_comment: int = 5) -> List[Dict]:
        """
        Fetch all comments from a YouTube video, including replies.
        
        Args:
            video_url: YouTube video URL
            max_results: Maximum total comments to fetch
            fetch_replies: Whether to fetch reply comments (increases total count significantly)
            max_replies_per_comment: Maximum replies to fetch per top-level comment
            
        Returns:
            List of dicts with keys: text, author, timestamp, likes, replies, is_reply
        """
        video_id = self.get_video_id_from_url(video_url)
        comments = []
        max_retries = 5
        total_comments_on_video = 0
        
        try:
            # Try with different sort orders to maximize comment retrieval
            sort_orders = ["relevance", "time"]  # Relevance first (gets most engaged), then time (gets recent)
            
            for sort_order in sort_orders:
                if len(comments) > 0:
                    print(f"[INFO] Switching to sort order: {sort_order} to fetch additional comments...")
                
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText",
                    order=sort_order
                )
                
                page_count = 0
                consecutive_empty_pages = 0
                max_consecutive_empty = 50  # Increased tolerance for more thorough fetching
                pages_without_new_comments = 0
                max_pages_without_new = 20  # Increased for exhaustive search
                
                while request and consecutive_empty_pages < max_consecutive_empty:
                    retry_count = 0
                    parent_request = None
                    while retry_count < max_retries:
                        try:
                            response = request.execute()
                            parent_request = None
                            
                            # Get total comment count if available
                            if not total_comments_on_video and "pageInfo" in response:
                                total_comments_on_video = response.get("pageInfo", {}).get("totalResults", 0)
                            
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
                                print(f"[INFO] Pagination stopped after {page_count} pages. API error: {error_msg}")
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
                        break
                    
                    page_count += 1
                    items = response.get("items", [])
                    
                    if not items:
                        consecutive_empty_pages += 1
                        pages_without_new_comments += 1
                        if page_count % 10 == 0:
                            print(f"[INFO] Empty pages: {consecutive_empty_pages}. Continuing aggressive search...")
                    else:
                        consecutive_empty_pages = 0
                        pages_without_new_comments = 0
                        
                        initial_count = len(comments)
                        for item in items:
                            snippet = item["snippet"]["topLevelComment"]["snippet"]
                            comment_text = snippet.get("textDisplay", "")
                            
                            # Check for duplicates
                            if not any(c["text"] == comment_text for c in comments):
                                comment_obj = {
                                    "platform": "YouTube",
                                    "video_url": video_url,
                                    "text": comment_text,
                                    "author": snippet.get("authorDisplayName", ""),
                                    "timestamp": snippet.get("publishedAt", ""),
                                    "likes": snippet.get("likeCount", 0),
                                    "reply_count": snippet.get("replyCount", 0),
                                    "is_reply": False,
                                    "comment_id": item.get("id", "")
                                }
                                comments.append(comment_obj)
                        
                        new_count = len(comments) - initial_count
                        if new_count == 0:
                            pages_without_new_comments += 1
                        
                        if page_count % 5 == 0:
                            print(f"[INFO] Fetched {len(comments)} top-level comments (page {page_count}, sort: {sort_order})...")
                        
                        # Stop if we've reached max_results for top-level comments
                        if len(comments) >= max_results:
                            print(f"[INFO] Reached max_results limit of {max_results} top-level comments")
                            break
                    
                    # Continue to next page if available
                    if "nextPageToken" in response and pages_without_new_comments < max_pages_without_new:
                        next_token_retry = 0
                        max_token_retries = 3
                        while next_token_retry < max_token_retries:
                            try:
                                time.sleep(0.05)
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
                                    print(f"[WARNING] Pagination retry {next_token_retry}/{max_token_retries}")
                                    time.sleep(wait_time)
                                else:
                                    print(f"[WARNING] Stopping pagination for {sort_order}")
                                    request = None
                                    break
                    else:
                        if pages_without_new_comments >= max_pages_without_new:
                            print(f"[INFO] No new comments for {pages_without_new_comments} pages. Ending pagination.")
                        print(f"[INFO] ✓ Completed {sort_order} pagination. Total top-level: {len(comments)}")
                        break
            
            # Fetch replies for each top-level comment if enabled
            total_replies = 0
            if fetch_replies:
                print(f"\n[INFO] === FETCHING REPLY COMMENTS ===")
                print(f"[INFO] Fetching up to {max_replies_per_comment} replies for {len(comments)} comments...")
                
                for idx, comment in enumerate(comments):
                    if idx % 20 == 0:
                        print(f"[INFO] Processed {idx}/{len(comments)} comments for replies...")
                    
                    reply_count = comment.get("reply_count", 0)
                    if reply_count > 0:
                        replies = self.fetch_replies(
                            comment["comment_id"],
                            video_url,
                            max_replies=max_replies_per_comment
                        )
                        comments.extend(replies)
                        total_replies += len(replies)
                
                print(f"[INFO] Fetched {total_replies} reply comments")
        
        except Exception as e:
            raise Exception(f"Failed to fetch comments: {str(e)}")
        
        print(f"\n[INFO] ========== FINAL RESULT ==========")
        print(f"[INFO] Total comments fetched: {len(comments)}")
        if total_comments_on_video > 0:
            percentage = (len(comments) / total_comments_on_video) * 100
            print(f"[INFO] Video has ~{total_comments_on_video} total comments")
            print(f"[INFO] Coverage: {percentage:.1f}%")
        print(f"[INFO] Top-level: {sum(1 for c in comments if not c.get('is_reply', False))}")
        print(f"[INFO] Replies: {sum(1 for c in comments if c.get('is_reply', False))}")
        print(f"[INFO] ===================================\n")
        
        return comments[:max_results]


class CommentFetcher:
    """Simple interface for fetching YouTube comments."""

    def __init__(self, youtube_api_key: str = None):
        self.youtube_fetcher = None
        if youtube_api_key:
            self.youtube_fetcher = YouTubeCommentFetcher(youtube_api_key)

    def fetch_comments(self, url: str, max_results: int = 10000, fetch_replies: bool = True, 
                      max_replies_per_comment: int = 5) -> List[Dict]:
        """
        Fetch comments from a YouTube URL with reply comments.
        
        Args:
            url: YouTube video URL
            max_results: Maximum total comments to fetch (default 10000 for more comprehensive)
            fetch_replies: Whether to also fetch reply comments (HIGHLY RECOMMENDED - greatly increases coverage)
            max_replies_per_comment: Maximum replies to fetch per comment (default 5)
            
        Returns:
            List of comment dicts with is_reply flag to distinguish replies
        """
        if "youtube.com" in url or "youtu.be" in url:
            if not self.youtube_fetcher:
                raise ValueError("YouTube API key not configured")
            return self.youtube_fetcher.fetch_comments(
                url, 
                max_results=max_results,
                fetch_replies=fetch_replies,
                max_replies_per_comment=max_replies_per_comment
            )
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
