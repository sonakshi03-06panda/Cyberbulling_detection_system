"""
Fetch comments from YouTube.
Requires: google-api-python-client
"""

import os
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
        """
        video_id = self.get_video_id_from_url(video_url)
        comments = []
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results),
                textFormat="plainText",
                order="relevance"
            )
            
            while request and len(comments) < max_results:
                try:
                    response = request.execute()
                except Exception as api_error:
                    error_msg = str(api_error)
                    if "403" in error_msg:
                        raise Exception("YouTube API quota exceeded or access denied. Try again later or check your API key.")
                    elif "404" in error_msg:
                        raise Exception(f"Video not found. Check if the URL is correct: {video_url}")
                    elif "commentsDisabled" in error_msg or "disabled" in error_msg.lower():
                        raise Exception("Comments are disabled on this video.")
                    else:
                        raise Exception(f"YouTube API error: {error_msg}")
                
                items = response.get("items", [])
                if not items and len(comments) == 0:
                    # No items on first page - likely comments disabled or API issue
                    raise Exception("No comments found. The video may have comments disabled or is unavailable.")
                
                for item in items:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    # some videos may not include likeCount or replyCount in snippet
                    comments.append({
                        "platform": "YouTube",
                        "video_url": video_url,
                        "text": snippet.get("textDisplay", ""),
                        "author": snippet.get("authorDisplayName", ""),
                        "timestamp": snippet.get("publishedAt", ""),
                        "likes": snippet.get("likeCount", 0),
                        "replies": snippet.get("replyCount", 0)
                    })
                
                if "nextPageToken" in response and len(comments) < max_results:
                    request = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        pageToken=response["nextPageToken"],
                        maxResults=min(100, max_results - len(comments)),
                        textFormat="plainText"
                    )
                else:
                    break
        except Exception as e:
            # Re-raise with context
            raise Exception(f"Failed to fetch comments: {str(e)}")
        
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
    
    load_dotenv()
    
    yt_key = os.getenv("YOUTUBE_API_KEY")
    fetcher = CommentFetcher(yt_key)
    
    # Example
    # comments = fetcher.fetch_comments("https://www.youtube.com/watch?v=...")
    # print(f"Fetched {len(comments)} comments")
