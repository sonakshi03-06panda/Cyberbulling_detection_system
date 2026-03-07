"""
Fetch comments from YouTube.
Requires: google-api-python-client
"""

"""
YouTube Comment Fetcher
Optimized for large videos with parallel reply fetching.
"""

import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


class YouTubeCommentFetcher:

    def __init__(self, api_key: str):
        from googleapiclient.discovery import build
        self.youtube = build("youtube", "v3", developerKey=api_key)

    # ---------------------------------------------
    # Extract video ID
    # ---------------------------------------------

    def get_video_id_from_url(self, url: str) -> str:

        if "youtube.com/watch?v=" in url:
            return url.split("v=")[1].split("&")[0]

        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]

        raise ValueError("Invalid YouTube URL")

    # ---------------------------------------------
    # Fetch replies for a single comment
    # ---------------------------------------------

    def fetch_replies(self, parent_id: str, video_url: str, max_replies=100):

        replies = []
        next_page = None

        while True:

            try:
                response = self.youtube.comments().list(
                    part="snippet",
                    parentId=parent_id,
                    maxResults=100,
                    pageToken=next_page,
                    textFormat="plainText"
                ).execute()

            except Exception as e:
                error_msg = str(e)
                
                # Log error but don't crash - return what we have
                if "400" in error_msg or "processingFailure" in error_msg:
                    print(f"[WARNING] Reply fetch failed for parent {parent_id}: {error_msg[:80]}")
                    return replies
                
                elif "429" in error_msg or "quota" in error_msg.lower():
                    print(f"[WARNING] Reply fetch rate limited - returning partial replies")
                    return replies
                
                else:
                    print(f"[WARNING] Unexpected error fetching replies: {error_msg[:80]}")
                    return replies

            for item in response.get("items", []):

                snippet = item["snippet"]

                replies.append({
                    "platform": "YouTube",
                    "video_url": video_url,
                    "text": snippet.get("textDisplay", ""),
                    "author": snippet.get("authorDisplayName", ""),
                    "timestamp": snippet.get("publishedAt", ""),
                    "likes": snippet.get("likeCount", 0),
                    "is_reply": True,
                    "parent_id": parent_id
                })

                if len(replies) >= max_replies:
                    return replies

            next_page = response.get("nextPageToken")

            if not next_page:
                break

            time.sleep(0.05)

        return replies

    # ---------------------------------------------
    # Parallel reply fetching
    # ---------------------------------------------

    def fetch_replies_parallel(
        self,
        parent_ids: List[str],
        video_url: str,
        max_replies=100,
        workers=10
    ):

        all_replies = []

        with ThreadPoolExecutor(max_workers=workers) as executor:

            futures = {
                executor.submit(self.fetch_replies, pid, video_url, max_replies): pid
                for pid in parent_ids
            }

            for future in as_completed(futures):

                try:
                    replies = future.result()
                    all_replies.extend(replies)

                except Exception as e:
                    print(f"[WARNING] Reply fetch failed: {e}")

        return all_replies

    # ---------------------------------------------
    # Fetch all comments
    # ---------------------------------------------

    def fetch_comments(
        self,
        video_url: str,
        max_results=20000,
        fetch_replies=True,
        max_replies_per_comment=100
    ):

        video_id = self.get_video_id_from_url(video_url)

        comments: List[Dict] = []
        seen_text = set()

        next_page = None
        page_count = 0

        print("\n[INFO] Fetching YouTube comments...")
        print(f"[INFO] Video ID: {video_id}")

        while True:

            try:
                print(f"[INFO] Fetching page {page_count + 1}...")
                response = self.youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page,
                    textFormat="plainText"
                ).execute()
                
                print(f"[INFO] Response received with {len(response.get('items', []))} items")

            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] API Exception: {error_msg}")
                
                # Handle pagination errors gracefully
                if "400" in error_msg or "processingFailure" in error_msg:
                    print(f"[WARNING] Pagination error on page {page_count + 1}: {error_msg[:100]}")
                    print(f"[INFO] Stopping pagination - returning {len(comments)} comments collected so far")
                    break
                
                # Handle rate limiting with backoff
                elif "429" in error_msg or "quota" in error_msg.lower():
                    print(f"[WARNING] Rate limit hit - waiting 60 seconds before retry")
                    time.sleep(60)
                    continue
                
                # Unknown error
                else:
                    print(f"[ERROR] Unexpected API error: {error_msg}")
                    raise

            page_count += 1
            reply_parent_ids = []

            for item in response.get("items", []):

                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet.get("textDisplay", "")

                if text not in seen_text:

                    seen_text.add(text)

                    comments.append({
                        "platform": "YouTube",
                        "video_url": video_url,
                        "text": text,
                        "author": snippet.get("authorDisplayName", ""),
                        "timestamp": snippet.get("publishedAt", ""),
                        "likes": snippet.get("likeCount", 0),
                        "reply_count": snippet.get("replyCount", 0),
                        "is_reply": False,
                        "comment_id": item.get("id")
                    })

                # -------------------------
                # Embedded replies
                # -------------------------

                if fetch_replies and "replies" in item:

                    embedded_replies = item["replies"]["comments"]

                    for reply in embedded_replies:

                        r = reply["snippet"]
                        r_text = r.get("textDisplay", "")

                        if r_text not in seen_text:

                            seen_text.add(r_text)

                            comments.append({
                                "platform": "YouTube",
                                "video_url": video_url,
                                "text": r_text,
                                "author": r.get("authorDisplayName", ""),
                                "timestamp": r.get("publishedAt", ""),
                                "likes": r.get("likeCount", 0),
                                "is_reply": True,
                                "parent_id": item.get("id")
                            })

                # -------------------------
                # Determine if extra replies needed
                # -------------------------

                if fetch_replies:

                    total_replies = snippet.get("replyCount", 0)
                    embedded_count = len(item.get("replies", {}).get("comments", []))

                    if total_replies > embedded_count:
                        reply_parent_ids.append(item.get("id"))

            # -------------------------
            # Parallel fetch extra replies
            # -------------------------

            if fetch_replies and reply_parent_ids:

                extra_replies = self.fetch_replies_parallel(
                    reply_parent_ids,
                    video_url,
                    max_replies=max_replies_per_comment,
                    workers=10
                )

                for reply in extra_replies:

                    if reply["text"] not in seen_text:
                        seen_text.add(reply["text"])
                        comments.append(reply)

            if page_count % 5 == 0:
                print(f"[INFO] Pages processed: {page_count} | Comments: {len(comments)}")

            if len(comments) >= max_results:
                break

            next_page = response.get("nextPageToken")

            if not next_page:
                break

            time.sleep(0.05)

        print("\n[INFO] ===== Fetch Summary =====")
        print(f"[INFO] Total comments: {len(comments)}")
        print(f"[INFO] Top-level: {sum(1 for c in comments if not c['is_reply'])}")
        print(f"[INFO] Replies: {sum(1 for c in comments if c['is_reply'])}")
        print("[INFO] ==========================\n")

        return comments[:max_results]


# ------------------------------------------------
# Wrapper class
# ------------------------------------------------

class CommentFetcher:

    def __init__(self, youtube_api_key=None):

        self.youtube_fetcher = None

        if youtube_api_key:
            self.youtube_fetcher = YouTubeCommentFetcher(youtube_api_key)

    def fetch_comments(
        self,
        url,
        max_results=20000,
        fetch_replies=True,
        max_replies_per_comment=100
    ):

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
            raise ValueError("Unsupported platform")

if __name__ == "__main__":
    """
    Run this file directly to test comment fetching.
    Uses YOUTUBE_API_KEY from your .env file.
    """

    import os
    from dotenv import load_dotenv

    # Load .env file
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)

    # Get API key
    yt_key = os.getenv("YOUTUBE_API_KEY")

    if not yt_key:
        raise ValueError("YOUTUBE_API_KEY not found in .env file")

    # Initialize fetcher
    fetcher = CommentFetcher(yt_key)

    # Test video
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # replace with your test video

    print("\nFetching comments...\n")

    comments = fetcher.fetch_comments(
        video_url,
        max_results=2000,
        fetch_replies=True
    )

    print(f"\nFetched {len(comments)} comments\n")

    # Show sample
    for c in comments[:5]:
        print(f"{c['author']}: {c['text'][:80]}")