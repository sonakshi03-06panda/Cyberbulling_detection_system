import os
from scr.comment_fetcher import YouTubeCommentFetcher
key = os.getenv('YOUTUBE_API_KEY')
print('KEY', key)
fetcher = YouTubeCommentFetcher(key)
comments = fetcher.fetch_comments('https://www.youtube.com/watch?v=dQw4w9WgXcQ', max_results=3)
print('COUNT', len(comments))
print(comments[:1])
