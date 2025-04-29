from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
import re

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    # Handle various URL formats
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:
        query = urlparse(url).query
        return parse_qs(query).get("v", [None])[0]
    return url  # Assume raw ID was passed

def get_transcript(video_input, languages=["en"]):
    """
    Get transcript from YouTube video URL or ID
    Returns formatted transcript text or None if unavailable
    """
    try:
        video_id = extract_video_id(video_input)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return " ".join([chunk["text"] for chunk in transcript_list])
    
    except TranscriptsDisabled:
        print(f"Transcripts disabled for video: {video_id}")
        return None
    except Exception as e:
        print(f"Error getting transcript: {str(e)}")
        return None

# Example usage:
# transcript = get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")