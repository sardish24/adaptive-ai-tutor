"""
Module for extracting YouTube transcripts and video metadata for relevance validation.
"""

import os
import re
from typing import Optional
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

load_dotenv()

def extract_video_id_from_text(text: str) -> Optional[str]:
    """
    Extracts an 11-character YouTube video ID from a URL or window title string.

    Args:
        text (str): Input window title or URL string.

    Returns:
        Optional[str]: Extracted YouTube video ID if found, otherwise None.
    """
    if not text:
        return None

    url_patterns = [
        r'(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/watch\?v=|\&v=)([a-zA-Z0-9_-]{11})',
        r'watch\?v=([a-zA-Z0-9_-]{11})'
    ]
    for pattern in url_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Fetches and concatenates English or auto-generated transcript text for a given YouTube video ID.

    Args:
        video_id (str): Unique 11-character YouTube video identifier.

    Returns:
        Optional[str]: Concatenated transcript text, or None if captions are unavailable.
    """
    if not video_id:
        return None

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        text_parts = [item.get('text', '') for item in transcript_list if item.get('text')]
        full_transcript = " ".join(text_parts).strip()
        return full_transcript if full_transcript else None
    except Exception:
        try:
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list_obj:
                data = transcript.fetch()
                text_parts = [item.get('text', '') for item in data if item.get('text')]
                full_transcript = " ".join(text_parts).strip()
                if full_transcript:
                    return full_transcript
        except Exception:
            return None
    return None

def fetch_video_title_metadata(video_id: str) -> Optional[str]:
    """
    Retrieves video title metadata using YouTube Data API v3 if key is configured.

    Args:
        video_id (str): Unique 11-character YouTube video identifier.

    Returns:
        Optional[str]: Video title if successfully resolved, otherwise None.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key or api_key == "your_key_here":
        return None

    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        items = response.get("items", [])
        if items:
            return items[0]["snippet"].get("title", "")
    except Exception:
        return None
    return None
