"""
Module for background tracking of operating system active windows using pygetwindow.
Monitors browser window titles and dispatches YouTube video detection events to a thread-safe queue.
Gracefully handles platforms without window server support (such as Linux cloud containers).
"""

import time
import threading
import queue
from typing import Optional, Dict, Any

# Safe import for pygetwindow (avoids NotImplementedError on Linux cloud containers)
try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except Exception:
    gw = None
    HAS_PYGETWINDOW = False

from youtube_engine import extract_video_id_from_text

# Global thread-safe queue for UI event ingestion
tracking_event_queue: queue.Queue = queue.Queue(maxsize=50)

BROWSER_SUFFIXES = [
    "- YouTube",
    "- Google Chrome",
    "- Microsoft Edge",
    "- Brave",
    "- Mozilla Firefox"
]

def clean_window_title(raw_title: str) -> str:
    """Strips common browser name and site suffixes from a window title."""
    title = raw_title
    for suffix in BROWSER_SUFFIXES:
        title = title.replace(suffix, "").strip()
    return title

class OSWindowTracker:
    """
    Background worker that polls active OS window titles and identifies potential distraction events.
    """

    def __init__(self, interval_sec: float = 2.0):
        """
        Initializes the OS window tracker.

        Args:
            interval_sec (float): Polling frequency in seconds.
        """
        self.interval_sec = interval_sec
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.last_detected_title = "Active (Desktop)" if HAS_PYGETWINDOW else "Cloud Environment (Desktop Tracking Inactive)"
        self.last_checked_key = ""
        self.is_supported = HAS_PYGETWINDOW

    def _enqueue_event(self, payload: Dict[str, Any]) -> None:
        """Pushes an event payload into the thread-safe queue."""
        try:
            tracking_event_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _process_youtube_title(self, raw_title: str) -> None:
        """Parses YouTube session identifiers and dispatches change events."""
        video_id = extract_video_id_from_text(raw_title)
        cleaned_title = clean_window_title(raw_title)
        unique_key = video_id if video_id else cleaned_title

        if not unique_key or unique_key == self.last_checked_key:
            return

        self.last_checked_key = unique_key
        event_payload: Dict[str, Any] = {
            "timestamp": time.time(),
            "raw_title": raw_title,
            "cleaned_title": cleaned_title,
            "video_id": video_id,
            "type": "youtube_active"
        }
        self._enqueue_event(event_payload)

    def _check_active_window(self) -> None:
        """Polls the OS for the active window and triggers processing."""
        if not HAS_PYGETWINDOW or gw is None:
            return

        try:
            active_window = gw.getActiveWindow()
            if not active_window or not active_window.title:
                return

            title = active_window.title.strip()
            self.last_detected_title = title

            if "youtube" in title.lower():
                self._process_youtube_title(title)
        except Exception:
            pass

    def _track_loop(self) -> None:
        """Continuous execution loop running on a background daemon thread."""
        while self.is_running:
            self._check_active_window()
            time.sleep(self.interval_sec)

    def start(self) -> None:
        """Starts background tracking thread if supported on the current platform."""
        if not HAS_PYGETWINDOW:
            return

        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._track_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stops background tracking thread."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

# Singleton tracker instance
tracker_instance = OSWindowTracker(interval_sec=2.0)
