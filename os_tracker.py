"""
Module for background tracking of operating system active windows using pygetwindow.
Monitors browser window titles and dispatches YouTube video detection events to a thread-safe queue.
"""

import time
import threading
import queue
from typing import Optional, Dict, Any
import pygetwindow as gw
from youtube_engine import extract_video_id_from_text

# Global thread-safe queue for UI event ingestion
tracking_event_queue: queue.Queue = queue.Queue(maxsize=50)

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
        self.last_detected_title = ""
        self.last_checked_key = ""

    def _track_loop(self):
        while self.is_running:
            try:
                active_window = gw.getActiveWindow()
                if active_window and active_window.title:
                    title = active_window.title.strip()
                    self.last_detected_title = title

                    # Identify active YouTube session in window title
                    if "youtube" in title.lower():
                        video_id = extract_video_id_from_text(title)
                        
                        # Strip standard browser suffix metadata
                        cleaned_title = title
                        for suffix in ["- YouTube", "- Google Chrome", "- Microsoft Edge", "- Brave", "- Mozilla Firefox"]:
                            cleaned_title = cleaned_title.replace(suffix, "").strip()

                        unique_key = video_id if video_id else cleaned_title

                        if unique_key and unique_key != self.last_checked_key:
                            self.last_checked_key = unique_key
                            event_payload: Dict[str, Any] = {
                                "timestamp": time.time(),
                                "raw_title": title,
                                "cleaned_title": cleaned_title,
                                "video_id": video_id,
                                "type": "youtube_active"
                            }
                            try:
                                tracking_event_queue.put_nowait(event_payload)
                            except queue.Full:
                                pass
            except Exception:
                pass

            time.sleep(self.interval_sec)

    def start(self) -> None:
        """Starts background tracking thread."""
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
