"""
Module for background session telemetry logging and database querying using SQLite.
Records student cognitive state metrics at periodic intervals and derives engagement scores.
"""

import os
import time
import sqlite3
import threading
from typing import List, Dict, Any, Optional
import pandas as pd
from constants import STATE_SCORE_MAPPING, STATE_FOCUSED

DB_PATH = os.path.join(os.path.dirname(__file__), "session_telemetry.db")

def init_db() -> None:
    """Initializes the SQLite telemetry database schema."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                datetime_str TEXT NOT NULL,
                cognitive_state TEXT NOT NULL,
                confidence REAL NOT NULL,
                focus_score REAL NOT NULL,
                is_spoof_flag INTEGER NOT NULL,
                yaw_ratio REAL,
                pitch_ratio REAL,
                roll_deg REAL
            )
            """
        )
        conn.commit()

class AnalyticsEngine:
    """
    Manages background logging of real-time telemetry into SQLite
    and provides data aggregation utilities for dashboard rendering.
    """

    def __init__(self, log_interval_sec: float = 5.0):
        self.log_interval_sec = log_interval_sec
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        init_db()

    def _logging_loop(self, state_holder_ref: Any):
        while self.is_running:
            try:
                now = time.time()
                dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                state = getattr(state_holder_ref, "current_state", STATE_FOCUSED)
                confidence = float(getattr(state_holder_ref, "confidence", 90.0))
                metrics = getattr(state_holder_ref, "last_metrics", {})
                is_spoof = 1 if getattr(state_holder_ref, "is_spoof_detected", False) else 0
                
                score = STATE_SCORE_MAPPING.get(state, 0.5)

                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO telemetry_logs (
                            timestamp, datetime_str, cognitive_state, confidence,
                            focus_score, is_spoof_flag, yaw_ratio, pitch_ratio, roll_deg
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            now,
                            dt_str,
                            state,
                            confidence,
                            score,
                            is_spoof,
                            metrics.get("yaw_ratio", 0.0),
                            metrics.get("pitch_ratio", 0.0),
                            metrics.get("roll_deg", 0.0),
                        ),
                    )
                    conn.commit()
            except Exception:
                pass

            time.sleep(self.log_interval_sec)

    def start(self, state_holder_ref: Any) -> None:
        """Starts background periodic logging thread."""
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(
                target=self._logging_loop,
                args=(state_holder_ref,),
                daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stops background logging thread."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    @staticmethod
    def get_session_dataframe(limit: int = 300) -> pd.DataFrame:
        """
        Retrieves recent telemetry records as a pandas DataFrame for visual plotting.

        Args:
            limit (int): Maximum number of recent entries to return.

        Returns:
            pd.DataFrame: Formatted telemetry records.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                query = f"""
                SELECT datetime_str, cognitive_state, confidence, focus_score, is_spoof_flag
                FROM telemetry_logs
                ORDER BY id DESC
                LIMIT {limit}
                """
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    df = df.iloc[::-1].reset_index(drop=True)
                return df
        except Exception:
            return pd.DataFrame()

analytics_instance = AnalyticsEngine(log_interval_sec=5.0)
