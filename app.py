"""
Adaptive AI Tutor Application
Integrates real-time webcam facial orientation, gaze metrics, anti-spoofing liveness detection,
OS-level window distraction monitoring, curriculum RAG indexing, and session telemetry analytics.
"""

import os
import time
import queue
from typing import Tuple, List, Dict, Any, Optional
import av
import cv2
import numpy as np
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from constants import (
    STATE_FOCUSED,
    STATE_CONFUSED,
    STATE_DISTRACTED,
    STATE_DROWSY,
)
from rag_engine import RAGEngine
from os_tracker import tracker_instance, tracking_event_queue
from youtube_engine import fetch_youtube_transcript, fetch_video_title_metadata
from analytics_engine import analytics_instance, AnalyticsEngine

# --- Environment & API Configuration ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.error("GEMINI_API_KEY not located. Please define GEMINI_API_KEY in the environment or secrets configuration.")
    st.stop()

genai.configure(api_key=api_key)

# --- Global Shared State ---
class StateHolder:
    """Thread-safe container holding real-time biometric and telemetry metrics."""
    current_state: str = STATE_FOCUSED
    confidence: float = 90.0
    last_metrics: Dict[str, float] = {
        "yaw_ratio": 0.0,
        "pitch_ratio": 0.5,
        "roll_deg": 0.0,
        "mouth_aspect": 0.5,
        "ear_value": 0.30
    }
    distracted_start_ts: float = 0.0
    is_distracted_sustained: bool = False
    is_spoof_detected: bool = False
    enable_xai_heatmap: bool = False
    last_update_ts: float = time.time()

# Initialize background services
tracker_instance.start()
analytics_instance.start(StateHolder)

# --- OpenCV Face Detector Setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet.onnx")
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def ensure_model_file() -> None:
    """Verifies that the ONNX face detection model exists locally, downloading if necessary."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

ensure_model_file()

def classify_cognitive_state(
    pitch_ratio: float, yaw_ratio: float, roll_deg: float, face_score: float
) -> Tuple[str, float, Tuple[int, int, int]]:
    """
    Classifies facial orientation and head pose metrics into defined cognitive states.

    Args:
        pitch_ratio (float): Vertical gaze displacement ratio.
        yaw_ratio (float): Horizontal gaze displacement ratio.
        roll_deg (float): Head tilt angle in degrees.
        face_score (float): Detection confidence score from face detector.

    Returns:
        Tuple[str, float, Tuple[int, int, int]]: Cognitive state name, confidence %, and RGB color tuple.
    """
    if pitch_ratio > 0.85:
        state = STATE_DROWSY
        conf = min(98.0, 75.0 + (pitch_ratio - 0.85) * 50)
        color = (0, 0, 255)
    elif abs(yaw_ratio) > 0.28:
        state = STATE_DISTRACTED
        conf = min(95.0, 70.0 + abs(yaw_ratio) * 60)
        color = (0, 165, 255)
    elif abs(roll_deg) > 11.0 or (0.28 <= pitch_ratio <= 0.40):
        state = STATE_CONFUSED
        conf = min(92.0, 68.0 + abs(roll_deg) * 2.0)
        color = (0, 255, 255)
    else:
        state = STATE_FOCUSED
        conf = max(80.0, face_score * 100)
        color = (0, 255, 0)
    return state, conf, color

def draw_face_annotations(img_bgr: np.ndarray, box: np.ndarray, landmarks: np.ndarray) -> None:
    """Renders bounding box and facial landmark coordinates on the frame."""
    x, y, bw, bh = box
    cv2.rectangle(img_bgr, (x, y), (x + bw, y + bh), (0, 255, 128), 2)
    for pt in landmarks:
        cv2.circle(img_bgr, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
    cv2.line(img_bgr, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[1][0]), int(landmarks[1][1])), (255, 200, 0), 1)
    cv2.line(img_bgr, (int(landmarks[3][0]), int(landmarks[3][1])), (int(landmarks[4][0]), int(landmarks[4][1])), (255, 200, 0), 1)

class VideoTransformer(VideoProcessorBase):
    """
    Processes WebRTC incoming video frames, extracts pose and landmark features,
    executes anti-spoofing liveness checks, computes optical flow XAI heatmaps,
    and determines cognitive states.
    """

    def __init__(self):
        self.detector = None
        self.current_size = None
        self.state_history: List[Tuple[str, float]] = []
        self.history_len = 15
        
        # Anti-spoofing rolling buffer (timestamps and metric samples)
        self.ear_history: List[Tuple[float, float]] = []
        self.liveness_window_sec: float = 60.0
        self.min_samples_for_liveness: int = 150

        # Optical Flow XAI tracking
        self.prev_gray: Optional[np.ndarray] = None

    def _get_detector(self, width: int, height: int):
        if self.detector is None or self.current_size != (width, height):
            self.detector = cv2.FaceDetectorYN.create(
                MODEL_PATH, "", (width, height), score_threshold=0.6, nms_threshold=0.3
            )
            self.current_size = (width, height)
        return self.detector

    def _apply_optical_flow_xai(self, img_bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Calculates Farneback dense optical flow and overlays a motion magnitude heatmap."""
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return img_bgr

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            blurred_mag = cv2.GaussianBlur(magnitude, (15, 15), 0)
            norm_mag = cv2.normalize(blurred_mag, None, 0, 255, cv2.NORM_MINMAX)
            norm_mag_uint8 = norm_mag.astype(np.uint8)

            heatmap = cv2.applyColorMap(norm_mag_uint8, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(img_bgr, 0.65, heatmap, 0.35, 0)
            self.prev_gray = gray
            return blended
        except Exception:
            self.prev_gray = gray
            return img_bgr

    def _evaluate_liveness(self, now: float, estimated_ear: float) -> None:
        """Evaluates Eye Aspect Ratio variance to detect static image presentation attacks."""
        self.ear_history.append((now, estimated_ear))
        self.ear_history = [
            (ts, val) for ts, val in self.ear_history if now - ts <= self.liveness_window_sec
        ]

        if len(self.ear_history) < self.min_samples_for_liveness:
            StateHolder.is_spoof_detected = False
            return

        ear_values = [val for _, val in self.ear_history]
        ear_variance = float(np.var(ear_values))
        StateHolder.is_spoof_detected = bool(ear_variance < 0.00005)

    def _smooth_cognitive_state(self, inst_state: str, inst_conf: float) -> Tuple[str, float]:
        """Applies rolling window majority voting to smooth instantaneous state classifications."""
        self.state_history.append((inst_state, inst_conf))
        if len(self.state_history) > self.history_len:
            self.state_history.pop(0)

        state_counts: Dict[str, int] = {}
        for st_name, _ in self.state_history:
            state_counts[st_name] = state_counts.get(st_name, 0) + 1

        smooth_state = max(state_counts, key=state_counts.get)
        smooth_conf = float(np.mean([c for st_name, c in self.state_history if st_name == smooth_state]))
        return smooth_state, smooth_conf

    def _update_distraction_tracker(self, smooth_state: str, now: float) -> None:
        """Tracks duration of sustained distraction (> 10 seconds)."""
        if smooth_state == STATE_DISTRACTED:
            if StateHolder.distracted_start_ts <= 1e-6:
                StateHolder.distracted_start_ts = now
            elif now - StateHolder.distracted_start_ts >= 10.0:
                StateHolder.is_distracted_sustained = True
        else:
            StateHolder.distracted_start_ts = 0.0
            StateHolder.is_distracted_sustained = False

    def _process_face_metrics(self, face: np.ndarray, img_bgr: np.ndarray, now: float) -> Tuple[int, int, int]:
        """Calculates pose ratios, runs liveness and classification checks, and updates shared state."""
        box = face[0:4].astype(int)
        face_score = float(face[14]) if len(face) > 14 else 0.9
        landmarks = face[4:14].reshape((5, 2)).astype(float)

        draw_face_annotations(img_bgr, box, landmarks)

        r_eye, l_eye, nose, r_mouth, l_mouth = landmarks
        eye_dist = np.linalg.norm(l_eye - r_eye) + 1e-5
        eye_center = (r_eye + l_eye) / 2.0

        yaw_ratio = (nose[0] - eye_center[0]) / eye_dist
        pitch_ratio = (nose[1] - eye_center[1]) / eye_dist
        roll_deg = float(np.degrees(np.arctan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0])))
        mouth_dist = np.linalg.norm(l_mouth - r_mouth)
        estimated_ear = float(np.abs(eye_center[1] - nose[1]) / eye_dist)

        self._evaluate_liveness(now, estimated_ear)

        inst_state, inst_conf, color = classify_cognitive_state(pitch_ratio, yaw_ratio, roll_deg, face_score)
        smooth_state, smooth_conf = self._smooth_cognitive_state(inst_state, inst_conf)
        self._update_distraction_tracker(smooth_state, now)

        StateHolder.current_state = smooth_state
        StateHolder.confidence = smooth_conf
        StateHolder.last_metrics = {
            "yaw_ratio": round(float(yaw_ratio), 2),
            "pitch_ratio": round(float(pitch_ratio), 2),
            "roll_deg": round(float(roll_deg), 1),
            "mouth_aspect": round(float(mouth_dist / eye_dist), 2),
            "ear_value": round(estimated_ear, 3)
        }
        StateHolder.last_update_ts = now
        return color

    def _render_state_badge(self, img_bgr: np.ndarray, width: int, status_color: Tuple[int, int, int]) -> None:
        """Renders live cognitive state badge on the top of the video frame."""
        cv2.rectangle(img_bgr, (10, 10), (width - 10, 50), (20, 20, 20), -1)
        if StateHolder.is_spoof_detected:
            state_label = "SPOOF DETECTED"
            badge_color = (0, 0, 255)
        else:
            state_label = f"State: {StateHolder.current_state} ({int(StateHolder.confidence)}%)"
            badge_color = status_color

        cv2.putText(
            img_bgr,
            state_label,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            badge_color,
            2,
            cv2.LINE_AA
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        height, width, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if StateHolder.enable_xai_heatmap:
            img_bgr = self._apply_optical_flow_xai(img_bgr, gray)
        else:
            self.prev_gray = gray

        detector = self._get_detector(width, height)
        _, faces = detector.detect(img_bgr)
        status_color = (128, 128, 128)
        now = time.time()

        if faces is not None and len(faces) > 0:
            status_color = self._process_face_metrics(faces[0], img_bgr, now)

        self._render_state_badge(img_bgr, width, status_color)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")



# --- Streamlit Layout Configuration ---
st.set_page_config(page_title="Adaptive AI Tutor", layout="wide")
st.title("Adaptive AI Tutor with RAG, Proctoring & Telemetry Analytics")
st.caption("Real-time biometric cognitive state detection feeding dynamically into an adaptive, curriculum-grounded Gemini tutor.")

# --- Initialize RAG Engine ---
@st.cache_resource
def get_rag_engine() -> RAGEngine:
    rag = RAGEngine()
    rag.ingest_study_materials()
    return rag

rag_engine = get_rag_engine()

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
    ]
}

# Session State Initialization
if "youtube_alerts" not in st.session_state:
    st.session_state.youtube_alerts = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Drain OS Window Event Queue
while not tracking_event_queue.empty():
    try:
        event = tracking_event_queue.get_nowait()
        video_id = event.get("video_id")
        title = event.get("cleaned_title", "")

        query_text = ""
        if video_id:
            transcript = fetch_youtube_transcript(video_id)
            if transcript:
                query_text = transcript[:1500]
            else:
                meta_title = fetch_video_title_metadata(video_id)
                query_text = meta_title if meta_title else title
        else:
            query_text = title

        if query_text:
            match_context = rag_engine.retrieve_relevant_context(query_text, top_k=1, min_similarity=0.62)
            if not match_context:
                st.session_state.youtube_alerts.append({
                    "title": title,
                    "video_id": video_id,
                    "status": "irrelevant",
                    "timestamp": time.strftime("%H:%M:%S")
                })
            else:
                st.session_state.youtube_alerts.append({
                    "title": title,
                    "video_id": video_id,
                    "status": "relevant",
                    "timestamp": time.strftime("%H:%M:%S")
                })
    except queue.Empty:
        break

# --- Sidebar: Curriculum & System Status ---
with st.sidebar:
    st.header("Curriculum & Knowledge Base")
    st.write("Documents in `./study_materials` are indexed for grounded tutoring.")
    
    study_files = os.listdir(os.path.join(os.path.dirname(__file__), "study_materials"))
    if study_files:
        st.markdown("**Indexed Documents:**")
        for file_name in study_files:
            st.markdown(f"- `{file_name}`")
    else:
        st.info("No documents detected. Add `.pdf`, `.txt`, or `.md` files to `./study_materials`.")

    if st.button("Re-Index Knowledge Base"):
        with st.spinner("Indexing curriculum documents..."):
            count = rag_engine.ingest_study_materials()
            st.success(f"Indexed {count} text chunks into local vector store.")

    st.markdown("---")
    st.header("Explainable AI (XAI)")
    enable_xai = st.checkbox("Enable XAI Heatmap Overlay", value=False)
    StateHolder.enable_xai_heatmap = enable_xai
    if enable_xai:
        st.caption("Visualizing dense optical flow motion magnitudes across facial regions.")

    st.markdown("---")
    st.header("Proctoring & Liveness Status")
    if StateHolder.is_spoof_detected:
        st.error("Anti-Spoofing: Static presentation attack detected.")
    elif StateHolder.is_distracted_sustained:
        st.error("Proctor Alert: Off-screen gaze detected for > 10 seconds.")
    elif StateHolder.current_state == STATE_DROWSY:
        st.warning("Fatigue Notice: Prolonged eye closure or head drooping detected.")
    else:
        st.success("Webcam Proctoring: Verified active.")

    st.markdown("---")
    st.header("Active Window Telemetry")
    active_title = tracker_instance.last_detected_title or "Monitoring..."
    st.caption(f"**Focused Window:** {active_title[:60]}...")
    
    if st.session_state.youtube_alerts:
        latest_alert = st.session_state.youtube_alerts[-1]
        if latest_alert["status"] == "irrelevant":
            st.error(f"Distraction Event ({latest_alert['timestamp']}): Irrelevant media `{latest_alert['title'][:35]}...`")
        else:
            st.info(f"Curriculum Video ({latest_alert['timestamp']}): Relevant media `{latest_alert['title'][:35]}...`")

# --- Notification Banners ---
if StateHolder.is_spoof_detected:
    st.error("Liveness Check Failed: Static image detected. Chat input is paused until genuine liveness is verified.")
elif StateHolder.is_distracted_sustained:
    st.toast("Proctor Alert: Refocus on study material.", icon=None)
    st.warning("Proctoring Notice: Sustained distraction detected (> 10s off-screen). Refocus on study material.")
elif StateHolder.current_state == STATE_DROWSY:
    st.warning("Proctoring Notice: Fatigue detected. Consider taking a brief rest interval.")

if st.session_state.youtube_alerts:
    latest_alert = st.session_state.youtube_alerts[-1]
    if latest_alert["status"] == "irrelevant":
        st.error(f"Distraction Alert: The media stream opened ({latest_alert['title']}) is unrelated to the active curriculum.")

# --- Primary Tabbed Interface ---
tab_chat, tab_analytics = st.tabs(["Adaptive Tutoring Session", "Session Analytics & Telemetry"])

with tab_chat:
    col_chat, col_webcam = st.columns([3, 2], gap="large")

    with col_webcam:
        st.subheader("Real-Time Biometric Stream")
        webrtc_streamer(
            key="adaptive-webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("##### Cognitive & Proctoring Telemetry")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("State Classification", StateHolder.current_state)
            st.metric("Head Roll (Tilt)", f"{StateHolder.last_metrics['roll_deg']}°")
        with metric_col2:
            st.metric("State Confidence", f"{int(StateHolder.confidence)}%")
            st.metric("Gaze Yaw Ratio", f"{StateHolder.last_metrics['yaw_ratio']}")

    with col_chat:
        st.subheader("Curriculum-Grounded Adaptive Chat")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "context" in message and message["context"]:
                    with st.expander("Grounded Curriculum Context"):
                        st.caption(message["context"])

        if StateHolder.is_spoof_detected:
            st.info("Chat is temporarily paused due to an unverified liveness state.")
            user_input = None
        else:
            user_input = st.chat_input("Ask a question regarding your study materials...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # RAG Context Retrieval
            retrieved_context = rag_engine.retrieve_relevant_context(user_input, top_k=2)

            # Adaptive Pedagogical Directives
            current_state = StateHolder.current_state
            
            if current_state == STATE_CONFUSED:
                adaptive_instruction = (
                    "The student's biometric analysis indicates confusion or elevated cognitive load. "
                    "Deconstruct the concept using intuitive analogies, straightforward phrasing, and step-by-step clarity. "
                    "Conclude with a targeted verification question."
                )
            elif current_state == STATE_DISTRACTED:
                adaptive_instruction = (
                    "The student's biometric analysis indicates distraction or off-screen gaze. "
                    "Keep the response concise and engaging (under three paragraphs), concluding with a direct question."
                )
            elif current_state == STATE_DROWSY:
                adaptive_instruction = (
                    "The student exhibits indicators of fatigue. "
                    "Provide a direct summary in structured bullet points and recommend a brief break or knowledge check."
                )
            else:
                adaptive_instruction = (
                    "The student is focused and attentive. "
                    "Provide a comprehensive, high-depth technical explanation."
                )

            rag_section = (
                f"\n\n--- OFFICIAL CURRICULUM CONTEXT ---\n{retrieved_context}\n--- END CONTEXT ---\n"
                "Ground your answer in the official curriculum context when relevant. "
                "Cite the source material if applicable."
                if retrieved_context else
                "\nNo direct study material context found. Answer using authoritative technical principles."
            )

            full_system_prompt = (
                "You are an expert, empathetic, and adaptive technical tutor grounded in official study materials.\n\n"
                f"STUDENT BIOMETRIC STATE: {current_state} (Confidence: {int(StateHolder.confidence)}%)\n"
                f"PEDAGOGICAL DIRECTIVE: {adaptive_instruction}\n"
                f"{rag_section}"
            )

            with st.chat_message("assistant"):
                with st.spinner(f"Adapting pedagogy to student state ({current_state})..."):
                    try:
                        adaptive_model = genai.GenerativeModel(
                            model_name="gemini-3.6-flash",
                            system_instruction=full_system_prompt
                        )
                        
                        chat_history = [
                            {"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]}
                            for msg in st.session_state.messages[:-1]
                        ]
                        chat_session = adaptive_model.start_chat(history=chat_history)
                        response = chat_session.send_message(user_input)
                        st.markdown(response.text)
                        
                        if retrieved_context:
                            with st.expander("Grounded Curriculum Context"):
                                st.caption(retrieved_context)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.text,
                            "context": retrieved_context
                        })
                    except Exception as err:
                        st.error(f"Error communicating with Gemini model: {err}")

with tab_analytics:
    st.subheader("Session Telemetry & Engagement Analytics")
    st.caption("Temporal logging of cognitive state classifications and derived focus indices (SQLite Persistence).")

    telemetry_df = AnalyticsEngine.get_session_dataframe(limit=200)

    if not telemetry_df.empty:
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            mean_focus = telemetry_df["focus_score"].mean() * 100
            st.metric("Mean Focus Index", f"{mean_focus:.1f}%")
        with stat_col2:
            total_records = len(telemetry_df)
            st.metric("Logged Telemetry Points", f"{total_records}")
        with stat_col3:
            spoof_count = int(telemetry_df["is_spoof_flag"].sum())
            st.metric("Spoof Anomalies Flagged", f"{spoof_count}")

        # Render Plotly Time Series Chart
        fig = px.line(
            telemetry_df,
            x="datetime_str",
            y="focus_score",
            color="cognitive_state",
            title="Temporal Focus Score Trajectory",
            labels={"datetime_str": "Timestamp", "focus_score": "Focus Index (0.0 - 1.0)", "cognitive_state": "Cognitive State"},
            markers=True,
            template="plotly_white"
        )
        fig.update_yaxes(range=[-0.05, 1.05])
        fig.update_layout(xaxis_tickangle=-45, legend_title_text="Detected State")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw Telemetry Log Table"):
            st.dataframe(telemetry_df, use_container_width=True)
    else:
        st.info("Telemetry data will populate here after initial periodic sampling (5-second intervals).")
