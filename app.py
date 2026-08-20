import os
import time
import math
import numpy as np
import av
import cv2
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Fallback to Streamlit secrets (for Streamlit Community Cloud deployment)
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your local .env file or Streamlit Cloud Secrets.")
    st.stop()

genai.configure(api_key=api_key)

# Global shared state container across WebRTC worker thread and Streamlit script runner
class SharedMetrics:
    def __init__(self):
        self.state = "Focused / Attentive"
        self.confidence = 0.95
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.last_updated = time.time()

if "shared_metrics" not in st.session_state:
    st.session_state.shared_metrics = SharedMetrics()

shared_metrics = st.session_state.shared_metrics

# --- High-Performance Face and Landmark Detection ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet.onnx")
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def ensure_model_file():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

ensure_model_file()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = None
        self.current_size = None
        # Smoothing filters & timers
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.smooth_roll = 0.0
        self.confused_start_time = None
        self.distracted_start_time = None
        self.drowsy_start_time = None
        self.current_state = "Focused / Attentive"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w, _ = img_bgr.shape
        now = time.time()

        # Initialize detector with matching frame size
        if self.detector is None or self.current_size != (w, h):
            self.detector = cv2.FaceDetectorYN.create(
                MODEL_PATH,
                "",
                (w, h),
                score_threshold=0.6,
                nms_threshold=0.3
            )
            self.current_size = (w, h)

        # Detect face and landmarks
        _, faces = self.detector.detect(img_bgr)

        status_text = "Tracking: Searching..."
        status_color = (0, 165, 255)

        if faces is not None and len(faces) > 0:
            face = faces[0]
            box = face[0:4].astype(int)
            x, y, bw, bh = box

            # Bounding box
            cv2.rectangle(img_bgr, (x, y), (x + bw, y + bh), (0, 255, 128), 2)

            # Key facial landmarks: right eye, left eye, nose tip, right mouth, left mouth
            landmarks = face[4:14].reshape((5, 2)).astype(float)
            r_eye, l_eye, nose, r_mouth, l_mouth = landmarks

            # Draw landmarks
            for lx, ly in landmarks.astype(int):
                cv2.circle(img_bgr, (lx, ly), 4, (0, 255, 255), -1)

            cv2.line(img_bgr, tuple(landmarks[0].astype(int)), tuple(landmarks[1].astype(int)), (255, 200, 0), 1)
            cv2.line(img_bgr, tuple(landmarks[3].astype(int)), tuple(landmarks[4].astype(int)), (255, 200, 0), 1)

            # 1. Roll angle (Head Tilt) from eye slope
            dx = l_eye[0] - r_eye[0]
            dy = l_eye[1] - r_eye[1]
            raw_roll = math.degrees(math.atan2(dy, dx + 1e-5))

            # 2. Yaw angle (Looking left/right) from nose position relative to eye mid-point
            eye_mid_x = (r_eye[0] + l_eye[0]) / 2.0
            eye_dist = math.hypot(dx, dy) + 1e-5
            raw_yaw = ((nose[0] - eye_mid_x) / eye_dist) * 60.0

            # 3. Pitch angle (Nodding / Looking up/down)
            eye_mid_y = (r_eye[1] + l_eye[1]) / 2.0
            mouth_mid_y = (r_mouth[1] + l_mouth[1]) / 2.0
            face_height_proxy = mouth_mid_y - eye_mid_y + 1e-5
            nose_vert_ratio = (nose[1] - eye_mid_y) / face_height_proxy
            raw_pitch = (nose_vert_ratio - 0.55) * 80.0

            # Apply Exponential Smoothing (alpha = 0.25)
            alpha = 0.25
            self.smooth_roll = alpha * raw_roll + (1 - alpha) * self.smooth_roll
            self.smooth_yaw = alpha * raw_yaw + (1 - alpha) * self.smooth_yaw
            self.smooth_pitch = alpha * raw_pitch + (1 - alpha) * self.smooth_pitch

            # 4. Multi-metric State Machine Evaluation
            is_distracted = abs(self.smooth_yaw) > 22.0 or abs(self.smooth_pitch) > 25.0
            is_tilted = abs(self.smooth_roll) > 13.0
            is_drooping = self.smooth_pitch > 20.0 and abs(self.smooth_yaw) < 10.0

            # Temporal persistence thresholds (prevent rapid flickering)
            if is_distracted:
                if self.distracted_start_time is None:
                    self.distracted_start_time = now
                if now - self.distracted_start_time > 1.5:
                    self.current_state = "Distracted / Looking Away"
            else:
                self.distracted_start_time = None

            if is_drooping:
                if self.drowsy_start_time is None:
                    self.drowsy_start_time = now
                if now - self.drowsy_start_time > 2.0:
                    self.current_state = "Drowsy / Fatigued"
            else:
                self.drowsy_start_time = None

            if is_tilted and not is_distracted and not is_drooping:
                if self.confused_start_time is None:
                    self.confused_start_time = now
                if now - self.confused_start_time > 1.2:
                    self.current_state = "Confused / Struggling"
            else:
                self.confused_start_time = None

            if not is_distracted and not is_tilted and not is_drooping:
                self.current_state = "Focused / Attentive"

            # Update shared metrics
            shared_metrics.state = self.current_state
            shared_metrics.yaw = round(self.smooth_yaw, 1)
            shared_metrics.pitch = round(self.smooth_pitch, 1)
            shared_metrics.roll = round(self.smooth_roll, 1)
            shared_metrics.last_updated = now

            # Visual cues styling
            state_styles = {
                "Focused / Attentive": ((0, 255, 0), "Focused & Attentive"),
                "Confused / Struggling": ((0, 215, 255), "Confused (High Cognitive Load)"),
                "Distracted / Looking Away": ((0, 140, 255), "Distracted / Looking Away"),
                "Drowsy / Fatigued": ((0, 69, 255), "Drowsy / Low Energy")
            }
            status_color, label = state_styles.get(self.current_state, ((0, 255, 0), self.current_state))
            status_text = f"Cognitive State: {label}"

            # Display orientation metrics HUD
            cv2.putText(
                img_bgr,
                f"Yaw: {self.smooth_yaw:.0f}deg | Pitch: {self.smooth_pitch:.0f}deg | Tilt: {self.smooth_roll:.0f}deg",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

        # Overlay status banner
        cv2.putText(
            img_bgr,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- Page Config ---
st.set_page_config(page_title="Adaptive AI Tutor", layout="wide", page_icon="🎓")
st.title("Adaptive AI Tutor")

# --- Two-Column Layout ---
col_chat, col_webcam = st.columns([3, 2], gap="medium")

with col_webcam:
    st.subheader("Live Cognitive State Stream")
    webrtc_streamer(
        key="adaptive-webcam",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # State HUD Card
    st.markdown("### Real-Time Cognitive Telemetry")
    current_state_badge = shared_metrics.state
    badge_colors = {
        "Focused / Attentive": "🟢 **Focused / Attentive** (Optimal learning state)",
        "Confused / Struggling": "🟡 **Confused / Struggling** (Adaptive explanation triggered)",
        "Distracted / Looking Away": "🟠 **Distracted / Looking Away** (Attention re-engagement prompt)",
        "Drowsy / Fatigued": "🔴 **Drowsy / Fatigued** (Bite-sized pacing recommendation)"
    }
    st.info(badge_colors.get(current_state_badge, current_state_badge))

    with st.expander("Telemetry Details & Angles", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Yaw (Horizontal)", f"{shared_metrics.yaw}°")
        c2.metric("Pitch (Vertical)", f"{shared_metrics.pitch}°")
        c3.metric("Roll (Head Tilt)", f"{shared_metrics.roll}°")

with col_chat:
    st.subheader("Tutor Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask your tutor something...")

    if user_input:
        detected_state = shared_metrics.state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Dynamic System Instruction based on Real-Time Cognitive State
        adaptive_guidance = ""
        if detected_state == "Confused / Struggling":
            adaptive_guidance = (
                "\n[COGNITIVE STATE TRIGGER: The student's facial analysis indicates confusion or struggle with the concept. "
                "Use a vivid real-world analogy, break the explanation into numbered bite-sized steps, and invite them to ask for clarification on any specific part.]"
            )
        elif detected_state == "Distracted / Looking Away":
            adaptive_guidance = (
                "\n[COGNITIVE STATE TRIGGER: The student appears distracted or looking away. "
                "Keep the explanation punchy, concise, and end with an engaging direct question to check their understanding.]"
            )
        elif detected_state == "Drowsy / Fatigued":
            adaptive_guidance = (
                "\n[COGNITIVE STATE TRIGGER: The student shows signs of fatigue. "
                "Provide a very brief summary and suggest a quick 1-minute reset or interactive quiz.]"
            )
        else:
            adaptive_guidance = (
                "\n[COGNITIVE STATE TRIGGER: The student is focused and attentive. Provide clear, structured, and deep conceptual explanation.]"
            )

        dynamic_system_prompt = (
            "You are an expert, empathetic AI tutor helping students master concepts."
            + adaptive_guidance
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    active_model = genai.GenerativeModel(
                        model_name="gemini-3.6-flash",
                        system_instruction=dynamic_system_prompt
                    )
                    # Convert history to format accepted by Gemini chat
                    gemini_history = [
                        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                        for m in st.session_state.messages[:-1]
                    ]
                    chat = active_model.start_chat(history=gemini_history)
                    response = chat.send_message(user_input)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

