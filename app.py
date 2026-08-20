import os
import time
import av
import cv2
import numpy as np
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

# Global shared state container for WebRTC thread -> Streamlit UI communication
class StateHolder:
    current_state = "Focused / Attentive"
    confidence = 90.0
    last_metrics = {
        "yaw_ratio": 0.0,
        "pitch_ratio": 0.5,
        "roll_deg": 0.0,
        "mouth_aspect": 0.5
    }
    last_update_ts = time.time()

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
        # State tracking buffers for temporal smoothing
        self.state_history = []
        self.history_len = 15

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w, _ = img_bgr.shape

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

        # Detect face and 5 key landmarks (right eye, left eye, nose, right mouth, left mouth)
        _, faces = self.detector.detect(img_bgr)

        detected_state = "Searching for Face..."
        conf = 0.0
        status_color = (128, 128, 128)

        if faces is not None and len(faces) > 0:
            face = faces[0]
            box = face[0:4].astype(int)
            x, y, bw, bh = box
            
            # Confidence score of face detection
            face_score = float(face[14]) if len(face) > 14 else 0.9

            # Landmarks: [0: r_eye, 1: l_eye, 2: nose, 3: r_mouth, 4: l_mouth]
            landmarks = face[4:14].reshape((5, 2)).astype(float)
            r_eye, l_eye, nose, r_mouth, l_mouth = landmarks

            # Draw bounding box
            cv2.rectangle(img_bgr, (x, y), (x + bw, y + bh), (0, 255, 128), 2)

            # Draw landmarks
            for pt in landmarks:
                cv2.circle(img_bgr, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)

            # Contour guide lines
            cv2.line(img_bgr, (int(r_eye[0]), int(r_eye[1])), (int(l_eye[0]), int(l_eye[1])), (255, 200, 0), 1)
            cv2.line(img_bgr, (int(r_mouth[0]), int(r_mouth[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 200, 0), 1)

            # --- Biometric Computations ---
            # 1. Inter-ocular distance (eye span)
            eye_dist = np.linalg.norm(l_eye - r_eye) + 1e-5
            eye_center = (r_eye + l_eye) / 2.0

            # 2. Horizontal yaw ratio (looking left / right)
            yaw_ratio = (nose[0] - eye_center[0]) / eye_dist

            # 3. Vertical pitch ratio (head drooping down / looking up)
            pitch_ratio = (nose[1] - eye_center[1]) / eye_dist

            # 4. Head roll angle in degrees (head tilt)
            roll_deg = float(np.degrees(np.arctan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0])))

            # 5. Mouth width ratio
            mouth_dist = np.linalg.norm(l_mouth - r_mouth)
            mouth_aspect = mouth_dist / eye_dist

            # --- Multi-Metric Cognitive State Classification ---
            if pitch_ratio > 0.85:
                # Head drooped down significantly
                detected_state = "Drowsy / Fatigued"
                conf = min(98.0, 75.0 + (pitch_ratio - 0.85) * 50)
                status_color = (0, 0, 255)  # Red
            elif abs(yaw_ratio) > 0.28:
                # Sustained off-center gaze
                detected_state = "Distracted / Looking Away"
                conf = min(95.0, 70.0 + abs(yaw_ratio) * 60)
                status_color = (0, 165, 255)  # Orange
            elif abs(roll_deg) > 11.0 or (0.28 <= pitch_ratio <= 0.40):
                # Head tilted sideways or brow lowered with squint/tension
                detected_state = "Confused / High Cognitive Load"
                conf = min(92.0, 68.0 + abs(roll_deg) * 2.0)
                status_color = (0, 255, 255)  # Yellow
            else:
                # Balanced forward orientation
                detected_state = "Focused / Attentive"
                conf = max(80.0, face_score * 100)
                status_color = (0, 255, 0)  # Green

            # Temporal smoothing over buffer
            self.state_history.append(detected_state)
            if len(self.state_history) > self.history_len:
                self.state_history.pop(0)

            # Majority voting for stable state transitions
            smoothed_state = max(set(self.state_history), key=self.state_history.count)

            StateHolder.current_state = smoothed_state
            StateHolder.confidence = conf
            StateHolder.last_metrics = {
                "yaw_ratio": round(float(yaw_ratio), 2),
                "pitch_ratio": round(float(pitch_ratio), 2),
                "roll_deg": round(float(roll_deg), 1),
                "mouth_aspect": round(float(mouth_aspect), 2)
            }
            StateHolder.last_update_ts = time.time()

        # Display Live Cognitive State Badge
        cv2.rectangle(img_bgr, (10, 10), (w - 10, 50), (20, 20, 20), -1)
        cv2.putText(
            img_bgr,
            f"State: {StateHolder.current_state} ({int(StateHolder.confidence)}%)",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- Page Config ---
st.set_page_config(page_title="Adaptive AI Tutor", layout="wide", page_icon="🎓")
st.title("🎓 Adaptive AI Tutor")
st.caption("Real-time biometric cognitive state detection feeding dynamically into an adaptive Gemini tutor.")

# --- STUN Server Configuration for Cloud WebRTC ---
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}

# --- Two-Column Layout ---
col_chat, col_webcam = st.columns([3, 2], gap="large")

with col_webcam:
    st.subheader("📹 Real-Time Biometric & Gaze Stream")
    webrtc_streamer(
        key="adaptive-webcam",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Live Biometric Metric Display Cards
    st.markdown("##### 🧠 Live Cognitive Telemetry")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Detected State", StateHolder.current_state)
        st.metric("Head Roll (Tilt)", f"{StateHolder.last_metrics['roll_deg']}°")
    with m_col2:
        st.metric("Confidence", f"{int(StateHolder.confidence)}%")
        st.metric("Gaze Yaw Ratio", f"{StateHolder.last_metrics['yaw_ratio']}")

with col_chat:
    st.subheader("💬 Adaptive Tutor Chat")

    # Session state init
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask your tutor a question...")

    if user_input:
        # Append and render user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Dynamic System Instruction Injection based on live cognitive state
        current_state = StateHolder.current_state
        
        if current_state == "Confused / High Cognitive Load":
            adaptive_instruction = (
                "The student's webcam biometric analysis indicates they are CONFUSED or experiencing high cognitive load. "
                "Break down the answer using intuitive real-world analogies, simpler vocabulary, and step-by-step clarity. "
                "Conclude with a brief check-in question to verify their understanding."
            )
        elif current_state == "Distracted / Looking Away":
            adaptive_instruction = (
                "The student's biometric stream indicates they are DISTRACTED or looking away. "
                "Keep the explanation punchy, engaging, and concise (under 3 paragraphs). "
                "End with an interactive quick question to pull their attention back."
            )
        elif current_state == "Drowsy / Fatigued":
            adaptive_instruction = (
                "The student appears DROWSY or fatigued. "
                "Give a very concise, direct summary in bullet points and suggest a quick 1-minute stretch or interactive quiz."
            )
        else:
            adaptive_instruction = (
                "The student is FOCUSED and engaged. "
                "Deliver a comprehensive, high-quality, structured explanation with standard technical depth."
            )

        full_system_prompt = (
            "You are an expert, empathetic, and adaptive AI tutor.\n\n"
            f"LIVE STUDENT STATE: {current_state} (Confidence: {int(StateHolder.confidence)}%)\n"
            f"ADAPTIVE PEDAGOGICAL DIRECTIVE: {adaptive_instruction}"
        )

        with st.chat_message("assistant"):
            with st.spinner(f"Adapting explanation to your state ({current_state})..."):
                try:
                    adaptive_model = genai.GenerativeModel(
                        model_name="gemini-3.6-flash",
                        system_instruction=full_system_prompt
                    )
                    
                    # Prepare history for Gemini
                    chat_history = [
                        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                        for m in st.session_state.messages[:-1]
                    ]
                    chat = adaptive_model.start_chat(history=chat_history)
                    response = chat.send_message(user_input)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error communicating with Gemini: {e}")
