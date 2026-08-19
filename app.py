import os
import av
import cv2
import mediapipe as mp
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

SYSTEM_PROMPT = (
    "You are an expert AI tutor. Explain concepts clearly and concisely. "
    "Adapt your explanations to be genuinely helpful for a learner."
)

model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
    system_instruction=SYSTEM_PROMPT,
)

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

        # Detect face and landmarks
        _, faces = self.detector.detect(img_bgr)

        status_text = "Tracking: Searching..."
        status_color = (0, 165, 255)  # Orange

        if faces is not None and len(faces) > 0:
            face = faces[0]
            # Bounding box coordinates
            box = face[0:4].astype(int)
            x, y, bw, bh = box
            cv2.rectangle(img_bgr, (x, y), (x + bw, y + bh), (0, 255, 128), 2)

            # Key facial landmarks: right eye, left eye, nose tip, right mouth, left mouth
            landmarks = face[4:14].reshape((5, 2)).astype(int)
            for i, (lx, ly) in enumerate(landmarks):
                # Draw landmark points
                cv2.circle(img_bgr, (lx, ly), 4, (0, 255, 255), -1)

            # Connect eye and mouth contour guide lines
            cv2.line(img_bgr, tuple(landmarks[0]), tuple(landmarks[1]), (255, 200, 0), 1)
            cv2.line(img_bgr, tuple(landmarks[3]), tuple(landmarks[4]), (255, 200, 0), 1)

            # Calculate gaze / head orientation proxy
            r_eye, l_eye, nose = landmarks[0], landmarks[1], landmarks[2]
            eye_center_x = (r_eye[0] + l_eye[0]) / 2.0
            horizontal_offset = (nose[0] - eye_center_x) / (bw + 1e-5)

            if abs(horizontal_offset) < 0.15:
                status_text = "Cognitive State: Engaged & Focused"
                status_color = (0, 255, 0)
            else:
                status_text = "Cognitive State: Looking Away"
                status_color = (0, 215, 255)

        # Overlay status banner
        cv2.putText(
            img_bgr,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- Page Config ---
st.set_page_config(page_title="Adaptive AI Tutor", layout="wide")
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

with col_chat:
    st.subheader("Tutor Chat")

    # Session state init
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask your tutor something...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_session.send_message(user_input)
                st.markdown(response.text)

        st.session_state.messages.append({"role": "assistant", "content": response.text})
