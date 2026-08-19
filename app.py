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

# --- MediaPipe Tasks Face Landmarker Setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def ensure_model_file():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

ensure_model_file()

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w, _ = img_bgr.shape

        # Convert to RGB for MediaPipe Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = self.detector.detect(mp_image)

        # Draw facial landmarks
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                for lm in face_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img_bgr, (cx, cy), 1, (0, 255, 0), -1)

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
