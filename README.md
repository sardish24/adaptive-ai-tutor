# Adaptive AI Tutor

Real-time cognitive state detection (via webcam) feeding into an adaptive
Gemini-powered tutoring chatbot. When the system detects confusion or disengagement through facial/gaze analysis, it dynamically adjusts the LLM's teaching style.

## Status

🚧 In development (Week 1 — MVP sprint)

## Tech Stack

- Streamlit (UI)
- Google Gemini API (LLM)
- MediaPipe (facial landmark detection)
- streamlit-webrtc (real-time video)
- OpenCV (image processing)

## Setup

1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it and run `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your Gemini API key
5. Run `streamlit run app.py`
