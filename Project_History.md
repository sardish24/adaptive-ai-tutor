# Project History: Adaptive AI Tutor

## Phase 1: Project Selection & Strategy

### User Initial Request

Requested an AI/ML project to build over a few months that would be highly impactful for a resume, open to research and innovation.

### Evaluated Project Ideas

1. **Swarm Intelligence for Precision Agriculture (Edge AI + Federated Learning)**: Decentralized Federated Learning architecture for edge devices to reduce bandwidth usage in crop disease detection.
2. **Zero-Shot Low-Resource Indic Language Translator & Synthesizer (NLP + Audio)**: Fine-tuned multi-modal translation model for zero-shot translation in low-resource Indic languages with custom TTS pipelines.
3. **Adversarial Robustness in Medical Imaging (Deep Learning + Cybersecurity)**: Engineered an adversarially robust deep learning pipeline for medical diagnostics, mitigating FGSM/PGD attacks using Grad-CAM explainability.
4. **Personalized AI Tutor with Cognitive State Detection (Computer Vision + LLMs)**: Adaptive EdTech feedback loop integrating real-time Computer Vision (facial micro-expression analysis) with LLMs to dynamically alter teaching strategies based on student cognitive load.

### Prerequisites & Comparative Study

- **Project 1 (Swarm Intelligence)**: Very High complexity (hardware + networking). Most difficult.
- **Project 2 (Indic Language Translator)**: Moderate to Difficult. Requires cloud GPU knowledge.
- **Project 3 (Adversarial Robustness)**: Difficult. Highly theoretical and math-heavy.
- **Project 4 (Personalized AI Tutor)**: Moderate complexity. High software engineering, fast feedback loop.

### Selected Project & Roadmap

- **Selection**: Personalized AI Tutor with Cognitive State Detection.
- **Goal**: 1-month MVP timeline prioritizing resume impact and end-to-end functionality, followed by robustness patches.
- **Tools & Tech Stack**:
  - **IDE**: Antigravity IDE
  - **Deployment**: Streamlit Community Cloud / Vercel / Render
  - **LLM**: Google Gemini API (`gemini-3.6-flash`)
  - **CV**: Google MediaPipe (local CPU) + OpenCV
  - **Framework**: Streamlit + `streamlit-webrtc`

---

## Phase 2: 4-Week MVP Execution Plan

### Week 1: Environment Setup & LLM Chatbot Foundation

- **Objective**: Build a functional chatbot using Streamlit and Gemini API.
- **Tasks**:
  - Environment setup (`venv`, `streamlit`, `google-generativeai`, `opencv-python`, `mediapipe`, `python-dotenv`).
  - GitHub repository initialization and remote synchronization: `https://github.com/sardish24/adaptive-ai-tutor`.
  - Streamlit chat UI with session memory (`st.session_state`) and system prompt.
  - Resolved API model version deprecation issues by targeting `gemini-3.6-flash`.
- **Result**: Successfully deployed Week 1 MVP.
  - Live App: [https://adaptive-ai-tutor.streamlit.app](https://adaptive-ai-tutor.streamlit.app/)
  - GitHub Repo: [https://github.com/sardish24/adaptive-ai-tutor](https://github.com/sardish24/adaptive-ai-tutor)

### Week 2: Real-Time Computer Vision in the Browser

- **Objective**: Display a live webcam feed inside the Streamlit app and extract facial landmarks.
- **Tasks**:
  - Integrate `streamlit-webrtc` for browser-based video streaming.
  - Integrate MediaPipe Face Mesh inside `video_frame_callback` to extract 468 facial landmarks.
  - Build two-column layout (Chat interface on left, Webcam feed on right).
  - Test latency and frame conversion stability (`bgr24`).

### Week 3: State Detection + Feedback Loop

- **Objective**: Calculate engagement metrics, determine cognitive state, and dynamically alter the LLM prompt.
- **Tasks**:
  - Calculate Eye Aspect Ratio (EAR) from landmarks to detect drowsiness / closed eyes.
  - Calculate gaze-direction metrics from iris landmarks to detect distraction / looking away.
  - Build state classifier: `ENGAGED` vs `CONFUSED` / `DISTRACTED` (with time-persistence thresholds, e.g., 2+ seconds).
  - Dynamically inject cognitive state into the Gemini system prompt (e.g., provide simplified analogies when confusion is detected).

### Week 4: Deploy + Package

- **Objective**: Production deployment, documentation, and resume artifacts.
- **Tasks**:
  - Finalize `requirements.txt` and deployment configs.
  - Production STUN/TURN server configuration for WebRTC where needed.
  - Complete `README.md` with architectural diagrams and project demo.
  - Record walkthrough demo and summarize project achievements for resume.

---

## Phase 3: Patching & Robustness (Post-MVP)

- **Performance**: Asynchronous/multithreaded CV frame processing to prevent UI blocking.
- **Robustness**: Head Pose Estimation (Pitch, Yaw, Roll) to reduce false positives from head tilting or low lighting.
- **Smoothing**: Temporal exponential smoothing on classification outputs to prevent rapid state flickering.

---

## Context Management Protocol

- Maintain `PROGRESS.md` for active tracking of sprints, architectural decisions, and bug fixes.
- Use `Project_History.md` as persistent reference documentation across sessions.
