# Adaptive AI Tutor with Cognitive Biometrics & Curriculum RAG

An enterprise-grade intelligent tutoring and proctoring platform that dynamically adapts pedagogical explanations based on real-time computer vision biometrics, curriculum-grounded Retrieval-Augmented Generation (RAG), OS-level window telemetry, presentation-attack anti-spoofing, and Explainable AI (XAI) motion heatmaps.

---

## System Overview

Traditional educational platforms operate statically, delivering identical content regardless of student engagement, confusion, or fatigue. The **Adaptive AI Tutor** bridges this gap by coupling real-time face pose and gaze analytics with Large Language Models (LLMs). By analyzing high-frequency facial telemetry alongside indexed curriculum documents, the system adapts its tone, depth, and pacing dynamically while safeguarding study integrity through automated distraction and spoofing detection.

---

## Core Innovations

### 1. High-Performance Vision Pipeline (OpenCV YuNet & WebRTC)
- Replaces high-overhead vision frameworks with lightweight ONNX-quantized **OpenCV YuNet** (`cv2.FaceDetectorYN`) running directly in asynchronous `streamlit-webrtc` processing loops.
- Delivers $\ge 30\text{ FPS}$ multi-metric face detection and 5-point landmark extraction without UI thread starvation.

### 2. Multi-Metric 4-State Cognitive Classifier
- Estimates instantaneous head pose ($roll$, $pitch$, $yaw$) and gaze ratios from inter-ocular and eye-to-nose distances.
- Classifies user engagement into four distinct states:
  - `Focused / Attentive`
  - `Confused / High Cognitive Load`
  - `Distracted / Looking Away`
  - `Drowsy / Fatigued`
- Employs a 15-frame rolling majority voting window to filter camera noise and eliminate state flickering.

### 3. Curriculum Retrieval-Augmented Generation (RAG)
- Ingests PDF, Markdown, and plain text study materials from `./study_materials` using LangChain character chunking.
- Generates high-dimensional vector embeddings with Google Gemini embedding models and retrieves context chunks via cosine similarity scoring ($\ge 0.50$ relevance threshold).
- Injects retrieved source citations and live pedagogical directives directly into Gemini system instructions.

### 4. OS-Level Cross-Tab Distraction Monitoring
- Operates a background daemon thread polling active desktop window titles via `pygetwindow`.
- Identifies active YouTube video sessions, extracts video identifiers, and fetches transcripts using `youtube-transcript-api`.
- Validates the semantic relevance of the external media against the active study materials vector store, firing automated UI distraction alerts for off-topic content.

### 5. Biometric Anti-Spoofing & Liveness Verification
- Tracks Eye Aspect Ratio (EAR) micro-movement variance across a 60-second rolling sample buffer ($N \ge 150$).
- Identifies static presentation attacks (photographs or frozen video feeds) when sample variance drops below physiological thresholds ($\sigma^2 < 0.00005$), temporarily pausing interactive LLM tutoring sessions.

### 6. Explainable AI (XAI) Motion-Magnitude Heatmaps
- Computes dense optical flow fields via `cv2.calcOpticalFlowFarneback` across consecutive video frames.
- Normalizes motion vector magnitudes and generates a thermal gradient overlay (`cv2.COLORMAP_JET`) blended onto the live camera feed to visually substantiate why specific movements triggered state transitions.

### 7. Session Telemetry & Persistence
- Asynchronously logs cognitive states, confidence values, derived focus indices ($1.0$ for focused, $0.5$ for confused, $0.1$ for drowsy, $0.0$ for distracted), spoof flags, and pose metrics into a local SQLite database (`session_telemetry.db`).
- Renders temporal engagement trajectories and summary metrics via an interactive Plotly analytics dashboard.

---

## System Architecture

```text
                        +---------------------------------------+
                        |         Webcam Input Frame            |
                        +---------------------------------------+
                                           |
                                           v
                        +---------------------------------------+
                        |  streamlit-webrtc VideoTransformer   |
                        +---------------------------------------+
                               /           |           \
                              /            |            \
                             v             v             v
                    +--------------+ +-----------+ +-------------------+
                    | OpenCV YuNet | | Optical   | | Rolling EAR       |
                    | Landmark Det | | Flow XAI  | | Liveness Analyzer |
                    +--------------+ +-----------+ +-------------------+
                            |              |                 |
                            v              v                 v
                    +--------------+  [Thermal    [Spoof Flag /
                    | 4-State Pose |   Overlay]    Input Lockout]
                    | Classifier   |
                    +--------------+
                            |
                            v
               +-------------------------+      +---------------------------+
               | Thread-Safe StateHolder | <--- | Background OS Tracker     |
               +-------------------------+      | (pygetwindow / YouTube)   |
                      |           |             +---------------------------+
                      |           |                           |
                      v           v                           v
          +-------------+   +-------------------+   +-----------------------+
          | SQLite DB   |   | Gemini LLM Engine |   | RAG Relevance Checker |
          | Telemetry   |   | (System Prompt    |   | (ChromaDB / JSON)     |
          +-------------+   |  Injection)       |   +-----------------------+
                 |          +-------------------+               ^
                 v                    |                         |
          +-------------+             v             +-----------------------+
          | Plotly      |    +------------------+   | Local Study Materials |
          | Dashboard   |    | Streamlit UI     |   | (PDF / MD / TXT)      |
          +-------------+    +------------------+   +-----------------------+
```

---

## Technology Stack

| Layer | Technologies |
| --- | --- |
| **User Interface & WebRTC** | Streamlit, streamlit-webrtc, PyAV |
| **Computer Vision & XAI** | OpenCV (YuNet ONNX, Farneback Dense Optical Flow, Jet Colormap), NumPy |
| **LLM & Embeddings** | Google Gemini Generative API (`gemini-3.6-flash`, `gemini-embedding-001`) |
| **Document Retrieval (RAG)** | LangChain Recursive Splitters, PyPDF, Vector Similarity Engine |
| **OS Monitoring & Media** | `pygetwindow`, `youtube-transcript-api`, Google API Client |
| **Analytics & Data Storage** | SQLite3, Pandas, Plotly Express |
| **Environment Configuration** | Python 3.10+, `python-dotenv` |

---

## Installation & Local Execution

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/sardish24/adaptive-ai-tutor.git
cd adaptive-ai-tutor

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a `.env` file in the project root based on `.env.example`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here  # Optional: for video title metadata resolution
```

### 4. Populate Knowledge Base
Place subject curriculum documents (PDFs, Markdown notes, or plain text files) into `./study_materials`:
```bash
study_materials/
  ├── linear_regression.md
  └── quantum_physics_notes.pdf
```

### 5. Launch Application
```bash
streamlit run app.py
```

---

## Operational Environment Requirements

- **OS Window Tracking (`pygetwindow`)**: Requires execution in a local desktop environment (Windows, macOS, or X11 desktop sessions) with active window management privileges.
- **Local Persistence (`session_telemetry.db`)**: Local file read/write access is required for SQLite session logging.
- **WebRTC Camera Stream**: A functioning webcam and standard browser WebRTC permissions are required for real-time telemetry extraction.
