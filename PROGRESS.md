# Progress Log

Chronological log of what was built, what broke, and how it was fixed.
This will be the basis for the final technical blog post.

## Week 1 / Phase 1 & 2 Transition

- [x] Environment setup
- [x] Gemini API test script
- [x] Streamlit chat UI
- [x] MediaPipe / OpenCV YuNet webcam facial & gaze analysis (WebRTC Streamer)
- [x] Multi-metric cognitive state detection (4 states: Focused, Confused, Distracted, Drowsy)
- [x] Adaptive Gemini feedback loop (dynamic system prompt injection)
- [x] Live proctoring alerts (sustained distraction > 10s toast & UI warnings)
- [x] Curriculum RAG vector pipeline (`rag_engine.py` with Gemini embeddings)
- [x] Phase 1 Testing & Verification ready (RAG ingestion, document grounding, proctoring alert triggers)
- [x] Phase 2 Implementation: Thread-safe background OS window tracking (`os_tracker.py` using `pygetwindow`)
- [x] Phase 2 Implementation: YouTube transcript & metadata extraction (`youtube_engine.py`)
- [x] Codebase Professionalization: PEP-8 compliant docstrings, formal terminology, snake_case conventions, complete removal of informal text and emojis
- [x] Phase 3 Implementation: Biometric anti-spoofing liveness detection (EAR variance micro-movement tracking in `VideoTransformer`)
- [x] Phase 3 Implementation: SQLite session telemetry logging (`analytics_engine.py` / `session_telemetry.db`)
- [x] Phase 3 Implementation: Plotly time-series engagement dashboard in dedicated Streamlit tab

### Log Details

- **Built**:
  - Configured Streamlit interface with chat history management and WebRTC webcam streamer.
  - Implemented 4-state real-time cognitive state classifier (Focused, Confused, Distracted, Drowsy) with head roll angle, yaw ratio, pitch ratio, and temporal buffer smoothing.
  - Built live biometric telemetry dashboard metrics.
  - Implemented persistent vector RAG engine (`rag_engine.py`) indexing PDF, Markdown, and TXT files from `./study_materials` using `models/gemini-embedding-001` embeddings.
  - Augmented Gemini system instructions with retrieved curriculum chunks and biometric state directives.
  - Implemented live proctoring alerts (`st.toast` and `st.warning`) for sustained off-screen distraction (> 10s) and drowsiness.
  - Created standalone test script `test_gemini.py` for verifying Gemini API integration.
  - Added new dependencies to `requirements.txt` (`chromadb`, `langchain`, `langchain-google-genai`, `pygetwindow`, `google-api-python-client`) and updated environment configurations.
- **What broke**:
  - Legacy model strings (`gemini-1.5-pro`, `gemini-2.5-flash`) returned `404 NotFound` errors when querying the latest API endpoints.
- **Fix**:
  - Verified active available generation models on the API key and updated integration to use `gemini-3.6-flash`.
