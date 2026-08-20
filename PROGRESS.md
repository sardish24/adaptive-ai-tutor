# Progress Log

Chronological log of what was built, what broke, and how it was fixed.
This will be the basis for the final technical blog post.

## Week 1

- [x] Environment setup
- [x] Gemini API test script
- [x] Streamlit chat UI
- [x] MediaPipe webcam facial/gaze analysis (WebRTC Streamer)
- [x] Multi-metric cognitive state detection (4 states: Focused, Confused, Distracted, Drowsy)
- [x] Adaptive Gemini feedback loop (dynamic system prompt injection)

### Log Details

- **Built**:
  - Configured Streamlit interface with chat history management and teaching style selector (Standard, Socratic, Simplified, Encouraging).
  - Created standalone test script `test_gemini.py` for verifying Gemini API integration.
- **What broke**:
  - Legacy model strings (`gemini-1.5-pro`, `gemini-2.5-flash`) returned `404 NotFound` errors when querying the latest API endpoints.
- **Fix**:
  - Verified active available generation models on the API key and updated integration to use `gemini-3.6-flash`.
