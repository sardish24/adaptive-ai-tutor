# 🎓 Adaptive AI Tutor

> **Real-time biometric cognitive state detection feeding dynamically into an adaptive Gemini tutor.**

[![Live App](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://adaptive-ai-tutor.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/sardish24/adaptive-ai-tutor)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📹 Video Demonstration

<!-- Replace with your Loom/YouTube recording link -->
[![Demo Video Placeholder](https://img.shields.io/badge/Watch-Demo%20Video%20(1--min)-red?style=for-the-badge&logo=youtube)](https://adaptive-ai-tutor.streamlit.app)

*Watch the 60-second walkthrough showing real-time biometric state transitions (`Focused` $\rightarrow$ `Confused` $\rightarrow$ `Distracted`) and dynamic Gemini pedagogical prompt adaptation.*

---

## 🏛️ System Architecture

```text
                                  +---------------------------------------+
                                  |            Client Browser             |
                                  |  (Webcam Feed + Streamlit Chat UI)    |
                                  +-------------------+-------------------+
                                                      |
                                           WebRTC Peer Connection
                                           (Google STUN Servers)
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |        Real-Time Vision Pipeline      |
                                  |       (OpenCV YuNet Deep Learning)    |
                                  +-------------------+-------------------+
                                                      |
                                          5-Point Facial Landmarks
                                          (Pitch, Yaw, Roll, Mouth)
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     4-State Cognitive Classifier      |
                                  |  (15-Frame Temporal Buffer Smoothing) |
                                  +-------------------+-------------------+
                                                      |
                                      Live State + Confidence Metric
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     Adaptive Gemini Feedback Loop     |
                                  | (Dynamic Pedagogical System Injection)|
                                  +-------------------+-------------------+
                                                      |
                                           Tailored Tutoring Output
                                                      v
                                  +---------------------------------------+
                                  |         Streamlit Chat UI View        |
                                  +---------------------------------------+
```

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Real-Time Streaming**: [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc) + `av` (PyAV)
- **Computer Vision & Inference**: [OpenCV](https://opencv.org/) (`cv2.FaceDetectorYN` YuNet ONNX runtime)
- **Large Language Model**: [Google Gemini API](https://ai.google.dev/) (`gemini-3.6-flash` via `google-generativeai`)
- **NAT / Traversal**: Google Public STUN Servers (`stun.l.google.com:19302`)
- **Deployment**: Streamlit Community Cloud

---

## 🧠 Core Engineering & Logic

### 1. Computer Vision & Biometric Extraction
Standard solutions like MediaPipe often suffer from thread-safety and execution stalls in asynchronous WebRTC workers. This project utilizes an embedded **YuNet deep learning ONNX detector**, extracting 5 key facial landmarks per frame:
- **Horizontal Yaw Ratio**: $\frac{\text{Nose}_x - \text{EyeCenter}_x}{\text{InterOcularDistance}}$ (Detects turning away/distraction).
- **Vertical Pitch Ratio**: $\frac{\text{Nose}_y - \text{EyeCenter}_y}{\text{InterOcularDistance}}$ (Detects head drooping/drowsiness).
- **Head Roll Angle**: $\arctan2(\Delta y_{\text{eyes}}, \Delta x_{\text{eyes}})$ (Detects head tilting associated with confusion/questioning).
- **Mouth Aspect Ratio**: $\frac{\text{MouthWidth}}{\text{InterOcularDistance}}$ (Tracks facial tension/jaw posture).

### 2. Multi-Metric 4-State Classifier
| Cognitive State | Biometric Trigger Thresholds | Adaptive Pedagogical Directive |
| :--- | :--- | :--- |
| **Focused / Attentive** | Balanced forward orientation ($\text{Yaw} < 0.28$, $\text{Roll} < 11^\circ$) | Delivers rigorous, structured explanations with standard technical depth. |
| **Confused / High Cognitive Load** | Head tilt $\text{Roll} > 11^\circ$ or brow-furrowed pitch ($0.28 \le \text{Pitch} \le 0.40$) | Breaks concepts down into step-by-step parts, introduces real-world analogies, and adds a check-in question. |
| **Distracted / Looking Away** | Sustained off-center gaze ($\text{Yaw} > 0.28$) | Delivers concise explanations under 3 paragraphs with a quick engagement check. |
| **Drowsy / Fatigued** | Significant head droop ($\text{Pitch} > 0.85$) | Provides bullet-point summaries and suggests a short stretch break. |

### 3. 15-Frame Temporal Smoothing
To prevent state flickering from rapid micro-movements, raw frame classifications are pushed to a FIFO sliding window ($N = 15$ frames). A majority-voting heuristic stabilizes state transitions before reporting telemetry to the LLM.

---

## 🚀 Local Installation & Setup

### Prerequisites
- Python 3.10 - 3.14
- A working webcam
- A [Google Gemini API Key](https://aistudio.google.com/)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sardish24/adaptive-ai-tutor.git
   cd adaptive-ai-tutor
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

5. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser and click **START** on the webcam feed.

---

## 🌐 Live Deployment

The application is deployed on Streamlit Community Cloud:
🔗 **[https://adaptive-ai-tutor.streamlit.app](https://adaptive-ai-tutor.streamlit.app)**

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
