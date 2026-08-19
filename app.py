import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

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

# --- Page config ---
st.set_page_config(page_title="Adaptive AI Tutor", layout="wide")
st.title("Adaptive AI Tutor")

# --- Session state init ---
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []  # for rendering in UI

# --- Render existing chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask your tutor something...")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get Gemini response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(user_input)
            st.markdown(response.text)

    st.session_state.messages.append({"role": "assistant", "content": response.text})
