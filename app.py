import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Adaptive AI Tutor", page_icon="🎓", layout="wide")
st.title("Adaptive AI Tutor")

if not api_key:
    st.error("GEMINI_API_KEY is not set. Please add it to your .env file or environment variables.")
    st.stop()

genai.configure(api_key=api_key)

# Sidebar settings
with st.sidebar:
    st.header("Tutor Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash"],
        index=0
    )
    teaching_style = st.selectbox(
        "Teaching Mode",
        ["Standard", "Socratic (Question-based)", "Simplified / Beginner", "Encouraging / Detailed"],
        index=0
    )
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# System instructions based on mode
system_prompts = {
    "Standard": "You are a knowledgeable, patient, and adaptive AI tutor helping students understand concepts clearly.",
    "Socratic (Question-based)": "You are a Socratic AI tutor. Guide the student by asking thoughtful questions rather than directly giving answers.",
    "Simplified / Beginner": "You are an AI tutor explaining complex concepts with simple analogies and easy-to-understand terms.",
    "Encouraging / Detailed": "You are a supportive, enthusiastic AI tutor providing detailed step-by-step breakdowns and encouraging feedback."
}

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask your tutor a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            model = genai.GenerativeModel(
                model_name=model_choice,
                system_instruction=system_prompts.get(teaching_style)
            )
            # Format history for Gemini chat
            chat_history = [
                {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                for m in st.session_state.messages[:-1]
            ]
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            error_message = f"Error communicating with Gemini: {e}"
            st.error(error_message)

