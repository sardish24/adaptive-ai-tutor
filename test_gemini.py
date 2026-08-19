import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3.6-flash")

response = model.generate_content("Explain quantum entanglement simply, in 3 sentences.")

print("--- Gemini Response ---")
print(response.text)
