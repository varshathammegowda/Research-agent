from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API key loaded:", bool(api_key))
if not api_key:
    raise SystemExit("GEMINI_API_KEY is not configured in backend/.env")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Reply only with OK",
)

if (response.text or "").strip() != "OK":
    raise AssertionError("Gemini did not return the expected OK response")
print("Gemini response: OK")