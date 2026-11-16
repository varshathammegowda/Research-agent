"""Compatibility facade for the existing /research-agent endpoint."""
import os
from dotenv import load_dotenv
from google import genai
from agent.research_agent import run

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")
client = genai.Client(api_key=api_key)

def run_research_agent(query: str):
    return run(client, query)
