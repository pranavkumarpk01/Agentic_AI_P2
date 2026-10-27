"""
llm.py  -  one tiny helper to talk to Gemini.
Used by router.py (to DECIDE) and integration.py (to ANSWER).
"""
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL


def generate(prompt: str, temperature: float = 0.2) -> str:
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is missing. Add it to your .env file.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text
