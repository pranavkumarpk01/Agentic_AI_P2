"""config.py - settings and connections shared by ingest/ and query/.

It does three jobs:
  1. read the .env file
  2. create the Gemini client (used for embeddings and for answers)
  3. create the Qdrant client (the vector database)
"""
import os
from datetime import date

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient

load_dotenv()  # reads the .env file so os.getenv() can see the values

# "Today", used to decide which version of a document is in force. Set AS_OF_DATE in .env to test another day.
TODAY = os.getenv("AS_OF_DATE") or date.today().isoformat()

# Two tables ("collections") inside Qdrant, so the ISSUE and FIX versions can sit side by side.
COLLECTION_ISSUE = "policies_issue"   # filled by ingest_issue, searched by query_issue
COLLECTION_FIX = "policies_fix"       # filled by ingest_fix, searched by query_fix
VECTOR_SIZE = 768         # how many numbers each vector has
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")  # turns text into vectors
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")   # writes text (rewrite, rerank, answer)

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

QDRANT_URL = os.getenv("QDRANT_URL")
if QDRANT_URL:
    qdrant = QdrantClient(url=QDRANT_URL)           # Qdrant running in Docker
else:
    qdrant = QdrantClient(path="./qdrant_local")    # no Docker: Qdrant runs inside this program and saves to a folder
