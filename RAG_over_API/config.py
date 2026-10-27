"""
config.py  -  all settings in ONE place.
Every other file imports from here, so students only need to look here
to see which model / database / key is being used.
"""
import os
from dotenv import load_dotenv
load_dotenv()  # reads the .env file (secrets) into environment variables

# ---- Qdrant (vector database, runs locally in a folder - no Docker needed) ----
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_data")
COLLECTION = os.getenv("COLLECTION", "hr_policy")

# ---- Embedding model (open source, served by Ollama on your laptop) ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# nomic-embed-text works best when text is tagged with a task prefix.
# If you switch to another model (e.g. mxbai-embed-large) set both to "" in .env
DOC_PREFIX = os.getenv("DOC_PREFIX", "search_document: ")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "search_query: ")

# ---- LLM (Google Gemini via Google AI Studio free tier) ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ---- Third-party API (Tavily web search, free tier = 1000 credits/month) ----
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---- Chunking ----
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "60"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "10"))

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "3"))

# ---- Re-ranking (rerank.py) ----
# Step A fetches FETCH_K candidates from Qdrant (wide net),
# Step B re-scores them with a cross-encoder and keeps the best TOP_K.
FETCH_K = int(os.getenv("FETCH_K", "8"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
