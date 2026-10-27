"""
embed.py  -  STEP 4 of ingestion (also used at query time): text -> vector.

We call the Ollama server running on your laptop. An embedding is a list of
numbers (768 for nomic-embed-text). Texts with similar MEANING get vectors
that are close to each other.
"""
import requests
from config import OLLAMA_URL, EMBED_MODEL, DOC_PREFIX, QUERY_PREFIX


def _embed(texts: list[str]) -> list[list[float]]:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit("Cannot reach Ollama. Is it running? (open the Ollama app or run: ollama serve)")
    except requests.exceptions.HTTPError as e:
        raise SystemExit(f"Ollama error: {e}. Did you run: ollama pull {EMBED_MODEL} ?")
    return resp.json()["embeddings"]


def embed_documents(chunks: list[str]) -> list[list[float]]:
    """Embed many document chunks (used during ingestion)."""
    return _embed([DOC_PREFIX + c for c in chunks])


def embed_query(question: str) -> list[float]:
    """Embed one user question (used during retrieval)."""
    return _embed([QUERY_PREFIX + question])[0]
