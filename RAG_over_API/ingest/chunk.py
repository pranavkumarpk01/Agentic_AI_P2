"""
chunk.py  -  STEP 3 of ingestion: split long text into small pieces (chunks).

Why? An embedding of a whole document is too blurry. Small chunks give
precise matches, and we only send the relevant chunks to the LLM.

Method: a sliding window of N words. Each chunk overlaps the previous one
a little, so a sentence cut at the edge still appears whole in one chunk.
"""
from config import CHUNK_WORDS, CHUNK_OVERLAP


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(words):  # last window reached the end
            break
    return chunks
