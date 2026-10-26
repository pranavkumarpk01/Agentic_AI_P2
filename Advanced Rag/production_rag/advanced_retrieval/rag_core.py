"""
rag_core.py - the reusable building blocks shared by the demo scripts.

Each demo script explains ONE technique step by step in its own file.
The pieces that every script needs (load documents, chunk text, embed,
keyword search, vector search, fusion) live here so the scripts stay short.

Read this file top to bottom once - every function is small on purpose.
"""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).parent / "data" / "docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs locally


# ---------------------------------------------------------------------------
# 1. Loading documents + metadata
# ---------------------------------------------------------------------------
def parse_document(path):
    """Split a file into (metadata dict, body text).

    Every file in data/docs starts with a header between two '---' lines:
        ---
        department: HR
        region: India
        year: 2025
        ---
        ...body text...
    """
    raw = Path(path).read_text(encoding="utf-8")
    _, header, body = raw.split("---", 2)

    metadata = {}
    for line in header.strip().splitlines():
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    metadata["year"] = int(metadata["year"])   # numbers must be numbers for $gte / $lte
    metadata["source"] = Path(path).name        # where this chunk came from
    return metadata, body.strip()


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def split_sentences(text):
    """Very small sentence splitter: break after . ! ? when a new sentence starts."""
    text = " ".join(text.split())                       # collapse new lines / extra spaces
    # split after . ! ?  - but NOT after a numbered heading like "2." (the (?<!\d\.) part)
    sentences = re.split(r"(?<!\d\.)(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def sentence_chunks(text, max_chars=350, overlap=1):
    """Group whole sentences into chunks of at most ~max_chars characters.

    - Never cuts a sentence in half (unlike fixed-size chunking).
    - `overlap` = how many sentences from the end of one chunk are repeated
      at the start of the next one, so context is not lost at the border.
    """
    sentences = split_sentences(text)
    chunks, current = [], []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))           # close the current chunk
            current = current[-overlap:] if overlap else []
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def load_knowledge_base(max_chars=200):
    """Read every document, chunk it, and attach the document's metadata to each chunk."""
    chunks = []
    for path in sorted(DOCS_DIR.rglob("*.txt")):
        metadata, body = parse_document(path)
        # Paragraph first (blank line), then sentences inside the paragraph.
        for paragraph in body.split("\n\n"):
            for text in sentence_chunks(paragraph, max_chars=max_chars, overlap=0):
                chunks.append(
                    {
                        "id": f"{path.stem}#{len(chunks)}",
                        "text": text,
                        "metadata": dict(metadata),   # a COPY for every chunk
                    }
                )
    return chunks


# ---------------------------------------------------------------------------
# 3. Embeddings (semantic / vector side)
# ---------------------------------------------------------------------------
_model = None


def embed(texts):
    """Turn a list of strings into a matrix of unit-length vectors (one row per text)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)   # downloaded once (~90 MB), then cached
    return np.asarray(_model.encode(texts, normalize_embeddings=True))


def vector_search(query, chunk_vectors, top_k=5):
    """Return [(chunk_index, cosine_similarity), ...] best first.

    Vectors are normalised, so a dot product IS the cosine similarity.
    """
    query_vector = embed([query])[0]
    similarities = chunk_vectors @ query_vector
    best = np.argsort(-similarities)[:top_k]
    return [(int(i), float(similarities[i])) for i in best]


# ---------------------------------------------------------------------------
# 4. BM25 (keyword side) - the same algorithm Elasticsearch/OpenSearch use
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "i", "my", "me", "we", "you", "your",
    "do", "does", "did", "can", "could", "how", "what", "when", "where", "which",
    "who", "why", "to", "of", "in", "on", "for", "from", "with", "and", "or", "at",
    "by", "it", "this", "that", "be", "will", "about", "get", "any",
}


def tokenize(text):
    """lowercase -> split into words (keeps codes like err-4012 together) -> drop stopwords."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(t) for t in texts]
        self.n_docs = len(texts)
        self.avg_len = sum(len(d) for d in self.doc_tokens) / self.n_docs
        # document frequency: in how many chunks does each word appear?
        self.df = Counter()
        for tokens in self.doc_tokens:
            self.df.update(set(tokens))

    def idf(self, word):
        """Rare words get a high weight, common words a low weight."""
        df = self.df.get(word, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query):
        query_words = tokenize(query)
        all_scores = []
        for tokens in self.doc_tokens:
            counts = Counter(tokens)
            length_norm = 1 - self.b + self.b * len(tokens) / self.avg_len
            score = 0.0
            for word in query_words:
                tf = counts.get(word, 0)
                if tf:
                    score += self.idf(word) * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
            all_scores.append(score)
        return all_scores

    def search(self, query, top_k=5):
        """Return [(chunk_index, bm25_score), ...] - only chunks that share a word with the query."""
        s = self.scores(query)
        ranked = sorted(range(len(s)), key=lambda i: -s[i])
        return [(i, s[i]) for i in ranked if s[i] > 0][:top_k]


# ---------------------------------------------------------------------------
# 5. Reciprocal Rank Fusion - merge several ranked lists into one
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(ranked_lists, k=60):
    """Each list is [chunk_index, ...] best first.  score = sum of 1 / (k + rank)."""
    fused = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_index in enumerate(ranked, start=1):
            fused[chunk_index] += 1 / (k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])


class HybridRetriever:
    """BM25 + vector search + RRF, packaged for reuse (explained in 03_hybrid_retrieval.py)."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25([c["text"] for c in chunks])
        self.vectors = embed([c["text"] for c in chunks])

    def search(self, query, top_k=3, candidates=10):
        keyword = [i for i, _ in self.bm25.search(query, top_k=candidates)]
        semantic = [i for i, _ in vector_search(query, self.vectors, top_k=candidates)]
        return reciprocal_rank_fusion([keyword, semantic])[:top_k]
