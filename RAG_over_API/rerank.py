"""
rerank.py  -  a second, more careful look at the search results.

WHY?
  Vector search is FAST but approximate: question and chunk are embedded
  separately, then compared. It can put a slightly-off chunk at #1.
  A re-ranker (cross-encoder) reads the question and a chunk TOGETHER and gives
  a much more accurate relevance score. It is too slow to run on everything,
  so we only run it on a short list.

HOW (2 stages):
  Stage 1  Qdrant vector search -> top FETCH_K candidates (wide net, e.g. 8)
  Stage 2  cross-encoder re-scores each (question, chunk) pair
           -> sort by the new score -> keep the best TOP_K (e.g. 3)

The model is open source (runs on your laptop, no API key).
First run downloads it (about 90 MB).

Run the demo:        python rerank.py
Your own question:   python rerank.py "your question here"
"""
import sys
from sentence_transformers import CrossEncoder
from config import TOP_K, FETCH_K, RERANK_MODEL
from query import retrieve

_model = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:                       # load once, reuse after
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(question: str, hits: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Re-score hits with the cross-encoder and return the best top_k."""
    pairs = [(question, h["text"]) for h in hits]      # model reads question + chunk together
    scores = get_model().predict(pairs)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = round(float(score), 3)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What happens to my unused leave at the end of the year?"
    print(f"Question: {question}\n")

    # Stage 1: vector search with a wide net
    candidates = retrieve(question, top_k=FETCH_K)
    print(f"STAGE 1 - vector search, top {FETCH_K} candidates (in vector-search order):")
    for i, h in enumerate(candidates, 1):
        print(f"   #{i} [{h['score']}]  {h['text'][:75]}...")

    # Stage 2: re-rank and keep the best few
    best = rerank(question, candidates)
    print(f"\nSTAGE 2 - after re-ranking, best {TOP_K} (these go to the LLM):")
    for i, h in enumerate(best, 1):
        print(f"   #{i} [{h['rerank_score']}]  {h['text'][:75]}...")
