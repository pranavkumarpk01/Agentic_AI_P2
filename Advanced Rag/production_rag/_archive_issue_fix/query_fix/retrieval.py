"""[FIX version] retrieval.py - STAGE 8 of querying: find the chunks closest in meaning to the question.

Steps: embed the question -> ask Qdrant for the nearest vectors (only allowed versions) -> get chunks back.
After reranking, expand_with_neighbours() adds the chunks around each hit.
"""
from google.genai import types
from qdrant_client import models

from config import COLLECTION_FIX as COLLECTION, EMBED_MODEL, VECTOR_SIZE, gemini, qdrant


def label_of(info):
    """A short label for printing: which document, which version, is it in force?"""
    if "status" in info:
        return f"{info['doc_key']} v{info['version']} ({info['status']})"
    return "(no version info stored)"


def show_hits(hits):
    """Print the hits: rank, score, file, version label and the chunk text."""
    for rank, hit in enumerate(hits, start=1):
        info = hit.payload
        print(f"\n  [{rank}] score={hit.score:.3f}  {info['source']}  {label_of(info)}")
        if info.get("context"):
            print(f"      context: {info['context']}")
        for line in info["text"].splitlines():
            print("      | " + line)


def retrieve(search_text, n, include_history=False):
    # 1. Turn the question into a vector. Same model as ingestion, but "RETRIEVAL_QUERY" this time.
    result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=[search_text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=VECTOR_SIZE),
    )
    question_vector = result.embeddings[0].values

    # ===== FIX V2: "current only" is mandatory. A history question can opt in to old versions with include_history =====
    allowed = ["current", "superseded"] if include_history else ["current"]
    search_filter = models.Filter(must=[
        models.FieldCondition(key="status", match=models.MatchAny(any=allowed))])

    # 2. Ask Qdrant for the n closest chunks that pass the filter.
    hits = qdrant.query_points(
        COLLECTION, query=question_vector, query_filter=search_filter, limit=n
    ).points

    print(f"  Qdrant returned {len(hits)} candidates, best first:")
    show_hits(hits)
    return hits


def expand_with_neighbours(hits):
    """Turn the final hits into the list of passages the model will read."""

    # ===== FIX C2: also fetch the chunk before and after each hit, so a fact on a boundary is not lost =====
    found = {}
    for hit in hits:
        here = hit.payload
        around, _ = qdrant.scroll(
            COLLECTION, limit=10, with_payload=True,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=here["source"])),
                models.FieldCondition(key="chunk_no", range=models.Range(gte=here["chunk_no"] - 1, lte=here["chunk_no"] + 1))]))
        for point in around:
            found[(point.payload["source"], point.payload["chunk_no"])] = point.payload
    passages = [found[key] for key in sorted(found)]  # in reading order

    hit_keys = {(hit.payload["source"], hit.payload["chunk_no"]) for hit in hits}
    print(f"  the model will read {len(passages)} passages:")
    for passage in passages:
        role = "found by search" if (passage["source"], passage["chunk_no"]) in hit_keys else "neighbour"
        print(f"\n  {passage['source']} #{passage['chunk_no']}  ({role})")
        for line in passage["text"].splitlines():
            print("      | " + line)
    return passages
