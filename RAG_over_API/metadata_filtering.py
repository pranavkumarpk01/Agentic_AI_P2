"""
metadata_filtering.py  -  search only the chunks that MATCH a condition.

WHY?
  A vague question like "How many days?" matches leave days, notice-period days,
  reimbursement deadlines... Similarity search alone returns a mix.
  If we already know the topic (category = leave), we can tell Qdrant:
  "only look at chunks whose metadata says category = leave".

HOW (3 steps):
  1. At ingestion each chunk was stored with metadata  (ingest/metadata.py):
         {"text": "...", "section": "Casual and Sick Leave", "category": "leave", "section_no": 2}
  2. At query time we build a Filter from the conditions we want.
  3. Qdrant applies the filter, THEN ranks the remaining chunks by similarity.

Run the demo:        python metadata_filtering.py
Your own question:   python metadata_filtering.py "How many days?" leave
"""
import sys
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from config import TOP_K
from ingest.embed import embed_query
from ingest.vector_store import get_client, search


def build_filter(category=None, section_no=None):
    """Turn simple Python values into a Qdrant Filter. All conditions must be true (AND)."""
    conditions = []
    if category is not None:
        if isinstance(category, list):            # category=["leave", "exit"] -> either one
            conditions.append(FieldCondition(key="category", match=MatchAny(any=category)))
        else:                                     # category="leave" -> exactly this
            conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if section_no is not None:
        conditions.append(FieldCondition(key="section_no", match=MatchValue(value=section_no)))
    return Filter(must=conditions) if conditions else None


def retrieve_filtered(question: str, category=None, section_no=None, top_k: int = TOP_K) -> list[dict]:
    """Same as query.retrieve(), plus an optional metadata filter."""
    query_vector = embed_query(question)
    hits = search(get_client(), query_vector, top_k, query_filter=build_filter(category, section_no))
    return [
        {
            "text": h.payload["text"],
            "section": h.payload["section"],
            "category": h.payload["category"],
            "score": round(h.score, 3),
        }
        for h in hits
    ]


def show(title: str, hits: list[dict]) -> None:
    print(f"--- {title} ---")
    if not hits:
        print("   (no chunks match this filter)")
    for h in hits:
        print(f"   [{h['score']}] ({h['category']} / {h['section']})  {h['text'][:70]}...")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:                         # python metadata_filtering.py "question" [category]
        question = sys.argv[1]
        category = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Question: {question}\n")
        show("NO filter", retrieve_filtered(question))
        show(f"FILTER category = {category}", retrieve_filtered(question, category=category))
    else:                                         # built-in demo: same question, different filters
        question = "How many days?"
        print(f"Question: {question}   (deliberately vague)\n")
        show("NO filter (mixed topics)", retrieve_filtered(question))
        show("FILTER category = leave", retrieve_filtered(question, category="leave"))
        show("FILTER category = exit", retrieve_filtered(question, category="exit"))
        show("FILTER category = leave OR exit", retrieve_filtered(question, category=["leave", "exit"]))
