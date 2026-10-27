"""
query.py  -  the RAG half: find the chunks of OUR documents that match a question.

    question -> embed (same model as ingestion) -> search Qdrant -> top chunks

Test it alone:   python query.py "How many casual leaves do I get?"
"""
import sys
from config import TOP_K
from ingest.embed import embed_query
from ingest.vector_store import get_client, search


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return [{"text":..., "source":..., "score":...}, ...] best match first."""
    query_vector = embed_query(question)
    client = get_client()
    hits = search(client, query_vector, top_k)
    return [
        {"text": h.payload["text"], "source": h.payload["source"], "score": round(h.score, 3)}
        for h in hits
    ]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many casual leaves do I get?"
    print(f"Question: {question}\n")
    for i, hit in enumerate(retrieve(question), 1):
        print(f"#{i}  score={hit['score']}  source={hit['source']}")
        print(f"    {hit['text']}\n")
