"""
vector_store.py  -  STEP 5 of ingestion: store vectors in Qdrant.

Qdrant runs in "local mode": it just saves files in the qdrant_data/ folder.
No server, no Docker. (Only ONE python process can open the folder at a time.)

Vocabulary:
  collection = like a table
  point      = like a row  ->  id + vector + payload
  payload    = the original text and metadata stored next to the vector
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import QDRANT_PATH, COLLECTION


def get_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def create_collection(client: QdrantClient, vector_size: int) -> None:
    """Create a fresh, empty collection (deletes the old one if present)."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def store_chunks(client: QdrantClient, chunks: list[dict], vectors: list[list[float]]) -> None:
    """chunks = [{"text": ..., "source": ...}], vectors = matching embeddings."""
    points = [
        PointStruct(id=i, vector=vec, payload=chunk)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)


def search(client: QdrantClient, query_vector: list[float], top_k: int, query_filter=None):
    """
    Return the top_k most similar chunks (each has .score and .payload).
    query_filter (optional) = a metadata filter; see metadata_filtering.py
    """
    result = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return result.points
