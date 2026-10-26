"""[ISSUE version] database.py - STAGE 6 of ingestion: store chunks + vectors in Qdrant.

Qdrant vocabulary:
  collection = a table of points
  point      = one row: an ID, a vector, and a payload (the chunk text and its labels)
"""
import uuid

from qdrant_client import models

from config import COLLECTION_ISSUE as COLLECTION, VECTOR_SIZE, qdrant


def store_chunks(chunks, vectors, reset=False):
    if reset and qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)  # start from an empty database
        print("  database emptied first")

    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )

    points = []
    for chunk, vector in zip(chunks, vectors):

        # ===== ISSUE X3: a brand-new random ID on every run, so running ingestion twice stores everything twice =====
        point_id = str(uuid.uuid4())

        # payload = everything we know about the chunk (its text + the labels from FIX V1)
        points.append(models.PointStruct(id=point_id, vector=vector, payload=chunk))

    # upsert = update if the ID already exists, insert if it does not
    qdrant.upsert(COLLECTION, points=points)
    print(f"  sent {len(points)} points to Qdrant")
    print(f"  the database now holds {qdrant.count(COLLECTION).count} points")


def show_stored():
    """Print what is really inside Qdrant: ID, payload and vector of every point."""
    points, _ = qdrant.scroll(COLLECTION, limit=200, with_payload=True, with_vectors=True)
    points.sort(key=lambda p: (p.payload["source"], p.payload["chunk_no"]))
    for p in points:
        labels = {k: v for k, v in p.payload.items() if k not in ("text", "source", "chunk_no", "path", "section", "context")}
        print(f"\npoint id={str(p.id)[:8]}...  {p.payload['source']} #{p.payload['chunk_no']}")
        print(f"  labels  : {labels if labels else '(none)'}")
        print(f"  context : {p.payload['context'] or '(none)'}")
        print(f"  vector  : {len(p.vector)} numbers, starting {[round(x, 3) for x in p.vector[:4]]} ...")
        for line in p.payload["text"].splitlines():
            print("  text    | " + line)
