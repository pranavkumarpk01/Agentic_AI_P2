"""
ingest.py  -  runs the whole ingestion pipeline, step by step.

    fetch  ->  split into sections + tag metadata  ->  preprocess  ->  chunk
           ->  embed  ->  store in Qdrant (vector + text + metadata)

Run from the project root:   python -m ingest.ingest
"""
from ingest.fetch import fetch_documents
from ingest.metadata import split_sections
from ingest.preprocess import preprocess
from ingest.chunk import chunk_text
from ingest.embed import embed_documents
from ingest.vector_store import get_client, create_collection, store_chunks


def main():
    # 1. fetch
    docs = fetch_documents("data")
    if not docs:
        raise SystemExit("No .txt/.md/.pdf files found in the data/ folder.")
    print(f"[1] Fetched {len(docs)} document(s): {[d['source'] for d in docs]}")

    # 2 + 3. split into sections (adds metadata), preprocess, chunk
    chunks = []
    for doc in docs:
        for sec in split_sections(doc["text"]):
            # put the section title in front, it helps both search and re-ranking
            clean = preprocess(f"{sec['section']}. {sec['body']}")
            for piece in chunk_text(clean):
                chunks.append({
                    "text": piece,
                    "source": doc["source"],
                    # ---- METADATA (used by metadata_filtering.py) ----
                    "section_no": sec["section_no"],
                    "section": sec["section"],
                    "category": sec["category"],
                })
    if not chunks:
        raise SystemExit("No '## ' sections found. Metadata splitting needs headings like: ## 1. Working Hours")
    print(f"[2-3] Split into sections, tagged metadata, made {len(chunks)} chunks")
    print(f"      example metadata: { {k: v for k, v in chunks[0].items() if k != 'text'} }")

    # 4. embed
    vectors = embed_documents([c["text"] for c in chunks])
    print(f"[4] Created {len(vectors)} embeddings (each has {len(vectors[0])} numbers)")

    # 5. store
    client = get_client()
    create_collection(client, vector_size=len(vectors[0]))
    store_chunks(client, chunks, vectors)
    print(f"[5] Stored {len(chunks)} points in Qdrant. Ingestion complete!")


if __name__ == "__main__":
    main()
