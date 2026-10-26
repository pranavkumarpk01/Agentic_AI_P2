"""[ISSUE version] ingest/main.py - runs the whole ingestion pipeline and prints what every stage produced.

    python -m ingest_issue.main --reset     # empty the database first, then ingest (the normal demo command)
    python -m ingest_issue.main             # ingest again WITHOUT emptying (used for the duplicates demo)
"""
import argparse

from config import COLLECTION_ISSUE as COLLECTION, qdrant
from ingest_issue.chunking import chunk_documents
from ingest_issue.database import show_stored, store_chunks
from ingest_issue.embedding import embed_chunks
from ingest_issue.fetch import fetch_documents
from ingest_issue.quality_gate import check_chunks
from ingest_issue.preprocessing import preprocess_documents


def banner(text):
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="empty the database before storing")
args = parser.parse_args()

banner("STAGE 1 - FETCH: read the files and note where they came from")
docs = fetch_documents()

banner("STAGE 2 - PREPROCESS: clean the text")
docs = preprocess_documents(docs)

banner("STAGE 3 - CHUNK: cut the text into pieces")
chunks = chunk_documents(docs)

banner("STAGE 4 - QUALITY GATE: are all the numbers still there?")
chunks = check_chunks(chunks)

banner("STAGE 5 - EMBED: turn every chunk into a vector")
vectors = embed_chunks(chunks)

banner("STAGE 6 - STORE: save chunks and vectors in Qdrant")
store_chunks(chunks, vectors, reset=args.reset)

banner("WHAT QDRANT NOW HOLDS")
show_stored()

banner("SUMMARY")
whole = sum(1 for c in chunks if c["text"].rstrip().endswith((".", "!", "?")))
print(f"  chunks made                    : {len(chunks)}")
print(f"  characters stored              : {sum(len(c['text']) for c in chunks)}")
print(f"  chunks that end at a full stop : {whole} of {len(chunks)}")
print(f"  points in Qdrant               : {qdrant.count(COLLECTION).count}")

qdrant.close()  # close the connection politely
