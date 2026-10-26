"""
05 - STORE EVERYTHING IN QDRANT AND LOOK INSIDE
===============================================
Scripts 01-04 kept chunks in Python lists so you could see every step.
Here the same chunks go into the real vector database and we look at them there:

  document  ->  chunk + metadata  ->  vector  ->  Qdrant POINT (id + vector + payload)

Then we repeat the metadata-filter search from script 02, but INSIDE Qdrant.

Needs: run 01_table_aware_chunking.py first (it writes output/chunks_table_aware.json).
Qdrant: uses QDRANT_URL from production_rag/.env (Docker, with a dashboard).
        If Docker is not running, it falls back to a local folder (advanced_retrieval/qdrant_local).

Run:  python 05_store_in_qdrant.py            (Enter between steps)
      python 05_store_in_qdrant.py --no-pause
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from display import console, fixed, note, pause, problem, short, step, table, title
from rag_core import embed, load_knowledge_base

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")            # QDRANT_URL from production_rag/.env
TABLE_CHUNKS_FILE = HERE / "output" / "chunks_table_aware.json"
COLLECTION = "advanced_retrieval"            # separate from policies_issue / policies_fix
VECTOR_SIZE = 384                            # all-MiniLM-L6-v2 gives 384 numbers per text

title("05 - Store everything in Qdrant and look inside")

# ---------------------------------------------------------------------------
step(1, "Connect to Qdrant")
# ---------------------------------------------------------------------------
url = os.getenv("QDRANT_URL")
qdrant = None
if url:
    try:
        qdrant = QdrantClient(url=url, timeout=5)
        qdrant.get_collections()                      # fails fast if Docker is not running
        console.print(f"Connected to Qdrant in Docker at [bold]{url}[/bold]")
        console.print(f"Dashboard: [bold]{url.rstrip('/')}/dashboard[/bold]  -> collection '{COLLECTION}'")
    except Exception:
        console.print(f"[yellow]Qdrant at {url} is not reachable (is Docker running?). Using a local folder.[/yellow]")
        qdrant = None
if qdrant is None:
    qdrant = QdrantClient(path=str(HERE / "qdrant_local"))   # Qdrant inside this program, saved to a folder
    console.print("Using local Qdrant folder: advanced_retrieval/qdrant_local (no dashboard)")
pause()

# ---------------------------------------------------------------------------
step(2, "Collect the chunks: policy documents (02-04) + handbook PDF chunks (01)")
# ---------------------------------------------------------------------------
if not TABLE_CHUNKS_FILE.exists():
    raise SystemExit("output/chunks_table_aware.json not found. Run: python 01_table_aware_chunking.py --no-pause")

chunks = []

# a) the 6 policy documents: text + metadata from their --- header (same as script 02)
for c in load_knowledge_base():
    chunks.append({"text": c["text"], **c["metadata"], "chunk_type": "text"})

# b) the handbook chunks that script 01 produced (table rows, whole table, sentence text)
handbook_meta = {"title": "Employee Handbook", "department": "HR", "region": "India",
                 "year": 2025, "doc_type": "handbook", "source": "employee_handbook.pdf"}
for c in json.loads(TABLE_CHUNKS_FILE.read_text()):
    chunks.append({"text": c["text"], **handbook_meta,
                   "chunk_type": c["metadata"]["type"], "page": c["metadata"]["page"]})

table(
    f"{len(chunks)} chunks ready to store",
    ["#", "source", "chunk_type", "region", "year", "text"],
    [[i, c["source"], c["chunk_type"], c["region"], c["year"], short(c["text"], 55)] for i, c in enumerate(chunks)],
)
pause()

# ---------------------------------------------------------------------------
step(3, "Create the collection (a 'table' of points)")
# ---------------------------------------------------------------------------
if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)             # start clean every run
qdrant.create_collection(
    COLLECTION,
    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
)
note(f"collection = '{COLLECTION}'\n"
     f"every vector must have exactly {VECTOR_SIZE} numbers\n"
     "distance = COSINE  (same similarity we computed by hand in scripts 02 and 03)")
pause()

# ---------------------------------------------------------------------------
step(4, "Turn every chunk into a POINT = id + vector + payload")
# ---------------------------------------------------------------------------
console.print("Embedding all chunks...")
vectors = embed([c["text"] for c in chunks])

points = []
for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
    points.append(models.PointStruct(
        id=i,                           # a simple number is enough here
        vector=vector.tolist(),         # 384 numbers = the MEANING of the text
        payload=chunk,                  # text + every metadata field = what we FILTER on and SHOW
    ))

example = next(i for i, c in enumerate(chunks) if "Grade: L3" in c["text"])
console.print("[bold]One point, exactly as it is sent to Qdrant:[/bold]")
console.print(json.dumps({
    "id": points[example].id,
    "vector": f"[{', '.join(f'{x:.3f}' for x in vectors[example][:6])}, ... 384 numbers]",
    "payload": points[example].payload,
}, indent=2))

qdrant.upsert(COLLECTION, points=points)             # upsert = insert, or overwrite if the id exists
console.print(f"\nStored [bold]{qdrant.count(COLLECTION).count}[/bold] points in '{COLLECTION}'")
pause()

# ---------------------------------------------------------------------------
step(5, "Look inside the database (read the points back)")
# ---------------------------------------------------------------------------
stored, _ = qdrant.scroll(COLLECTION, limit=100, with_payload=True, with_vectors=True)
stored.sort(key=lambda p: p.id)
table(
    "What Qdrant holds",
    ["id", "source", "type", "region", "year", "vector (first 3 of 384)", "text"],
    [[p.id, p.payload["source"], p.payload["chunk_type"], p.payload["region"], p.payload["year"],
      ", ".join(f"{x:.2f}" for x in p.vector[:3]), short(p.payload["text"], 45)] for p in stored],
)
note("Each row is a POINT. The vector is used to search by meaning.\n"
     "The payload (source, region, year, text...) is used to filter and to show the answer's source.")
pause()


# ---------------------------------------------------------------------------
def qdrant_search(query, search_filter=None, limit=4):
    hits = qdrant.query_points(
        COLLECTION,
        query=embed([query])[0].tolist(),   # the question as a vector
        query_filter=search_filter,         # None = search every point
        limit=limit,
    ).points
    return hits


def valid_for_user(payload):
    return payload["region"] in ("India", "Global") and payload["year"] == 2025


def show_hits(heading, hits):
    rows = []
    for h in hits:
        p = h.payload
        rows.append([f"{h.score:.3f}", p["source"], p["region"], p["year"], short(p["text"], 60),
                     "[green]ok[/green]" if valid_for_user(p) else "[red]WRONG for this user[/red]"])
    table(heading, ["score", "source", "region", "year", "text", "India user, 2025"], rows)
    return sum(not valid_for_user(h.payload) for h in hits)


# ---------------------------------------------------------------------------
step(6, "Metadata filtering INSIDE Qdrant (same idea as script 02)")
# ---------------------------------------------------------------------------
QUERY = "How many paid leave days do I get?"
console.print(f"User: India employee, 2025   Query: [bold]{QUERY}[/bold]")

wrong = show_hits("BEFORE - no filter", qdrant_search(QUERY))
if wrong:
    problem(f"{wrong} of the results are wrong for this user. Without a filter Qdrant ranks by meaning only.")
else:
    note("No wrong result this time - but nothing in the search PREVENTS an old or US policy from coming back.")

user_filter = models.Filter(must=[
    models.FieldCondition(key="region", match=models.MatchAny(any=["India", "Global"])),
    models.FieldCondition(key="year", match=models.MatchValue(value=2025)),
])
console.print("\nfilter = region in [India, Global]  AND  year == 2025")
show_hits("AFTER - with Qdrant filter", qdrant_search(QUERY, user_filter))
fixed("Qdrant applies the filter while searching, so only valid points can be returned.")
pause()

# ---------------------------------------------------------------------------
step(7, "How the handbook TABLE from script 01 lives in the database")
# ---------------------------------------------------------------------------
row_filter = models.Filter(must=[
    models.FieldCondition(key="source", match=models.MatchValue(value="employee_handbook.pdf")),
    models.FieldCondition(key="chunk_type", match=models.MatchValue(value="table_row")),
])
rows, _ = qdrant.scroll(COLLECTION, scroll_filter=row_filter, limit=20, with_payload=True)
rows.sort(key=lambda p: p.id)
table("Table 1 = 5 points, one per row (filter: chunk_type = table_row)", ["id", "payload text"],
      [[p.id, p.payload["text"]] for p in rows])

QUERY_2 = "How many sick leave days does an L3 employee get?"
hits = qdrant_search(QUERY_2, row_filter, limit=3)
table(f"Vector search in those rows: {QUERY_2}", ["score", "text"],
      [[f"{h.score:.3f}", h.payload["text"]] for h in hits])
if "Grade: L3" in hits[0].payload["text"]:
    fixed("The L3 row comes first, and it carries its headers: the answer '12 days' is in the payload.")
else:
    note("The 5 rows look almost the same to the embedding model (only 'L3' vs 'L4' differs),\n"
         "so the top row may not be L3. That is exactly why script 03 adds keyword search:\n"
         "'L3' is an exact token that BM25 matches.")

qdrant.close()
