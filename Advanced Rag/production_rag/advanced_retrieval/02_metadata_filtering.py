"""
02 - METADATA FILTERING
=======================
Situation: an employee in INDIA asks in 2025 "How many paid leave days do I get?"
The knowledge base has the India 2025 policy, an OLD India 2024 policy and a
US policy. They all look almost the same to a vector search.

  BEFORE  search everything -> outdated / wrong-country chunks show up.
  AFTER   attach metadata to every chunk at ingestion time, then FILTER
          on metadata first and rank by similarity second.

Run:  python 02_metadata_filtering.py            (Enter between steps)
      python 02_metadata_filtering.py --no-pause
"""

import json
from pathlib import Path

from display import console, fixed, note, pause, problem, short, step, table, title
from rag_core import DOCS_DIR, embed, sentence_chunks

title("02 - Metadata filtering")

# ---------------------------------------------------------------------------
step(1, "Look at ONE raw document - it has a metadata header on top")
# ---------------------------------------------------------------------------
sample_path = DOCS_DIR / "hr" / "leave_policy_india_2025.txt"
console.print(f"[dim]{sample_path.relative_to(DOCS_DIR.parent.parent)}[/dim]")
console.print(sample_path.read_text())
pause()


# ---------------------------------------------------------------------------
step(2, "Parse the header into a Python dict  (this is the 'metadata')")
# ---------------------------------------------------------------------------
def parse_document(path):
    raw = Path(path).read_text(encoding="utf-8")
    _, header, body = raw.split("---", 2)        # ['', header, body]

    metadata = {}
    for line in header.strip().splitlines():     # "region: India"
        key, value = line.split(":", 1)          # -> "region", " India"
        metadata[key.strip()] = value.strip()

    metadata["year"] = int(metadata["year"])     # store numbers as numbers
    metadata["source"] = Path(path).name         # remember where it came from
    return metadata, body.strip()


metadata, body = parse_document(sample_path)
console.print("metadata =", metadata)
console.print("body     =", repr(short(body, 120)))
note("In real systems metadata comes from the folder path, the file properties,\n"
     "the source system (SharePoint, Confluence, DB) or is extracted by an LLM.")
pause()

# ---------------------------------------------------------------------------
step(3, "Chunk the body and ATTACH the metadata to every chunk")
# ---------------------------------------------------------------------------
plain_chunks = sentence_chunks(body, max_chars=200, overlap=0)

console.print("[bold]BEFORE - a chunk is only text:[/bold]")
console.print(json.dumps(plain_chunks[0], indent=2))

chunk_with_metadata = {"id": "leave_policy_india_2025#0", "text": plain_chunks[0], "metadata": dict(metadata)}
console.print("\n[bold]AFTER - a chunk is text + metadata:[/bold]")
console.print(json.dumps(chunk_with_metadata, indent=2))
pause()

# ---------------------------------------------------------------------------
step(4, "Do the same for EVERY document and build the vector index")
# ---------------------------------------------------------------------------
chunks = []
for path in sorted(DOCS_DIR.rglob("*.txt")):
    metadata, body = parse_document(path)
    for paragraph in body.split("\n\n"):
        for text in sentence_chunks(paragraph, max_chars=200, overlap=0):
            chunks.append({"id": f"{path.stem}#{len(chunks)}", "text": text, "metadata": dict(metadata)})

console.print("Embedding chunks (first run downloads the model)...")
vectors = embed([c["text"] for c in chunks])      # one row per chunk

table(
    f"Knowledge base: {len(chunks)} chunks",
    ["#", "department", "region", "year", "text"],
    [[i, c["metadata"]["department"], c["metadata"]["region"], c["metadata"]["year"], short(c["text"], 60)]
     for i, c in enumerate(chunks)],
)
console.print(f"vectors shape = {vectors.shape}  (chunks x dimensions)")
pause()


def verdict(metadata, user_region="India", user_year=2025):
    if metadata["region"] not in (user_region, "Global"):
        return f"WRONG COUNTRY ({metadata['region']})"
    if metadata["year"] != user_year:
        return f"OUTDATED ({metadata['year']})"
    return "ok"


def show_results(heading, results):
    rows = []
    for index, score in results:
        m = chunks[index]["metadata"]
        v = verdict(m)
        colour = "green" if v == "ok" else "red"
        rows.append([f"{score:.3f}", m["region"], m["year"], short(chunks[index]["text"], 70), f"[{colour}]{v}[/{colour}]"])
    table(heading, ["similarity", "region", "year", "text", "allowed for this user?"], rows)


def search(query, candidate_indexes, top_k=4):
    """Rank only the candidate chunks by cosine similarity."""
    query_vector = embed([query])[0]
    scored = [(i, float(vectors[i] @ query_vector)) for i in candidate_indexes]
    return sorted(scored, key=lambda item: -item[1])[:top_k]


QUERY = "How many paid leave days do I get?"
console.print(f"\nUser: [bold]India employee, 2025[/bold]   Query: [bold]{QUERY}[/bold]")

# ---------------------------------------------------------------------------
step(5, "BEFORE - search ALL chunks, no filter")
# ---------------------------------------------------------------------------
before = search(QUERY, candidate_indexes=range(len(chunks)))
show_results("Without metadata filter", before)
bad = [verdict(chunks[i]["metadata"]) for i, _ in before if verdict(chunks[i]["metadata"]) != "ok"]
if bad:
    problem(f"{len(bad)} of the top {len(before)} chunks are not valid for this user: {', '.join(bad)}.\n"
            "To the embedding model '18 days in India' (2024) and '15 days in the US' look\n"
            "just as relevant as '24 days in India' (2025). The LLM may give a wrong number.")
else:
    note("This time no wrong chunk made the top results - but nothing PREVENTS it.\n"
         "Similarity alone does not know the user's country or which year is current.")
pause()

# ---------------------------------------------------------------------------
step(6, "AFTER - filter by metadata FIRST, then rank by similarity")
# ---------------------------------------------------------------------------
def matches(metadata, where):
    """Return True if the chunk's metadata satisfies every condition in `where`.

    where examples (same style as Chroma / Pinecone):
        {"region": "India"}                       -> equals
        {"year": {"$gte": 2025}}                  -> greater or equal
        {"region": {"$in": ["India", "Global"]}}  -> one of a list
    """
    for key, condition in where.items():
        value = metadata.get(key)
        if isinstance(condition, dict):                      # operator form
            for operator, target in condition.items():
                if operator == "$eq" and not value == target:
                    return False
                if operator == "$ne" and not value != target:
                    return False
                if operator == "$gte" and not value >= target:
                    return False
                if operator == "$lte" and not value <= target:
                    return False
                if operator == "$in" and value not in target:
                    return False
        elif value != condition:                             # plain equality
            return False
    return True


# The filter comes from the LOGGED-IN USER's profile - not from the question text.
where = {"region": {"$in": ["India", "Global"]}, "year": 2025}
console.print(f"where = {where}")

candidates = [i for i, c in enumerate(chunks) if matches(c["metadata"], where)]
console.print(f"Filter kept [bold]{len(candidates)}[/bold] of {len(chunks)} chunks -> only these are ranked.")

after = search(QUERY, candidate_indexes=candidates)
show_results("With metadata filter", after)
fixed("Only India/Global 2025 chunks can be returned - the answer will be 24 days.")
pause()

# ---------------------------------------------------------------------------
step(7, "Another filter - restrict to one department")
# ---------------------------------------------------------------------------
QUERY_2 = "Something expired and I cannot connect"
for heading, where_2 in [("No filter", {}), ("department = IT", {"department": "IT"})]:
    candidate_ids = [i for i, c in enumerate(chunks) if matches(c["metadata"], where_2)]
    results = search(QUERY_2, candidate_ids)
    table(
        f"{QUERY_2!r}  ->  {heading}",
        ["similarity", "department", "text"],
        [[f"{s:.3f}", chunks[i]["metadata"]["department"], short(chunks[i]["text"], 80)] for i, s in results],
    )

note(
    "Same idea in a real vector database:\n"
    "  Chroma  : collection.query(query_texts=['How many leave days?'], n_results=3,\n"
    "                where={'$and': [{'region': {'$in': ['India','Global']}}, {'year': 2025}]})\n"
    "  Pinecone: index.query(vector=v, top_k=3, filter={'region': {'$in': [...]}, 'year': 2025})"
)
