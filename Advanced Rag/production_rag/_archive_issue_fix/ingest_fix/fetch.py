"""[FIX version] fetch.py - STAGE 1 of ingestion: find the documents and read them.

Output: a list of documents (dicts). Every document has "text" and "source".
With the fixes it also has:
  - an identity and date: title, doc_key, version, effective_date  (FIX V1)
  - a status decided by code: current / superseded / upcoming       (FIX V1)
  - a kind: "text" or "table"                                       (FIX T1)
"""
import json
from pathlib import Path

import pdfplumber

from config import TODAY


def add_status(docs, today):
    """The code, not a person, decides which version is in force today."""
    in_force = {}  # doc_key -> the newest effective_date that has already started
    for doc in docs:
        if doc["effective_date"] <= today:
            in_force[doc["doc_key"]] = max(in_force.get(doc["doc_key"], ""), doc["effective_date"])
    for doc in docs:
        if doc["effective_date"] > today:
            doc["status"] = "upcoming"      # starts in the future
        elif doc["effective_date"] == in_force[doc["doc_key"]]:
            doc["status"] = "current"       # the newest version that has started
        else:
            doc["status"] = "superseded"    # an older version
    return docs


def read_pdf_plain(path):
    """The simple way: ask each page for its text, tables included."""
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def find_title(page_text):
    """The table title is the line like 'Table 1: Loan interest rates and fees'."""
    for line in page_text.splitlines():
        if line.startswith("Table ") and ":" in line:
            return line.split(":", 1)[1].strip()
    return "Table"


def read_pdf_with_tables(path):
    """Helper for FIX T1: find the tables, keep the other text apart, join split tables, turn rows into sentences."""
    outside_text, tables = [], []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            found = page.find_tables()
            rest = page
            for table in found:
                rest = rest.outside_bbox(table.bbox)          # keep only what is NOT inside a table
            page_text = rest.extract_text() or ""
            outside_text.append(page_text)
            for table in found:
                grid = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in table.extract()]
                header, rows = grid[0], grid[1:]
                if tables and tables[-1]["header"] == header:
                    tables[-1]["rows"] += rows                # same header again = the table continues: join, drop the repeated header
                else:
                    tables.append({"title": find_title(page_text), "header": header, "rows": rows})

    pieces = [{"text": "\n".join(outside_text), "kind": "text"}]
    for table in tables:
        # every row becomes one sentence: "Title - Column: value; Column: value."
        sentences = [
            f"{table['title']} - " + "; ".join(f"{col}: {value}" for col, value in zip(table["header"], row)) + "."
            for row in table["rows"]
        ]
        pieces.append({"text": f"## {table['title']}\n" + "\n".join(sentences), "kind": "table"})
    return pieces


def fetch_documents(folder="data/raw"):
    manifest = json.loads(Path("data/manifest.json").read_text())  # what the document system knows about each file
    docs = []
    for path in sorted(Path(folder).rglob("*")):  # rglob = look inside every sub-folder too
        if path.suffix not in (".md", ".pdf"):
            continue

        # ===== FIX V1: every file needs an identity and a date (doc_key, version, effective_date). No version information = not indexed =====
        if path.name not in manifest:
            print(f"  WARNING: {path.name} has no version information, so it is NOT indexed")
            continue
        info = {"source": path.name, "path": str(path), "department": path.parent.name, **manifest[path.name]}

        if path.suffix == ".pdf":

            # ===== FIX T1: find the tables, keep the other text apart, join split tables, turn every row into a sentence =====
            pieces = read_pdf_with_tables(path)

        else:
            pieces = [{"text": path.read_text(), "kind": "text"}]

        for piece in pieces:
            docs.append({**info, **piece})

    # ===== FIX V1: part 2 - the newest version whose date has passed is current, older are superseded, future are upcoming =====
    docs = add_status(docs, TODAY)

    for doc in docs:
        labels = {k: v for k, v in doc.items() if k not in ("text", "path")}
        print(f"  {doc['source']}  [{doc['kind']}]  {len(doc['text'])} characters")
        print(f"       labels: {labels}")
    return docs
