"""[ISSUE version] fetch.py - STAGE 1 of ingestion: find the documents and read them.

Output: a list of documents (dicts). Every document has "text" and "source".
With the fixes it also has:
  - an identity and date: title, doc_key, version, effective_date  (FIX V1)
  - a status decided by code: current / superseded / upcoming       (FIX V1)
  - a kind: "text" or "table"                                       (FIX T1)
"""
from pathlib import Path

import pdfplumber


def read_pdf_plain(path):
    """The simple way: ask each page for its text, tables included."""
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def fetch_documents(folder="data/raw"):
    docs = []
    for path in sorted(Path(folder).rglob("*")):  # rglob = look inside every sub-folder too
        if path.suffix not in (".md", ".pdf"):
            continue

        # ===== ISSUE V1: keep only the file name. Nothing says which version this is or whether it is in force =====
        info = {"source": path.name, "path": str(path)}

        if path.suffix == ".pdf":

            # ===== ISSUE T1: read the PDF as plain text. Tables come out as loose lines, and a table that runs onto page 2 is cut in the middle =====
            pieces = [{"text": read_pdf_plain(path), "kind": "text"}]

        else:
            pieces = [{"text": path.read_text(), "kind": "text"}]

        for piece in pieces:
            docs.append({**info, **piece})

    # ===== ISSUE V1: part 2 - no status is stored, so search cannot tell the old rulebook from the new one =====
    docs = docs

    for doc in docs:
        labels = {k: v for k, v in doc.items() if k not in ("text", "path")}
        print(f"  {doc['source']}  [{doc['kind']}]  {len(doc['text'])} characters")
        print(f"       labels: {labels}")
    return docs
