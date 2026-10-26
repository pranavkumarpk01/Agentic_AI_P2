"""[FIX version] preprocessing.py - STAGE 2 of ingestion: clean the text before it is cut into chunks.

Documents exported from a handbook or PDF carry noise: a header on every page, page markers.
If we leave it in, it ends up inside the chunks, inside the database, and inside every prompt.
"""


def clean(text):
    """Helper for FIX X1: throw away page headers, footers and page-break markers."""
    kept = []
    for line in text.splitlines():
        if line.startswith("ZENTRA RETAIL | INTERNAL"):  # the header repeated on every page
            continue
        if line.startswith("--- Page"):                   # the page-break marker
            continue
        kept.append(line)
    return "\n".join(kept)


def preprocess_documents(docs):
    cleaned = []
    for doc in docs:

        # ===== FIX X1: remove page headers, footers and page-break markers =====
        text = clean(doc["text"])

        cleaned.append({**doc, "text": text})  # same labels, new text
        print(f"  {doc['source']} [{doc['kind']}]: {len(doc['text'])} characters -> {len(text)} characters")
    return cleaned
