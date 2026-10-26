"""[ISSUE version] preprocessing.py - STAGE 2 of ingestion: clean the text before it is cut into chunks.

Documents exported from a handbook or PDF carry noise: a header on every page, page markers.
If we leave it in, it ends up inside the chunks, inside the database, and inside every prompt.
"""


def preprocess_documents(docs):
    cleaned = []
    for doc in docs:

        # ===== ISSUE X1: use the text exactly as it is, so page headers and footers stay in =====
        text = doc["text"]

        cleaned.append({**doc, "text": text})  # same labels, new text
        print(f"  {doc['source']} [{doc['kind']}]: {len(doc['text'])} characters -> {len(text)} characters")
    return cleaned
