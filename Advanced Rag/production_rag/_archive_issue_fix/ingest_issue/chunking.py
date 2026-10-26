"""[ISSUE version] chunking.py - STAGE 3 of ingestion: cut each document into chunks.

A chunk is the unit we search for later. One chunk should hold one complete idea.
Output: a list of chunks. A chunk is the document's dict with the text replaced by the piece, plus
  chunk_no : its position inside the file (unique per file)
  section  : the heading it sits under
  context  : a short line "document > section" that travels with the chunk
"""


def chunk_documents(docs):
    chunks = []
    position = {}  # how many chunks each file has so far, so chunk_no is unique per file
    for doc in docs:
        text = doc["text"]

        # ===== ISSUE C1: cut every 300 characters, wherever that lands (even in the middle of a sentence or a number) =====
        pieces = [("", text[i:i + 300]) for i in range(0, len(text), 300)]

        pieces = [(section, piece.strip()) for section, piece in pieces if len(piece.strip()) > 40]  # skip empty or tiny leftovers

        for section, piece in pieces:
            n = position.get(doc["source"], 0)
            position[doc["source"]] = n + 1

            # ===== ISSUE C1: no context travels with the chunk =====
            context = ""

            chunks.append({**doc, "text": piece, "chunk_no": n, "section": section, "context": context})

    # Show the class exactly what the chunks look like.
    for number, chunk in enumerate(chunks, start=1):
        print(f"\n[chunk {number}] {chunk['source']} #{chunk['chunk_no']}  ({len(chunk['text'])} characters)")
        if chunk["context"]:
            print(f"  context: {chunk['context']}")
        for line in chunk["text"].splitlines():
            print("  | " + line)
    return chunks
