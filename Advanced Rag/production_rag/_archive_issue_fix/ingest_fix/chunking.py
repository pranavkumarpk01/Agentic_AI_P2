"""[FIX version] chunking.py - STAGE 3 of ingestion: cut each document into chunks.

A chunk is the unit we search for later. One chunk should hold one complete idea.
Output: a list of chunks. A chunk is the document's dict with the text replaced by the piece, plus
  chunk_no : its position inside the file (unique per file)
  section  : the heading it sits under
  context  : a short line "document > section" that travels with the chunk
"""
import re


def split_into_chunks(text, max_chars=500, min_chars=60):
    """Helper for FIX C1. Returns a list of (section, chunk_text)."""

    # 1. Split on headings and remember the heading for each piece.
    #    "# Title" is the document title, "## Section" is a section.
    sections, section, body = [], "", []
    for line in text.splitlines():
        heading = re.match(r"^(#{1,3}) (.*)", line)
        if heading:
            if "".join(body).strip():
                sections.append((section, "\n".join(body).strip()))
            body = []
            if len(heading.group(1)) > 1:
                section = heading.group(2).strip()
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((section, "\n".join(body).strip()))

    # 2. Merge a very small section into the section before it.
    merged = []
    for section, body in sections:
        if merged and len(body) < min_chars:
            previous_section, previous_body = merged[-1]
            merged[-1] = (previous_section, previous_body + "\n" + section + ": " + body)
        else:
            merged.append((section, body))

    # 3. Pack whole sentences up to max_chars. Never cut inside a sentence.
    chunks = []
    for section, body in merged:
        current = ""
        parts = re.split(r"(?<=[.!?])(\s+)", body)  # [sentence, gap, sentence, gap, ...]
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            gap = parts[i - 1] if i else ""          # the space or line break that came before this sentence
            if current and len(current) + len(gap) + len(sentence) > max_chars:
                chunks.append((section, current))    # it would not fit: close this chunk, start the next
                current = sentence
            else:
                current += gap + sentence
        if current:
            chunks.append((section, current))
    return chunks


def chunk_documents(docs):
    chunks = []
    position = {}  # how many chunks each file has so far, so chunk_no is unique per file
    for doc in docs:
        text = doc["text"]

        # ===== FIX C1: split on headings, pack whole sentences up to 500 characters, merge tiny sections, remember the heading =====
        pieces = split_into_chunks(text)

        pieces = [(section, piece.strip()) for section, piece in pieces if len(piece.strip()) > 40]  # skip empty or tiny leftovers

        for section, piece in pieces:
            n = position.get(doc["source"], 0)
            position[doc["source"]] = n + 1

            # ===== FIX C1: a short context line (document > section) is stored with every chunk =====
            context = f"{doc.get('title', doc['source'])} > {section}"

            chunks.append({**doc, "text": piece, "chunk_no": n, "section": section, "context": context})

    # Show the class exactly what the chunks look like.
    for number, chunk in enumerate(chunks, start=1):
        print(f"\n[chunk {number}] {chunk['source']} #{chunk['chunk_no']}  ({len(chunk['text'])} characters)")
        if chunk["context"]:
            print(f"  context: {chunk['context']}")
        for line in chunk["text"].splitlines():
            print("  | " + line)
    return chunks
