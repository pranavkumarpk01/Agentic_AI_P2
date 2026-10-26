"""[FIX version] embedding.py - STAGE 5 of ingestion: turn every chunk into a vector with Gemini.

A vector is a list of numbers that captures the MEANING of a text.
Texts with similar meaning get vectors that are close together. That is what search uses later.
"""
from google.genai import types

from config import EMBED_MODEL, VECTOR_SIZE, gemini


def embed_chunks(chunks):

    # ===== FIX X2: put the context line (document > section) in front of the text (needs FIX C1) =====
    texts_to_embed = [(chunk["context"] + "\n" + chunk["text"]).strip() for chunk in chunks]

    # One call to Gemini for all chunks. "RETRIEVAL_DOCUMENT" says: these texts will be searched.
    result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=texts_to_embed,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=VECTOR_SIZE),
    )
    vectors = [item.values for item in result.embeddings]

    # Show the class what went in and what came out (first few chunks only).
    for number, (sent, vector) in enumerate(zip(texts_to_embed, vectors), start=1):
        if number > 6:
            print(f"  ... and {len(vectors) - 6} more")
            break
        short = sent.replace("\n", " / ")[:90]
        print(f"\n[chunk {number}] text sent to Gemini: {short}")
        print(f"           vector back: {len(vector)} numbers, starting {[round(x, 3) for x in vector[:4]]} ...")
    return vectors
