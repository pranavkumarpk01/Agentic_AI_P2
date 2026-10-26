"""[FIX version] rerank.py - STAGE 9 of querying: re-order the candidates and keep the best k.

Vector search is fast but rough: it compares one number-list to another.
A reranker reads the question AND each chunk together and judges how well the chunk answers it.
"""
import json

from google.genai import types

from config import CHAT_MODEL, gemini


def rerank(question, hits, k):

    # ===== FIX Q2: ask Gemini to score every candidate from 0 to 10 for this question, keep the best k =====
    passages = "\n\n".join(f"[{i}] {hit.payload['text']}" for i, hit in enumerate(hits))
    prompt = (
        "Score how well each passage answers the question, from 0 (useless) to 10 (answers it directly).\n"
        "Return a JSON list with exactly one number per passage, in the same order.\n\n"
        f"Question: {question}\n\nPassages:\n{passages}"
    )
    reply = gemini.models.generate_content(
        model=CHAT_MODEL, contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    scores = json.loads(reply.text)
    best_first = sorted(zip(scores, hits), key=lambda pair: pair[0], reverse=True)
    final = [hit for score, hit in best_first[:k]]

    print(f"  kept the top {k}:")
    for rank, hit in enumerate(final, start=1):
        was = hits.index(hit) + 1  # where this chunk stood in the vector-search order
        print(f"\n  [{rank}] (was #{was} in vector order)  {hit.payload['source']}")
        for line in hit.payload["text"].splitlines():
            print("      | " + line)
    return final
