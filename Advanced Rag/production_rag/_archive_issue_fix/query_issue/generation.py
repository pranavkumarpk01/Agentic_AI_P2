"""[ISSUE version] generation.py - STAGE 11 of querying: let Gemini write the answer from the passages.

This is the "G" in RAG. The model only sees what we put in the prompt.
"""
from config import CHAT_MODEL, gemini


def generate_answer(question, passages, show_prompt=False):
    context = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))

    # ===== ISSUE Q3: a bare prompt. Nothing stops the model from guessing when the passages do not contain the answer =====
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    if show_prompt:
        print("--- PROMPT SENT TO GEMINI ---\n" + prompt + "\n--- END OF PROMPT ---\n")

    reply = gemini.models.generate_content(model=CHAT_MODEL, contents=prompt)
    print("ANSWER:", reply.text)
    print("\nPassages behind the numbers:")
    for i, p in enumerate(passages, start=1):
        print(f"  [{i}] {p['source']} #{p['chunk_no']}")
