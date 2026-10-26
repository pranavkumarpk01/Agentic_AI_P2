"""[FIX version] generation.py - STAGE 11 of querying: let Gemini write the answer from the passages.

This is the "G" in RAG. The model only sees what we put in the prompt.
"""
from config import CHAT_MODEL, gemini


def generate_answer(question, passages, show_prompt=False):
    context = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))

    # ===== FIX Q3: ground the model. Use only the context, refuse when the answer is missing, cite passage numbers =====
    prompt = (
        "Answer the question using ONLY the context below.\n"
        'If the answer is not in the context, say exactly: "I don\'t know based on the documents."\n'
        "After the answer, cite the passage numbers you used, like [1].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    if show_prompt:
        print("--- PROMPT SENT TO GEMINI ---\n" + prompt + "\n--- END OF PROMPT ---\n")

    reply = gemini.models.generate_content(model=CHAT_MODEL, contents=prompt)
    print("ANSWER:", reply.text)
    print("\nPassages behind the numbers:")
    for i, p in enumerate(passages, start=1):
        print(f"  [{i}] {p['source']} #{p['chunk_no']}")
