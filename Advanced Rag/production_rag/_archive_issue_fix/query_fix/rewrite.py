"""[FIX version] rewrite.py - STAGE 7 of querying: turn the user's question into a better search text.

People type short, vague or chatty questions. Documents use formal words.
A rewrite step bridges that gap before we search.
"""
from config import CHAT_MODEL, gemini


def rewrite_question(question):

    # ===== FIX Q1: ask Gemini to rewrite the question in the words a policy document would use =====
    prompt = (
        "Rewrite the question below as a short search query that uses the formal wording "
        "of a company policy document. Return only the query.\n\n"
        f"Question: {question}"
    )
    search_text = gemini.models.generate_content(model=CHAT_MODEL, contents=prompt).text.strip()

    print(f"  user typed  : {question}")
    print(f"  search text : {search_text}")
    return search_text
