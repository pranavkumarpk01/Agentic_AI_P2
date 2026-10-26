"""[ISSUE version] rewrite.py - STAGE 7 of querying: turn the user's question into a better search text.

People type short, vague or chatty questions. Documents use formal words.
A rewrite step bridges that gap before we search.
"""


def rewrite_question(question):

    # ===== ISSUE Q1: search with the question exactly as the user typed it =====
    search_text = question

    print(f"  user typed  : {question}")
    print(f"  search text : {search_text}")
    return search_text
