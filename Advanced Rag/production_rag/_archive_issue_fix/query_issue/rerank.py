"""[ISSUE version] rerank.py - STAGE 9 of querying: re-order the candidates and keep the best k.

Vector search is fast but rough: it compares one number-list to another.
A reranker reads the question AND each chunk together and judges how well the chunk answers it.
"""


def rerank(question, hits, k):

    # ===== ISSUE Q2: trust the vector-search order and simply keep the first k =====
    final = hits[:k]

    print(f"  kept the top {k}:")
    for rank, hit in enumerate(final, start=1):
        was = hits.index(hit) + 1  # where this chunk stood in the vector-search order
        print(f"\n  [{rank}] (was #{was} in vector order)  {hit.payload['source']}")
        for line in hit.payload["text"].splitlines():
            print("      | " + line)
    return final
