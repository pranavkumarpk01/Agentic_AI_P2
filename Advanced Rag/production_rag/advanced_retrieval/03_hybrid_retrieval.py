"""
03 - HYBRID RETRIEVAL  (BM25 keyword search + vector search + RRF)
==================================================================
  Keyword search (BM25) is great at EXACT words: codes, names, IDs (ERR-4012).
  Vector search is great at MEANING: paraphrases ("work from my house").
  Each one fails where the other one shines -> run BOTH and merge the lists.

Run:  python 03_hybrid_retrieval.py            (Enter between steps)
      python 03_hybrid_retrieval.py --no-pause
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

from display import console, fixed, note, pause, problem, short, step, table, title
from rag_core import STOPWORDS, embed, load_knowledge_base

title("03 - Hybrid retrieval")

chunks = load_knowledge_base()
texts = [c["text"] for c in chunks]
console.print(f"Loaded {len(chunks)} chunks from data/docs")

# ---------------------------------------------------------------------------
step(1, "Keyword search with BM25 - tokenize")
# ---------------------------------------------------------------------------
# STOPWORDS (imported from rag_core) = very common words like "the", "how", "my"
# that carry no meaning for search, so we throw them away.


def tokenize(text):
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())   # keeps "err-4012" as ONE word
    return [w for w in words if w not in STOPWORDS]


console.print("tokenize('How do I fix ERR-4012?') ->", tokenize("How do I fix ERR-4012?"))
console.print("tokenize(chunk 0)               ->", tokenize(texts[0]))
pause()

# ---------------------------------------------------------------------------
step(2, "Keyword search with BM25 - score every chunk")
# ---------------------------------------------------------------------------
note(
    "BM25 score = sum over query words of   IDF(word) x TF-part(word, chunk)\n"
    "  IDF  : rare word (appears in few chunks)  -> big weight\n"
    "  TF   : word appears more often in chunk    -> higher, but it saturates (k1)\n"
    "  b    : long chunks are penalised a little  (length normalisation)"
)

doc_tokens = [tokenize(t) for t in texts]
N = len(doc_tokens)
avg_len = sum(len(d) for d in doc_tokens) / N
doc_freq = Counter(word for tokens in doc_tokens for word in set(tokens))
K1, B = 1.5, 0.75


def bm25_search(query, top_k=5):
    scores = []
    for tokens in doc_tokens:
        counts = Counter(tokens)
        score = 0.0
        for word in tokenize(query):
            if word not in counts:
                continue
            idf = math.log(1 + (N - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5))
            tf = counts[word]
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(tokens) / avg_len))
        scores.append(score)
    ranked = sorted(range(N), key=lambda i: -scores[i])
    return [(i, scores[i]) for i in ranked if scores[i] > 0][:top_k]   # only real matches


for word in ["err-4012", "leave", "employees"]:
    idf = math.log(1 + (N - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5))
    console.print(f"  '{word}' appears in {doc_freq[word]:>2} chunks  ->  IDF = {idf:.2f}")
pause()

# ---------------------------------------------------------------------------
step(3, "Vector search - embed chunks once, embed the query, cosine similarity")
# ---------------------------------------------------------------------------
console.print("Embedding chunks (first run downloads the model)...")
chunk_vectors = embed(texts)                              # shape: (chunks, 384)
console.print(f"chunk_vectors shape = {chunk_vectors.shape}")
console.print(f"first 6 numbers of chunk 0 = {np.round(chunk_vectors[0][:6], 3)}")


def vector_search(query, top_k=5):
    query_vector = embed([query])[0]
    similarities = chunk_vectors @ query_vector           # cosine (vectors are normalised)
    best = np.argsort(-similarities)[:top_k]
    return [(int(i), float(similarities[i])) for i in best]


pause()


# ---------------------------------------------------------------------------
step(4, "Reciprocal Rank Fusion (RRF) - merge two ranked lists")
# ---------------------------------------------------------------------------
note(
    "BM25 scores (0..~10) and cosine scores (0..1) are on different scales,\n"
    "so we do NOT add the scores. We use only the RANK:\n"
    "      RRF(chunk) = 1/(60 + rank_in_bm25) + 1/(60 + rank_in_vector)\n"
    "A chunk ranked high in BOTH lists wins. Missing from a list = adds 0."
)


def rrf(ranked_lists, k=60):
    fused = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_index in enumerate(ranked, start=1):
            fused[chunk_index] += 1 / (k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])


pause()


# ---------------------------------------------------------------------------
def compare(query, lesson):
    """Run keyword, vector and hybrid search for one query and print all three."""
    console.print(f"\nQuery: [bold yellow]{query}[/bold yellow]")
    keyword = bm25_search(query)
    semantic = vector_search(query)

    table("Keyword (BM25)", ["rank", "score", "source", "text"],
          [[r, f"{s:.2f}", chunks[i]["metadata"]["source"], short(chunks[i]["text"], 70)]
           for r, (i, s) in enumerate(keyword[:3], start=1)] or [["-", "-", "-", "NO keyword match at all"]])
    table("Vector (semantic)", ["rank", "cosine", "source", "text"],
          [[r, f"{s:.3f}", chunks[i]["metadata"]["source"], short(chunks[i]["text"], 70)]
           for r, (i, s) in enumerate(semantic[:3], start=1)])

    keyword_rank = {i: r for r, (i, _) in enumerate(keyword, start=1)}
    vector_rank = {i: r for r, (i, _) in enumerate(semantic, start=1)}
    fused = rrf([[i for i, _ in keyword], [i for i, _ in semantic]])

    table("HYBRID (RRF)  - how each score is built", ["rank", "BM25 rank", "vector rank", "RRF score", "text"],
          [[r, keyword_rank.get(i, "-"), vector_rank.get(i, "-"),
            " + ".join(f"1/(60+{x})" for x in [keyword_rank.get(i), vector_rank.get(i)] if x) + f" = {s:.4f}",
            short(chunks[i]["text"], 60)]
           for r, (i, s) in enumerate(fused[:3], start=1)])
    note(lesson)


# ---------------------------------------------------------------------------
step(5, "Case A - exact code: keyword search shines")
# ---------------------------------------------------------------------------
compare("How do I fix ERR-4012?",
        "BM25 matches the exact token 'err-4012' (it is in only ONE chunk, so its IDF is high).\n"
        "To the embedding model ERR-4012, ERR-4013 and ERR-4021 are 'the same kind of text' -\n"
        "compare their cosine scores: the gap is small, so the wrong code can easily win.")
pause()

# ---------------------------------------------------------------------------
step(6, "Case B - paraphrase: vector search shines")
# ---------------------------------------------------------------------------
compare("Can I do my job from my house sometimes?",
        "None of the words 'job', 'house', 'sometimes' exist in the documents, so BM25\n"
        "finds nothing. Vector search understands it means 'remote work'.")
pause()

# ---------------------------------------------------------------------------
step(7, "Case C - both signals in one query: hybrid shines")
# ---------------------------------------------------------------------------
compare("ERR-4012 popped up, how do I get my VPN working again?",
        "BM25 contributes the exact error code, vector search contributes the meaning\n"
        "('VPN working again' = troubleshooting). RRF rewards the chunk that ranks\n"
        "well in BOTH lists, so the ERR-4012 fix ends up on top.")

fixed("Hybrid retrieval = recall of vector search + precision of keyword search.")
