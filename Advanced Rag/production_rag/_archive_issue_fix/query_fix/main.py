"""[FIX version] query/main.py - ask a question and watch every stage of retrieval.

    python -m query_fix.main "How many days do I have to return unopened electronics?"
    python -m query_fix.main "..." --k 2 --candidates 5
    python -m query_fix.main "..." --history        # a history question: also search superseded versions
    python -m query_fix.main "..." --no-answer      # stop before the final answer
    python -m query_fix.main "..." --show-prompt    # also print the prompt sent to Gemini
"""
import argparse

from config import qdrant
from query_fix.generation import generate_answer
from query_fix.rerank import rerank
from query_fix.retrieval import expand_with_neighbours, retrieve
from query_fix.rewrite import rewrite_question


def banner(text):
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--k", type=int, default=2, help="how many chunks are kept after reranking")
parser.add_argument("--candidates", type=int, default=5, help="how many chunks vector search fetches first")
parser.add_argument("--history", action="store_true", help="also search superseded versions (needs FIX V2 to matter)")
parser.add_argument("--no-answer", action="store_true", help="skip the final Gemini answer")
parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to Gemini")
args = parser.parse_args()

banner("STAGE 7 - REWRITE: prepare the search text")
search_text = rewrite_question(args.question)

banner(f"STAGE 8 - RETRIEVE: fetch {args.candidates} candidates from Qdrant")
hits = retrieve(search_text, args.candidates, include_history=args.history)

banner(f"STAGE 9 - RERANK: keep the best {args.k}")
final = rerank(args.question, hits, args.k)

banner("STAGE 10 - NEIGHBOURS: choose the passages the model will read")
passages = expand_with_neighbours(final)

if not args.no_answer:
    banner("STAGE 11 - ANSWER: Gemini writes the answer from the passages")
    generate_answer(args.question, passages, args.show_prompt)

qdrant.close()  # close the connection politely
