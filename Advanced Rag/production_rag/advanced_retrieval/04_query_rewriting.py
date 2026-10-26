"""
04 - QUERY REWRITING
====================
Users do not type perfect search queries. We ask an LLM to rewrite the
question BEFORE retrieval, then search with the rewritten version.

Three situations:
  A. Follow-up question  "what about the US?"         -> needs chat history
  B. Messy question      "vpn not wrking err4012 pls" -> typos, broken codes
  C. Multi-query         one question -> 3 variants   -> search all, fuse (RRF)

LLM: Google Gemini. Uses GEMINI_API_KEY and GEMINI_CHAT_MODEL from the production_rag/.env file.
No key / no internet?  The script replays saved LLM answers from
data/llm_cache.json so the class demo still works - it tells you when it does.

Run:  python 04_query_rewriting.py            (Enter between steps)
      python 04_query_rewriting.py --no-pause
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from display import console, fixed, note, pause, problem, short, step, table, title
from rag_core import HybridRetriever, load_knowledge_base, reciprocal_rank_fusion, tokenize

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")          # optional own .env inside advanced_retrieval/
load_dotenv(HERE.parent / ".env")   # the production_rag/.env you already have (does not override the first)
CACHE_PATH = HERE / "data" / "llm_cache.json"
MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")   # same setting as the main project
API_KEY = os.getenv("GEMINI_API_KEY")

title("04 - Query rewriting")


# ---------------------------------------------------------------------------
# The ONLY place we talk to the LLM
# ---------------------------------------------------------------------------
def ask_llm(cache_key, prompt):
    """Send `prompt` to Gemini and return the parsed JSON answer.
    Falls back to the saved answer in data/llm_cache.json if there is no key or the call fails."""
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

    if API_KEY:
        try:
            from google import genai

            client = genai.Client(api_key=API_KEY)
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0},
            )
            answer = json.loads(response.text)
            console.print(f"[green]LIVE answer from {MODEL}[/green]")
            cache[cache_key] = answer                     # save it for offline use next time
            CACHE_PATH.write_text(json.dumps(cache, indent=2))
            return answer
        except Exception as error:                         # quota, network, bad model name...
            console.print(f"[red]Gemini call failed:[/red] {short(str(error), 150)}")

    console.print("[yellow]OFFLINE: replaying a saved LLM answer from data/llm_cache.json[/yellow]")
    return cache[cache_key]


def show(heading, results):
    table(heading, ["rank", "source", "text"],
          [[r, chunks[i]["metadata"]["source"], short(chunks[i]["text"], 80)]
           for r, (i, _) in enumerate(results, start=1)])


# ---------------------------------------------------------------------------
step(0, "Build the hybrid retriever from script 03")
# ---------------------------------------------------------------------------
chunks = load_knowledge_base()
retriever = HybridRetriever(chunks)
console.print(f"Indexed {len(chunks)} chunks. Gemini key found: {bool(API_KEY)}")
pause()

# ---------------------------------------------------------------------------
step(1, "Case A - a follow-up question that makes no sense on its own")
# ---------------------------------------------------------------------------
history = [
    {"role": "user", "content": "How many paid leave days do employees in India get?"},
    {"role": "assistant", "content": "Employees in India get 24 days of paid annual leave per year."},
]
follow_up = "what about the US?"

for message in history:
    console.print(f"[dim]{message['role']:>9}:[/dim] {message['content']}")
console.print(f"[bold]     user:[/bold] {follow_up}")

show(f"BEFORE - search with the raw text {follow_up!r}", retriever.search(follow_up))
problem("'what about the US?' has no topic. The retriever does not know we are talking about leave.")

conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history)
prompt_a = f"""You rewrite questions for a company knowledge-base search engine.
Use the conversation history to turn the follow-up into a complete, standalone question.
Keep names, codes and numbers exactly. Do not answer the question.

Conversation:
{conversation}

Follow-up question: {follow_up}

Return JSON: {{"rewritten_query": "..."}}"""

console.print("\n[bold]Prompt sent to the LLM:[/bold]")
console.print(prompt_a, style="dim")
rewritten = ask_llm("followup", prompt_a)["rewritten_query"]
console.print(f"\nRewritten query: [bold green]{rewritten}[/bold green]")

show(f"AFTER - search with {rewritten!r}", retriever.search(rewritten))
fixed("The standalone question pulls the US leave policy to the top.")
pause()

# ---------------------------------------------------------------------------
step(2, "Case B - a messy question (typos, broken error code, filler words)")
# ---------------------------------------------------------------------------
messy = "vpn not wrking!!! says err4012 pls help asap"
show(f"BEFORE - search with {messy!r}", retriever.search(messy))
console.print(f"keyword tokens: {tokenize(messy)}")
console.print(f"BM25 keyword matches for the code: {retriever.bm25.search('err4012')}  <- empty list")
problem("'err4012' is not the same token as 'err-4012' and 'wrking' is not 'working'.\n"
        "Keyword search cannot match the exact error code - only vector search is helping here.")

prompt_b = f"""You rewrite questions for a company IT knowledge-base search engine.
Fix spelling, remove filler words, and write error codes in the official format ERR-NNNN.
Do not answer the question.

Question: {messy}

Return JSON: {{"rewritten_query": "..."}}"""
rewritten_b = ask_llm("messy", prompt_b)["rewritten_query"]
console.print(f"Rewritten query: [bold green]{rewritten_b}[/bold green]")
console.print(f"keyword tokens: {tokenize(rewritten_b)}")

show(f"AFTER - search with {rewritten_b!r}", retriever.search(rewritten_b))
fixed("Clean query -> 'err-4012' is now an exact keyword match, so BOTH retrievers agree\n"
      "and the ERR-4012 fix is retrieved first.")
pause()

# ---------------------------------------------------------------------------
step(3, "Case C - multi-query: ask the same thing in several ways, then fuse")
# ---------------------------------------------------------------------------
question = "Can I expense my cab to the airport?"
show(f"BEFORE - single query {question!r}", retriever.search(question))

prompt_c = f"""Generate 3 different search queries that would find documents answering the
question below. Use different wording and synonyms (for example official policy terms).
Do not answer the question.

Question: {question}

Return JSON: {{"queries": ["...", "...", "..."]}}"""
variants = ask_llm("multi", prompt_c)["queries"]

ranked_lists = []
for v in [question] + variants:
    results = retriever.search(v, top_k=5)
    ranked_lists.append([i for i, _ in results])
    console.print(f"  query: [cyan]{v}[/cyan]  ->  top: {short(chunks[results[0][0]]['text'], 60)}")

fused = reciprocal_rank_fusion(ranked_lists)[:3]
show("AFTER - all variants fused with RRF", fused)
fixed("The user said 'cab' and 'expense'; the documents say 'taxi', 'reimbursed' and\n"
      "'expense claim'. Each variant catches different wording, and chunks found by MANY\n"
      "variants rise to the top - here: the taxi rule AND how to submit the claim.")

note("Trade-off: every rewrite is an extra LLM call -> more latency and cost.\n"
     "Use it when users write short, vague, or conversational questions.")
