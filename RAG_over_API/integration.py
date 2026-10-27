"""
integration.py  -  puts it all together:

    question
       |
       v
    STEP 1  RAG     (query.py)   always runs: what does OUR policy say?
       |
       v
    STEP 2  ROUTER  (router.py)  Gemini decides: is the API needed?
       |-- NO_API -> skip the API
       |-- API    -> STEP 3: call api.py (Tavily) for live web data
       v
    STEP 4  ANSWER  one prompt with the contexts we collected -> Gemini -> answer

Run:   python integration.py "What is our maternity leave, and has the law changed recently?"
"""
import sys
from llm import generate
from query import retrieve
from router import decide
from api import search_web


def build_prompt(question: str, rag_hits: list[dict], api_results: list[dict]) -> str:
    policy_context = "\n".join(f"- {h['text']}" for h in rag_hits)

    # The web section is only added when the API was actually called.
    web_section = ""
    if api_results:
        web_context = "\n".join(f"- {r['title']} ({r['url']}): {r['content']}" for r in api_results)
        web_section = f"\nLIVE WEB INFORMATION (from a search API):\n{web_context}\n"

    return f"""You are an HR assistant. Answer the question using ONLY the context below.

COMPANY POLICY (from our internal HR documents):
{policy_context}
{web_section}
Rules:
- Say clearly which part of your answer comes from the company policy and which from the web (if web info is given).
- If the sources disagree, point it out.
- If the answer is not in the context, say you don't know.

Question: {question}
Answer:"""


def ask(question: str) -> str:
    # STEP 1 - RAG: always runs (cheap and local)
    print("STEP 1: RAG    - searching our HR policy in Qdrant...")
    rag_hits = retrieve(question)
    for h in rag_hits:
        print(f"   [{h['score']}] {h['text'][:90]}...")

    # STEP 2 - ROUTER: the LLM decides whether the API is needed
    print("\nSTEP 2: ROUTER - asking Gemini: do we need the API?")
    needs_api, reason = decide(question, rag_hits)
    print(f"   Decision: {'CALL API' if needs_api else 'NO API - policy is enough'}")
    print(f"   Reason  : {reason}")

    # STEP 3 - API: only if the router said so
    api_results = []
    if needs_api:
        print("\nSTEP 3: API    - calling the Tavily web search API...")
        api_results = search_web(question)
        for r in api_results:
            print(f"   {r['title'][:90]}")
    else:
        print("\nSTEP 3: API    - skipped (saved an API credit)")

    # STEP 4 - ANSWER: one prompt, whatever contexts we collected
    print("\nSTEP 4: LLM    - generating the final answer with Gemini...\n")
    return generate(build_prompt(question, rag_hits, api_results))


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is our maternity leave, and has the law changed recently?"
    print(f"QUESTION: {question}\n")
    print("ANSWER:\n" + ask(question))
