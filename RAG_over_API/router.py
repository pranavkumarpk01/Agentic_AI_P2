"""
router.py  -  THE DECISION: should we call the API for this question?

############################################################################
#  HOW THE SYSTEM DECIDES WHEN TO CALL THE API                             #
#                                                                          #
#  1. RAG always runs first (it is free, local and fast).                  #
#  2. We show Gemini two things: the QUESTION and the POLICY CHUNKS RAG    #
#     found.                                                               #
#  3. We ask Gemini ONE narrow question:                                   #
#        "Can the policy chunks fully answer this question, or do we need  #
#         live / external / up-to-date information from the web?"          #
#  4. Gemini replies in a fixed format:                                    #
#        DECISION: API       -> call Tavily too                            #
#        DECISION: NO_API    -> answer from the policy only                #
#        REASON: <one sentence>                                            #
#  5. Our code reads that reply. If it can't understand it, we choose API  #
#     (safe default: extra information is better than a wrong answer).     #
#                                                                          #
#  So the LLM is acting as a ROUTER: it does not answer the question,      #
#  it only chooses the path.                                               #
############################################################################

Test it alone:   python router.py "What is the latest EPFO contribution rate?"
"""
import sys
from llm import generate
from query import retrieve

ROUTER_PROMPT = """You are a routing assistant for an HR chatbot.
You do NOT answer the question. You only decide whether a web search API is needed.

Use NO_API when the company policy excerpts below fully answer the question
(for example: leave counts, working hours, notice period, reimbursement rules).

Use API when ANY of these is true:
- the question asks about current, latest, recent or changed information
  (laws, government rates, news, holidays announced this year, etc.)
- the policy excerpts only refer to an external law or authority without giving
  the actual current details
- the policy excerpts do not contain the answer and the topic is general / public knowledge

COMPANY POLICY EXCERPTS:
{policy_context}

QUESTION: {question}

Reply in EXACTLY this format and nothing else:
DECISION: API or NO_API
REASON: one short sentence"""


def decide(question: str, rag_hits: list[dict]) -> tuple[bool, str]:
    """Return (needs_api, reason)."""
    policy_context = "\n".join(f"- {h['text']}" for h in rag_hits)
    prompt = ROUTER_PROMPT.format(policy_context=policy_context, question=question)

    reply = generate(prompt, temperature=0)   # temperature 0 = consistent decisions

    # --- read the LLM's reply ---
    needs_api = None
    reason = ""
    for line in reply.splitlines():
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            value = line.split(":", 1)[1].strip().upper()
            needs_api = (value == "API")          # "NO_API" -> False
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if needs_api is None:                          # reply was not in the expected format
        return True, "Router reply unclear, calling API to be safe."
    return needs_api, reason


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many casual leaves do I get?"
    hits = retrieve(question)
    needs_api, reason = decide(question, hits)
    print(f"Question : {question}")
    print(f"Decision : {'CALL API' if needs_api else 'POLICY ONLY (no API call)'}")
    print(f"Reason   : {reason}")
