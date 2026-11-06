# 08 - Retry mechanism
# If the LLM call fails OR the answer is invalid, we try again (up to max_attempts).
import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from common import get_llm, run_and_show

llm = get_llm()


class State(TypedDict):
    attempts: int
    max_attempts: int
    answer: str
    error: str    # empty string means "no error"


def call_llm(state: State) -> dict:
    attempt = state["attempts"] + 1
    if attempt > 1:
        time.sleep(1)  # short pause before retrying
    try:
        text = llm.invoke("Pick a random whole number from 1 to 10. Reply with ONLY the number.").content.strip()
    except Exception as e:  # network / API problem
        return {"attempts": attempt, "error": f"API error: {e}"}

    # Check the answer. A bad answer counts as a failure too.
    if not (text.isdigit() and 1 <= int(text) <= 10):
        return {"attempts": attempt, "error": f"invalid output: '{text}'"}
    return {"attempts": attempt, "answer": text, "error": ""}


def give_up(state: State) -> dict:
    return {"answer": f"Gave up after {state['attempts']} attempts. Last error: {state['error']}"}


def route_after_call(state: State) -> str:
    if state["error"] == "":
        return "done"
    if state["attempts"] >= state["max_attempts"]:
        return "give_up"
    return "retry"


builder = StateGraph(State)
builder.add_node("call_llm", call_llm)
builder.add_node("give_up", give_up)
builder.add_edge(START, "call_llm")
builder.add_conditional_edges("call_llm", route_after_call, {
    "done": END,
    "give_up": "give_up",
    "retry": "call_llm",   # loop back and try again
})
builder.add_edge("give_up", END)
graph = builder.compile()


if __name__ == "__main__":
    initial = {"attempts": 0, "max_attempts": 3, "answer": "", "error": ""}
    run_and_show(graph, initial)
