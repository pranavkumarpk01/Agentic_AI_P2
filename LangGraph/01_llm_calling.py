# 01 - LLM calling
# The simplest possible graph: one node sends a question to Gemini.
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from common import get_llm, run_and_show


# 1. STATE = the shared memory of the graph.
class State(TypedDict):
    question: str
    answer: str


llm = get_llm()


# 2. NODE = a normal function. It reads the state and returns only the keys it changes.
def call_llm(state: State) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content}


# 3. Wire it up: START -> call_llm -> END
builder = StateGraph(State)
builder.add_node("call_llm", call_llm)
builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)
graph = builder.compile()


if __name__ == "__main__":
    initial = {"question": "What is LangGraph in one sentence?", "answer": ""}
    run_and_show(graph, initial)
