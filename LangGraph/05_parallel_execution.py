# 05 - Parallel execution
# pros, cons and examples run AT THE SAME TIME. combine waits until all three are done.
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from common import get_llm, run_and_show

llm = get_llm()


class State(TypedDict):
    topic: str
    # Annotated[list, operator.add] means: when a node returns a list,
    # it is ADDED to the existing list instead of replacing it.
    # That lets the three parallel nodes all write to "results" safely.
    results: Annotated[list[str], operator.add]
    summary: str


def ask(instruction: str, topic: str) -> str:
    return llm.invoke(f"{instruction} {topic}. Keep it to 2 short sentences.").content


def pros(state: State) -> dict:
    return {"results": [f"PROS: {ask('List the advantages of', state['topic'])}"]}


def cons(state: State) -> dict:
    return {"results": [f"CONS: {ask('List the disadvantages of', state['topic'])}"]}


def examples(state: State) -> dict:
    return {"results": [f"EXAMPLES: {ask('Give real-world examples of', state['topic'])}"]}


def combine(state: State) -> dict:
    text = "\n".join(state["results"])
    summary = llm.invoke(f"Combine these notes into a short summary:\n{text}").content
    return {"summary": summary}


builder = StateGraph(State)
builder.add_node("pros", pros)
builder.add_node("cons", cons)
builder.add_node("examples", examples)
builder.add_node("combine", combine)

# Fan-out: START sends the work to all three nodes at once
builder.add_edge(START, "pros")
builder.add_edge(START, "cons")
builder.add_edge(START, "examples")
# Fan-in: combine runs only after ALL three have finished
builder.add_edge(["pros", "cons", "examples"], "combine")
builder.add_edge("combine", END)
graph = builder.compile()


if __name__ == "__main__":
    initial = {"topic": "using AI agents in customer support", "results": [], "summary": ""}
    run_and_show(graph, initial)
