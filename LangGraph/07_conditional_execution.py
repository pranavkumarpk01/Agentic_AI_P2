# 07 - Conditional execution
# A router looks at the state and decides which node runs next.
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from common import get_llm, run_and_show

llm = get_llm()


class State(TypedDict):
    question: str
    category: str   # "math", "creative" or "general"
    answer: str


def classify(state: State) -> dict:
    prompt = (
        "Classify this question as exactly one word: math, creative or general.\n"
        f"Question: {state['question']}\nAnswer with only that one word."
    )
    text = llm.invoke(prompt).content.lower()
    # Be forgiving: pick the first known word found in the reply
    category = next((c for c in ["math", "creative", "general"] if c in text), "general")
    return {"category": category}


def math_expert(state: State) -> dict:
    answer = llm.invoke(f"Solve this step by step, then give a short answer: {state['question']}").content
    return {"answer": answer}


def creative_writer(state: State) -> dict:
    answer = llm.invoke(f"Answer creatively in 3 lines: {state['question']}").content
    return {"answer": answer}


def general_helper(state: State) -> dict:
    answer = llm.invoke(f"Answer briefly: {state['question']}").content
    return {"answer": answer}


def route_by_category(state: State) -> str:
    # The router only READS the state and returns the name of the next node
    return state["category"]


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("math_expert", math_expert)
builder.add_node("creative_writer", creative_writer)
builder.add_node("general_helper", general_helper)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_category, {
    "math": "math_expert",
    "creative": "creative_writer",
    "general": "general_helper",
})
builder.add_edge("math_expert", END)
builder.add_edge("creative_writer", END)
builder.add_edge("general_helper", END)
graph = builder.compile()


if __name__ == "__main__":
    initial = {"question": "What is 15% of 240?", "category": "", "answer": ""}
    run_and_show(graph, initial)
