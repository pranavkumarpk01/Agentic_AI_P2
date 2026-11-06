# 04 - Human in the loop
# The graph PAUSES, shows a draft to you, waits for your reply, then continues.
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from common import get_llm, show_state

llm = get_llm()


class State(TypedDict):
    task: str
    draft: str
    approved: bool
    feedback: str
    final: str


def write_draft(state: State) -> dict:
    message = f"Write a 2-line catchy tagline for: {state['task']}."
    if state["feedback"]:
        message += f" Improve it using this feedback: {state['feedback']}"
    return {"draft": llm.invoke(message).content}


def human_review(state: State) -> dict:
    # interrupt() pauses the graph here and sends a value out to the caller.
    # Whatever the caller sends back with Command(resume=...) becomes `answer`.
    answer = interrupt({
        "draft": state["draft"],
        "question": "Type 'yes' to approve, or type your feedback to improve it.",
    })
    if answer.strip().lower() == "yes":
        return {"approved": True}
    return {"approved": False, "feedback": answer}


def route_after_review(state: State) -> str:
    return "finish" if state["approved"] else "write_draft"


def finish(state: State) -> dict:
    return {"final": state["draft"]}


builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("human_review", human_review)
builder.add_node("finish", finish)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "human_review")
builder.add_conditional_edges("human_review", route_after_review,
                              {"finish": "finish", "write_draft": "write_draft"})
builder.add_edge("finish", END)


if __name__ == "__main__":
    # MemorySaver remembers where the graph paused, so it can resume later
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "demo-1"}}

    state = {"task": "a coffee shop mobile app", "draft": "", "approved": False,
             "feedback": "", "final": ""}
    show_state("BEFORE (input)", state)

    result = graph.invoke(state, config)
    while "__interrupt__" in result:          # the graph is paused, waiting for you
        payload = result["__interrupt__"][0].value
        print("\nDraft:", payload["draft"])
        print(payload["question"])
        answer = input("Your answer: ")
        result = graph.invoke(Command(resume=answer), config)   # continue from the pause

    show_state("AFTER (final)", result)
