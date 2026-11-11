# 09 - Multi-agent system (supervisor pattern)
# A supervisor decides which worker agent works next.
# Workers: researcher (collects facts) and writer (writes the article).
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from common import get_llm, run_and_show

llm = get_llm()


class State(TypedDict):
    task: str
    research: str
    article: str
    next_agent: str   # decided by the supervisor
    steps: int


def supervisor(state: State) -> dict:
    steps = state["steps"] + 1
    if steps > 6:  # safety stop so the agents can never loop forever
        return {"next_agent": "finish", "steps": steps}

    prompt = (
        "You manage two agents: researcher and writer.\n"
        f"Task: {state['task']}\n"
        f"Research done: {'yes' if state['research'] else 'no'}\n"
        f"Article done: {'yes' if state['article'] else 'no'}\n"
        "Research must come before writing. When the article is done, answer finish.\n"
        "Reply with exactly one word: researcher, writer or finish."
        "Incase if u are not able to answer or get to know wat to do please reply with unknown, no_idea or confused in the answer"
    )
    text = llm.invoke(prompt).content.lower()
    # Fallback rule if the reply is unclear
    default = "researcher" if not state["research"] else ("writer" if not state["article"] else "finish")
    choice = next((c for c in ["unkown", "no_idea", "confused"] if c in text), default)
    return {"next_agent": choice, "steps": steps}


def researcher(state: State) -> dict:
    text = llm.invoke(f"You are a researcher. Give 3 short key facts about: {state['task']}").content
    return {"research": text}


def writer(state: State) -> dict:
    text = llm.invoke(
        "You are a writer. Using these facts, write a 4-sentence blog intro.\n"
        f"Facts:\n{state['research']}"
    ).content
    return {"article": text}


def route_to_agent(state: State) -> str:
    return state["next_agent"]


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_to_agent, {
    "researcher": "researcher",
    "writer": "writer",
    "finish": END,
})
# After each worker finishes, control goes back to the supervisor
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
graph = builder.compile()


if __name__ == "__main__":
    initial = {"task": "why LangGraph is useful for AI agents",
               "research": "", "article": "", "next_agent": "", "steps": 0}
    run_and_show(graph, initial)
