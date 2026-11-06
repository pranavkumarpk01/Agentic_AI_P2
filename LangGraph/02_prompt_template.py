# 02 - Prompt template
# A prompt template has placeholders like {topic}. We fill them in from the state.
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from common import get_llm, run_and_show


class State(TypedDict):
    topic: str
    style: str
    prompt: str   # filled in by build_prompt
    answer: str   # filled in by call_llm


# The template, with placeholders
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher who explains things in a {style} style."),
    ("human", "Explain {topic} in 3 short bullet points."),
])

llm = get_llm()


def build_prompt(state: State) -> dict:
    messages = prompt_template.format_messages(topic=state["topic"], style=state["style"])
    prompt_text = "\n".join(f"[{m.type}] {m.content}" for m in messages)
    return {"prompt": prompt_text}


def call_llm(state: State) -> dict:
    response = llm.invoke(state["prompt"])
    return {"answer": response.content}


builder = StateGraph(State)
builder.add_node("build_prompt", build_prompt)
builder.add_node("call_llm", call_llm)
builder.add_edge(START, "build_prompt")
builder.add_edge("build_prompt", "call_llm")
builder.add_edge("call_llm", END)
graph = builder.compile()


if __name__ == "__main__":
    initial = {"topic": "LangGraph state", "style": "simple", "prompt": "", "answer": ""}
    run_and_show(graph, initial)


