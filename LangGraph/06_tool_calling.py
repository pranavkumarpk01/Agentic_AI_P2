# 06 - Tool calling
# The LLM can ask for a Python function (a "tool") to be run.
# Flow: agent -> tools (if requested) -> agent -> ... -> END
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from common import get_llm


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (demo data)."""
    return f"It is 28 C and sunny in {city}."


tools = [add_numbers, get_weather]
llm = get_llm(temperature=0).bind_tools(tools)  # tell Gemini which tools exist


class State(TypedDict):
    # add_messages: new messages are appended to the list
    messages: Annotated[list, add_messages]


def agent(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))   # runs the tool the LLM asked for
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # "tools" if a tool was requested, else END
builder.add_edge("tools", "agent")
graph = builder.compile()


if __name__ == "__main__":
    initial = {"messages": [HumanMessage(content="What is 12 + 30? Also, what's the weather in Chennai?")]}
    print("BEFORE: 1 message (your question)")
    for state in graph.stream(initial, stream_mode="values"):
        last = state["messages"][-1]
        print(f"\n=== messages so far: {len(state['messages'])}, last = {last.type} ===")
        print(str(last.content)[:300])
        if getattr(last, "tool_calls", None):
            print("tool calls requested:", last.tool_calls)
