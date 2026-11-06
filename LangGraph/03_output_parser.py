# 03 - Output parser
# The LLM returns plain text. A parser turns that text into a clean Python object.
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from common import get_llm, run_and_show


# The shape we want back from the LLM
class Movie(BaseModel):
    title: str = Field(description="name of the movie")
    year: int = Field(description="release year")
    genre: str = Field(description="main genre")


parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("human", "Suggest one famous {genre} movie.\n{format_instructions}"),
])

llm = get_llm(temperature=0)


class State(TypedDict):
    genre: str
    raw_output: str   # the plain text the LLM returned
    movie: dict       # the parsed result


def ask_llm(state: State) -> dict:
    chain = prompt | llm
    response = chain.invoke({
        "genre": state["genre"],
        "format_instructions": parser.get_format_instructions(),
    })
    return {"raw_output": response.content}


def parse_output(state: State) -> dict:
    movie = parser.parse(state["raw_output"])   # text -> Movie object
    return {"movie": movie.model_dump()}        # store as a plain dict in the state


builder = StateGraph(State)
builder.add_node("ask_llm", ask_llm)
builder.add_node("parse_output", parse_output)
builder.add_edge(START, "ask_llm")
builder.add_edge("ask_llm", "parse_output")
builder.add_edge("parse_output", END)
graph = builder.compile()


if __name__ == "__main__":
    initial = {"genre": "science fiction", "raw_output": "", "movie": {}}
    run_and_show(graph, initial)
