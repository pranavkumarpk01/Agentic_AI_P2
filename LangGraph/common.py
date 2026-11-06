# common.py - shared helpers used by every example.
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # reads GOOGLE_API_KEY from the .env file


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model. Every example uses this one function."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key (see setup.md)."
        )
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=temperature,
    )


def show_state(label: str, state: dict, max_len: int = 300) -> None:
    """Print the state in a readable way. Long values are shortened."""
    print(f"\n=== {label} ===")
    for key, value in state.items():
        text = str(value).replace("\n", " | ")
        if len(text) > max_len:
            text = text[:max_len] + "..."
        print(f"  {key}: {text}")


def run_and_show(graph, initial_state: dict, config: dict | None = None) -> dict:
    """Run a graph and print the state BEFORE and AFTER every step."""
    final = initial_state
    for i, state in enumerate(graph.stream(initial_state, config, stream_mode="values")):
        show_state("BEFORE (input)" if i == 0 else f"AFTER step {i}", state)
        final = state
    return final
