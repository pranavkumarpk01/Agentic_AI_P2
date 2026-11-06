# LangGraph Basics with Gemini

Nine small, separate examples that each teach one LangGraph idea. Every example uses Google Gemini as the LLM.
Setup steps are in [setup.md](setup.md).

## Three words you need to know

- **State**: a dictionary that holds the data while the graph runs (for example `question`, `answer`).
- **Node**: a normal Python function. It reads the state and returns only the keys it wants to change.
- **Edge**: the arrow that says which node runs next. A *conditional edge* picks the next node based on the state.

Each example prints the state **BEFORE** the graph starts and **AFTER** every step, so you can see exactly what changed.

## Files

| # | File | What it teaches |
|---|------|-----------------|
| 1 | `01_llm_calling.py` | Calling Gemini from a graph node |
| 2 | `02_prompt_template.py` | Filling placeholders like `{topic}` from the state |
| 3 | `03_output_parser.py` | Turning LLM text into a clean Python object |
| 4 | `04_human_in_loop.py` | Pausing the graph so a human can approve or give feedback |
| 5 | `05_parallel_execution.py` | Running several nodes at the same time, then joining |
| 6 | `06_tool_calling.py` | Letting the LLM call Python functions (tools) |
| 7 | `07_conditional_execution.py` | Routing to different nodes based on the state |
| 8 | `08_retry_mechanism.py` | Retrying when a call fails or the answer is invalid |
| 9 | `09_multi_agent.py` | A supervisor agent directing researcher and writer agents |
| - | `common.py` | Shared helpers: Gemini setup and state printing |
| - | `requirements.txt` | Python libraries to install |
| - | `.env.example` | Template for your API key |

## Example walkthroughs

### 1. LLM calling
One node, one call. `call_llm` sends `question` to Gemini and stores the reply in `answer`.

### 2. Prompt template
`build_prompt` fills the template with `topic` and `style` and stores the text in `prompt`. `call_llm` then uses that text. Watch `prompt` go from empty to filled.

### 3. Output parser
`ask_llm` gets plain text into `raw_output`. `parse_output` converts it into a `Movie` object and saves it as `movie`.

### 4. Human in the loop
`human_review` calls `interrupt()`, which pauses the graph. The script shows the draft, you type `yes` or feedback, and `Command(resume=...)` continues the graph. Feedback sends the graph back to `write_draft`. A `MemorySaver` checkpointer remembers the pause.

### 5. Parallel execution
`pros`, `cons` and `examples` all start from `START`, so they run together. `results` uses `operator.add` so each node appends its result without overwriting the others. `combine` runs only after all three finish.

### 6. Tool calling
`add_numbers` and `get_weather` are normal Python functions marked with `@tool`. `bind_tools` tells Gemini they exist. If Gemini asks for a tool, `tools_condition` sends the flow to `ToolNode`, which runs it and returns the result to the agent.

### 7. Conditional execution
`classify` labels the question as `math`, `creative` or `general`. `route_by_category` reads that label and picks the matching expert node.

### 8. Retry mechanism
`call_llm` records an `error` if the API fails or the answer is not a number from 1 to 10. The router sends the flow back to `call_llm` until it succeeds or `attempts` reaches `max_attempts`. Then `give_up` runs.
In production you can also use LangGraph's built-in retry: `builder.add_node("call_llm", call_llm, retry_policy=RetryPolicy(max_attempts=3))` with `from langgraph.types import RetryPolicy`.

### 9. Multi-agent system
A `supervisor` reads the task and what is already done, then chooses `researcher`, `writer` or `finish`. Each worker does its job and returns control to the supervisor. `steps` is a safety limit so the loop always ends.

## Run order suggestion
Run 1 to 9 in order. Each file builds on the idea in the one before it.
