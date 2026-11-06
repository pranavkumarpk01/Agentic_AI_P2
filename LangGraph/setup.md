# Setup Guide (LangGraph + Gemini)

Works on macOS and Linux. Windows notes are at the end.

## 1. Prerequisites
- Python 3.10 or newer. Check with: `python3 --version`
- A Google account (for a free Gemini API key)

## 2. Get a Gemini API key
1. Open https://aistudio.google.com/apikey
2. Click **Create API key** and copy it.

## 3. Open the project folder
```bash
cd ~/Documents/Agentic_AI_P2/LangGraph
```

## 4. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Your prompt should now start with `(.venv)`.

## 5. Install the libraries
```bash
pip install -r requirements.txt
```

## 6. Add your API key
```bash
cp .env.example .env
```
Open `.env` in any editor and replace `your_gemini_api_key_here` with your real key.
Never share or commit the `.env` file.

## 7. Run the examples
Run any example from inside this folder:
```bash
python 01_llm_calling.py
python 02_prompt_template.py
python 03_output_parser.py
python 04_human_in_loop.py      # this one asks you questions in the terminal
python 05_parallel_execution.py
python 06_tool_calling.py
python 07_conditional_execution.py
python 08_retry_mechanism.py
python 09_multi_agent.py
```

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `GOOGLE_API_KEY is missing` | Make sure `.env` exists in this folder and has your key. |
| `ModuleNotFoundError` | Activate the venv (`source .venv/bin/activate`) and run `pip install -r requirements.txt` again. |
| `404 model not found` | Set `GEMINI_MODEL` in `.env` to a model available in AI Studio, for example `gemini-2.5-flash` or `gemini-2.0-flash`. |
| `429 quota` / rate limit errors | Wait a minute and run again, or use a lighter model such as `gemini-2.5-flash-lite`. |
| `python3: command not found` | Install Python from https://www.python.org/downloads/ |

## Windows notes
- Create the venv with `python -m venv .venv`
- Activate it with `.venv\Scripts\activate`
- Copy the env file with `copy .env.example .env`
