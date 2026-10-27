# SETUP.md – Run RAG over API from scratch (macOS, Apple Silicon)

Time needed: about 20 minutes.

---

## 0. What you need

| Item | Cost | Where |
|---|---|---|
| Python 3.12 | Free | python.org or `brew install python@3.12` |
| VS Code | Free | code.visualstudio.com |
| Ollama (runs the embedding model locally) | Free | ollama.com |
| Gemini API key (LLM) | Free tier (rate limited) | https://aistudio.google.com/apikey |
| Tavily API key (web search API) | Free tier: 1,000 credits/month, no card | https://app.tavily.com |
| Qdrant | Free – runs as a local folder, **no install, no Docker** | installed by pip |

---

## 1. Open the project in VS Code

```bash
cd path/to/RAG_over_API
code .
```

Open the VS Code terminal: **Terminal → New Terminal**. All commands below run from the
`RAG_over_API` folder.

---

## 2. Create and activate a virtual environment

A virtual environment keeps this project's packages separate from the rest of your Mac.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. Check:

```bash
python --version      # should print Python 3.12.x
```

> In VS Code: press `Cmd+Shift+P` → "Python: Select Interpreter" → choose `.venv`.
> Every time you open a new terminal, run `source .venv/bin/activate` again.

---

## 3. Install the Python packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Install Ollama and download the embedding model

1. Install the Ollama app from https://ollama.com/download and open it once
   (it runs a small server on `http://localhost:11434`).
2. Download the open-source embedding model (~270 MB):

```bash
ollama pull nomic-embed-text
```

3. Check that Ollama can create embeddings:

```bash
curl http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"hello"}'
```

You should see JSON containing a long list of numbers under `"embeddings"`.

> If you want to use a different Ollama embedding model (e.g. `mxbai-embed-large`),
> pull it, then set `EMBED_MODEL=mxbai-embed-large`, `DOC_PREFIX=` and `QUERY_PREFIX=`
> (empty) in `.env`. Re-run ingestion afterwards (different models → different vectors).

---

## 5. Get your API keys

**Gemini (LLM)**
1. Go to https://aistudio.google.com/apikey and sign in with your Google account.
2. Click **Create API key** and copy it.

**Tavily (web search API)**
1. Go to https://app.tavily.com and sign up.
2. Copy the API key from the dashboard (starts with `tvly-`).

---

## 6. Create the `.env` file (your secrets)

```bash
cp .env.example .env
```

Open `.env` in VS Code and paste your keys:

```
GEMINI_API_KEY=AIza...your key...
TAVILY_API_KEY=tvly-...your key...
```

Never share or commit `.env` (it is already in `.gitignore`).

---

## 7. Add your HR policy

A sample policy is already in `data/hr_policy.md`.
To use your own, copy your file(s) into the `data/` folder (`.txt`, `.md` or `.pdf`).
You can delete the sample file.

---

## 8. Ingest: store the data in Qdrant

Make sure Ollama is running, then:

```bash
python -m ingest.ingest
```

Expected output (numbers will vary with your document):

```
[1] Fetched 1 document(s): ['hr_policy.md']
[2-3] Split into sections, tagged metadata, made 29 chunks
      example metadata: {'source': 'hr_policy.md', 'section_no': 1, 'section': 'Working Hours', 'category': 'work'}
[4] Created 29 embeddings (each has 768 numbers)
[5] Stored 29 points in Qdrant. Ingestion complete!
```

Each chunk is stored with **metadata** (section name, section number, category) next to its vector.
This is what `metadata_filtering.py` uses later. (If you ingested before this version, run it again.)

A new folder `qdrant_data/` appears. That folder **is** your vector database.

Re-running ingestion deletes the old collection and rebuilds it, so it is safe to repeat.

---

## 9. Fetch data from the database (RAG retrieval only)

```bash
python query.py "How many casual leaves do I get?"
```

Expected: the top 3 chunks with similarity scores; the casual-leave chunk should be #1.

```
#1  score=0.7xx  source=hr_policy.md
    2. Casual and Sick Leave Every employee gets 12 casual leaves ...
```

---

## 8A. About the sample policy

`data/hr_policy.md` has 22 sections, tagged automatically into 5 categories:

| Category | Sections |
|---|---|
| leave | Casual and Sick, Earned, Maternity and Paternity, Public Holidays, Leave Encashment, Bereavement and Marriage, Leave Without Pay and Sabbatical |
| work | Working Hours, Work From Home, Attendance and Timesheets |
| benefits | Provident Fund and Gratuity, Reimbursements, Health Insurance, Bonus and Increments, Learning and Development, Travel Reimbursement, Payroll and Tax Deduction |
| exit | Notice Period, Resignation and Exit Process, Full and Final Settlement |
| general | Code of Conduct and POSH, Performance Review |

Questions that show each idea (try them with `integration.py`, `metadata_filtering.py` or `rerank.py`):

| Question | What it shows |
|---|---|
| "How many casual leaves do I get?" | Policy answers it, router says NO_API |
| "What is the notice period for confirmed employees?" | Policy answers it, NO_API |
| "How many days?" with filter `leave` vs `exit` | Metadata filtering: leave days vs notice and settlement days |
| "Can I encash my leave when I resign?" | Several leave and exit sections compete; good for re-ranking |
| "What is the health insurance cover for my parents?" | Policy answers it, NO_API |
| "What is the latest EPFO provident fund contribution rate?" | Policy only refers to EPFO, router says API |
| "What is our maternity leave, and has the law changed recently?" | Mixed: policy plus live web, router says API |
| "How is income tax deducted from my final settlement and what are the current tax slabs?" | Policy plus live web, API |

---

## 9A. Metadata filtering (same document)

```bash
python metadata_filtering.py
```

The question "How many days?" is deliberately vague. The script runs it four times:
no filter, `category = leave`, `category = exit`, and `leave OR exit`.
Expected: with no filter you get a mix of topics; with a filter every result belongs to that category.

Try your own question and filter:

```bash
python metadata_filtering.py "How many days?" exit
```

---

## 9B. Re-ranking (same document)

```bash
python rerank.py
```

The first run downloads a small open-source model (about 90 MB) from Hugging Face, so it needs internet
once and takes a minute. Expected: the list of 8 candidates in vector-search order, then the best 3
after re-ranking with new scores. Sometimes the order changes, sometimes it does not (see README section 2C).

```bash
python rerank.py "How long is paternity leave?"
```

---

## 10. Test the API alone

```bash
python api.py "latest changes to maternity benefit law India"
```

Expected: 3 web results with title, URL and a text snippet.

---

## 11. Test the router (decides API or no API)

This uses Gemini only. No Tavily credit is spent.

```bash
python router.py "How many casual leaves do I get?"
python router.py "What is the latest EPFO provident fund contribution rate?"
```

Expected (wording of the reason will vary):

```
Question : How many casual leaves do I get?
Decision : POLICY ONLY (no API call)
Reason   : The policy states the number of casual leaves directly.

Question : What is the latest EPFO provident fund contribution rate?
Decision : CALL API
Reason   : The policy only refers to EPFO rates and the question asks for the latest rate.
```

---

## 12. Run the full application (RAG → router → API if needed → Gemini)

Question that needs the API:

```bash
python integration.py "What is our maternity leave, and has the law changed recently?"
```

Expected flow:

```
STEP 1: RAG    - searching our HR policy in Qdrant...
STEP 2: ROUTER - asking Gemini: do we need the API?
   Decision: CALL API
   Reason  : ...
STEP 3: API    - calling the Tavily web search API...
STEP 4: LLM    - generating the final answer with Gemini...

ANSWER:
Company policy: ... (from internal HR document)
Web information: ... (from search results)
```

Question that does NOT need the API:

```bash
python integration.py "How many casual leaves do I get?"
```

Expected: `Decision: NO API - policy is enough`, then `STEP 3: API - skipped`, and an answer from the policy only.

Also try:

```bash
python integration.py "What is the notice period for confirmed employees?"
python integration.py "What is the latest EPFO provident fund contribution rate?"
```

Each normal question uses 2 Gemini calls (router + answer) and 0 or 1 Tavily credit.

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama` | Open the Ollama app, or run `ollama serve` in another terminal |
| `model "nomic-embed-text" not found` | Run `ollama pull nomic-embed-text` |
| `ModuleNotFoundError` | Activate the venv: `source .venv/bin/activate`, then `pip install -r requirements.txt` |
| `No module named 'ingest'` | Run commands from the project root; use `python -m ingest.ingest` (not `python ingest/ingest.py`) |
| `Storage folder qdrant_data is already accessed by another instance` | Another Python process has Qdrant open. Close it and retry |
| `Collection ... doesn't exist` / empty results | Run `python -m ingest.ingest` first |
| `KeyError: 'category'` or `'section'` | Data was ingested with the old version. Run `python -m ingest.ingest` again |
| `No '## ' sections found` | Metadata splitting needs headings like `## 1. Working Hours` in your document |
| `rerank.py` is slow the first time | It is downloading the cross-encoder model once; later runs are fast |
| `GEMINI_API_KEY is missing` / `TAVILY_API_KEY is missing` | Check `.env` exists in the project root and has no spaces around `=` |
| Gemini `404 model not found` | Models change over time. Open https://ai.google.dev/gemini-api/docs/models, pick a current Flash model and set `GEMINI_MODEL=` in `.env` |
| Gemini `429` / quota errors | Free-tier rate limit. Wait a minute, or use a lighter model such as a Flash-Lite variant |
| Tavily `401` | Wrong key or missing `tvly-` prefix |
| Retrieval returns wrong chunks | You changed the embedding model without re-running ingestion |

---

## 14. Reset everything

```bash
rm -rf qdrant_data       # deletes the vector database
python -m ingest.ingest  # rebuilds it
```

---

## 15. Suggested 20-minute teaching flow

1. Show `README.md` architecture diagram (2 min).
2. Show `ingest/` files one by one, then run ingestion (5 min).
3. Run `query.py` – show scores and chunks (3 min).
4. Run `metadata_filtering.py` – same vague question, different filters (3 min).
   Then `rerank.py` – before/after list (3 min).
5. Run `api.py` – show raw results (2 min).
6. Open `router.py` – show `ROUTER_PROMPT` and the DECISION format, run it on 2 questions (4 min).
7. Open `integration.py`, show `build_prompt()` and the `if needs_api` branch (2 min).
8. Run 3 questions: policy-only, API-only, and a mixed one (4 min).
