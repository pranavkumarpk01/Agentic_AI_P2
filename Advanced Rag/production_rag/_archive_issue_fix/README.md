# Production RAG: three ingestion failures, side by side

You run the same pipeline twice and compare:

- **Issue version** (`ingest_issue/`, `query_issue/`): the common way, with the problems.
- **Fix version** (`ingest_fix/`, `query_fix/`): the same files with the fixes. Changes are marked `# ===== FIX ...` in the code.

For the full explanation of each problem, its example and its fix, read `ISSUES_AND_FIXES.md`. This README is the run order.

Nothing to comment or uncomment. Each version writes to its own Qdrant table (`policies_issue`, `policies_fix`).

| # | Failure | Fix | Compare these files |
| --- | --- | --- | --- |
| 1 | **The table never made it in**: a PDF table is read as loose text, and a table over two pages is cut in the middle | Find tables, join the split table, write each row as `Column: value`, reject the file if a PDF number is lost | `fetch.py`, `quality_gate.py` |
| 2 | **The answer got cut in half**: cutting every 300 characters separates a rule from its exception | Split on headings, pack whole sentences, store a context line, fetch neighbours of every hit | `chunking.py`, `query_*/retrieval.py` |
| 3 | **Old and new versions share one index**: the 2023, 2025 and 2027 leave policies all match | Give each file a version and date, let code decide the status, search "current only" always | `fetch.py`, `query_*/retrieval.py` |

---

## Step 0. Setup (once)

```bash
cd "$HOME/Documents/Agentic_AI_P2/Advanced Rag/production_rag"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and paste your key after `GEMINI_API_KEY=` (free key from Google AI Studio). Leave `QDRANT_URL` empty: Qdrant then runs inside Python and saves to `qdrant_local/`, so **no Docker is needed**. Only one command can use it at a time.

Run every command below from this folder with the venv active.

---

## Part A. The issue version (show the problems)

### A1. Build the database

```bash
python -m ingest_issue.main --reset
```

Read the output stage by stage (1 fetch, 2 preprocess, 3 chunk, 4 gate, 5 embed, 6 store). Point at stage 3:

```text
[chunk 3] loan_rate_guide.pdf #0
  | Loan type Term Interest rate Processing fee Maximum amount
  | Personal loan 1 year 12
[chunk 4] loan_rate_guide.pdf #1
  | .5% 2.0% Rs 5,00,000 ...
```
```text
[chunk 16] refund_policy.md #1
  | ... Electronics may be returned within 1
[chunk 17] refund_policy.md #2
  | 5 days of delivery, and only if the item is unopened and sealed.
```

Say: "12.5% became 12 and .5%. 15 days became 1 and 5. Column names are far from the numbers. Chunks start and end mid-sentence."

The summary at the end: 8 files, 17 chunks, only 3 of 17 end at a full stop.

### A2. Look inside Qdrant

```bash
python show_db.py issue
```

Shows each stored point: id, text, vector preview. No version, no section, nothing but raw text.

### A3. Ask the three questions

```bash
python -m query_issue.main "What is the interest rate for a 5-year personal loan?" --k 2 --no-answer
python -m query_issue.main "How many days do I have to return unopened electronics?" --k 1 --no-answer
python -m query_issue.main "How many weeks of paid parental leave do employees get?" --k 3 --no-answer
```

| Question | What goes wrong |
| --- | --- |
| 5-year personal loan rate | The text says "11.0% 1.5%" with no column names. Rate or fee? |
| Unopened electronics return days | One broken chunk. The exception is in the next chunk |
| Weeks of paid parental leave | 2023, 2025, 2027 and the draft note ("26 weeks") all come back with near-equal scores, labelled `(no version info stored)` |

Say: "The model did not lie. We gave it the wrong material." (Remove `--no-answer` if you want them to see Gemini answer confidently from it.)

### A4. Show the quality gate rejecting the issue chunks

```bash
python gate_demo.py
```

```text
  REJECTED loan_rate_guide.pdf: 1 number(s) from the PDF are missing from the chunks: ['12.5']
```

Say: "Every number in the PDF must still exist in the chunks. One was cut in half, so the file is rejected. The issue pipeline has no gate."

---

## Part B. The fix version (show the solutions)

### B1. Build the database

```bash
python -m ingest_fix.main --reset
```

What to point at, in output order:

- **Stage 1** warns `draft_notes.md has no version information, so it is NOT indexed` and shows statuses: 2023 `superseded`, 2025 `current`, 2027 `upcoming`. The PDF arrives as `[text]` and `[table]` pieces.
- **Stage 3** shows table rows as full sentences, and section chunks with a context line:

```text
Loan interest rates and fees - Loan type: Personal loan; Term: 5 years; Interest rate: 11.0%; Processing fee: 1.5%; Maximum amount: Rs 15,00,000.
```
- **Stage 4** says `OK loan_rate_guide.pdf: all 28 numbers from the PDF are in the chunks`.
- **Stage 5** shows the text sent to Gemini: context line + chunk.
- **Summary:** 7 files, 19 chunks, 19 of 19 end at a full stop.

### B2. Look inside Qdrant

```bash
python show_db.py fix
```

Each point now has a section, a context line (for example `Refund and Returns Policy > Exceptions for electronics`) and a status.

### B3. Ask the same three questions

```bash
python -m query_fix.main "What is the interest rate for a 5-year personal loan?" --k 2 --no-answer
python -m query_fix.main "How many days do I have to return unopened electronics?" --k 1 --no-answer
python -m query_fix.main "How many weeks of paid parental leave do employees get?" --k 3 --no-answer
```

| Question | What is fixed |
| --- | --- |
| 5-year personal loan rate | Every number has its column name next to it |
| Unopened electronics return days | 3 passages: the hit and its two neighbours, so the exception is included |
| Weeks of paid parental leave | Only `leave-policy v2025 (current)` |

### B4. Show the code behind each fix

Open the issue file and the fix file side by side (select both in VS Code, right-click, **Compare Selected**).

| Failure | Files | What to explain |
| --- | --- | --- |
| 1 | `fetch.py` `read_pdf_with_tables()` | `find_tables()` finds grids; `outside_bbox` keeps the other text apart; equal headers mean the table continues, so join rows and drop the repeated header; each row becomes `Title - Column: value; ...` |
| 1 | `quality_gate.py` | Compare the numbers found in the PDF with the numbers in the chunks. Any missing means reject |
| 2 | `chunking.py` `split_into_chunks()` | Split on headings; merge tiny sections; pack whole sentences up to 500 characters; add the context line |
| 2 | `query_fix/retrieval.py` `expand_with_neighbours()` | Also fetch the chunk before and after each hit |
| 3 | `fetch.py` `add_status()` and `data/manifest.json` | Each file has `doc_key`, `version`, `effective_date`. Newest version whose date has passed is current. A file with no entry is not indexed |
| 3 | `query_fix/retrieval.py` | "Current only" filter on every search |

Say: "Changing the chunk size does not help, because every size breaks something. Cut where the document is already cut, and bring the neighbours along."

---

## Part C. Optional extras

```bash
# A history question opts in to old versions (2023 appears, 2027 never does)
python -m query_fix.main "How many weeks of parental leave did employees get in the past?" --k 3 --history --no-answer

# Pretend it is 2027: the 2027 policy becomes current, 2025 superseded
AS_OF_DATE=2027-02-01 python -m ingest_fix.main --reset
python -m query_fix.main "How many weeks of paid parental leave do employees get?" --k 3 --no-answer

# Put the date back
python -m ingest_fix.main --reset

# Run ingestion twice without --reset: issue doubles the points (34), fix overwrites (19)
python -m ingest_issue.main
python -m ingest_fix.main
```

Production note: the status is stored at ingestion time, so re-run ingestion (a nightly job is enough) after an effective date passes.

Smaller changes, each a pair of files to compare. The last three call Gemini, so try each once before class:

| File | Issue | Fix | Try |
| --- | --- | --- | --- |
| `preprocessing.py` | Page headers and footers stay in chunks | `clean()` removes them | |
| `embedding.py` | Embeds the chunk text only | Embeds context line + text | |
| `database.py` | Random IDs, duplicates on re-run | IDs from file name + chunk number | |
| `rewrite.py` | Searches the question as typed | Gemini rewrites it in policy wording | `"can I send back a gadget I haven't opened yet?"` |
| `rerank.py` | Trusts the vector order | Gemini scores candidates, keeps the best | `"What is the maximum I can spend on a hotel per night?"` |
| `generation.py` | Bare prompt | Answer only from context, say "I don't know", cite [n] | `"Who is the CEO of Zentra Retail?" --show-prompt` |

Question options: `--k` (chunks kept), `--candidates` (chunks fetched first), `--no-answer` (skip Gemini), `--show-prompt`, and `--history` (fix only).

---

## Reference

### Folder map

```text
production_rag/
├── config.py          settings, today's date, Gemini client, Qdrant client
├── show_db.py         python show_db.py issue|fix
├── gate_demo.py       quality gate vs the issue chunks
├── data/manifest.json identity of each file (title, doc_key, version, effective_date)
├── data/raw/          finance (loan PDF, expense policy), hr (3 leave policies),
│                      retail (refund policy), general (travel, draft_notes with no version)
├── ingest_issue/ and ingest_fix/     (same file names in both)
│   ├── fetch.py  preprocessing.py  chunking.py  quality_gate.py  embedding.py  database.py
│   └── main.py        runs stages 1 to 6
└── query_issue/ and query_fix/
    ├── rewrite.py  retrieval.py  rerank.py  generation.py
    └── main.py        runs stages 7 to 11
```

### Expected numbers (when you run it)

| | Issue | Fix |
| --- | --- | --- |
| Files read | 8 | 7 (draft not indexed) |
| Chunks | 17 | 19 |
| Chunks ending at a full stop | 3 of 17 | 19 of 19 |
| Points after running ingestion twice | 34 | 19 |

### Qdrant dashboard (optional, needs Docker)

```bash
docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Set `QDRANT_URL=http://localhost:6333` in `.env`, re-run both ingestions, then open `http://localhost:6333/dashboard`. Next time: `docker start qdrant`.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named 'config'` | Run from the project folder and use `python -m ingest_fix.main` |
| `No module named pdfplumber` (or `qdrant_client`, `google`, `dotenv`) | `source .venv/bin/activate`, then `pip install -r requirements.txt` |
| `No API key was provided` | `cp .env.example .env` and paste the key |
| `Connection refused` | `QDRANT_URL` is set but Qdrant is not running. `docker start qdrant`, or empty the line |
| Model not found (404) | Put a model your key can use in `.env` (`GEMINI_CHAT_MODEL`, `GEMINI_EMBED_MODEL`) |
| 429 / quota | Wait a minute, use `--no-answer`, ingest once per version |
| Vector dimension error | Re-run with `--reset` |
| `Collection ... doesn't exist` | Ingest that version first (`ingest_issue` for `query_issue`, `ingest_fix` for `query_fix`) |
| Local storage "already in use" | Another command still has `qdrant_local/` open. Wait, or use Docker |
| `gate_demo.py` says OK, not REJECTED | A different pdfplumber version laid out the text differently, so no number was cut. Show the broken rows from A1 and explain the gate with the code |
| `JSONDecodeError` in `rerank.py` | Gemini returned non-JSON. Run again or try another model |
| Counts differ from this README | Use `--reset` (except when testing duplicates) |

### What was tested

Run end to end on Linux with Qdrant in local-folder mode and Gemini replaced by a stand-in (word-hash embedder and a stub text model): both ingestions (also twice), `show_db.py`, `gate_demo.py`, all questions on both query versions, `--history` and `AS_OF_DATE`. So chunk counts, the gate message, statuses and neighbour passages are exact; retrieval scores and rankings are not.

**Not run:** real Gemini embeddings and answers (rewrite, rerank, grounded answer), Docker Qdrant, a Mac with Python 3.12. Default model names are unchecked. The sample PDF is tuned so a 300-character cut lands inside "12.5%" with pdfplumber 0.11.

### Dry run before class (10 minutes)

Do Parts A and B once with your key. Check 17 and 19 chunks, REJECTED in A4, `OK ... all 28 numbers` in B1, 1 passage vs 3 for the electronics question, and several versions vs only `v2025 (current)` for the leave question. Try the three Gemini extras and keep the ones where the difference is visible.
