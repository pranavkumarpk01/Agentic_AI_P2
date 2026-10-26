# Advanced Retrieval

Five demos that show a retrieval problem (BEFORE) and its fix (AFTER) in the terminal.
**Full teaching instructions: `TEACHING_GUIDE.pdf`.**

| Script | Shows |
|---|---|
| `01_table_aware_chunking.py` | Tables in a PDF: find them, keep headers with every row, sentence-chunk the rest |
| `02_metadata_filtering.py` | Attach metadata to chunks, filter before ranking |
| `03_hybrid_retrieval.py` | BM25 keyword search + vector search, merged with RRF |
| `04_query_rewriting.py` | Gemini rewrites follow-up, messy and vague questions (works offline too) |
| `05_store_in_qdrant.py` | All chunks stored in Qdrant: points, payload, filtered search |

## Run

```bash
cd "$HOME/Documents/Agentic_AI_P2/Advanced Rag/production_rag"
source .venv/bin/activate
pip install -r advanced_retrieval/requirements.txt
docker start qdrant                 # only for script 05
cd advanced_retrieval
python 01_table_aware_chunking.py   # Enter moves to the next step; add --no-pause to run straight through
```

Run 01 before 05. Gemini key and `QDRANT_URL` are read from `production_rag/.env`.

## Files

- `rag_core.py`: shared code (load documents, sentence chunking, embedding, BM25, RRF)
- `display.py`: terminal formatting only
- `data/employee_handbook.pdf`: handbook with two tables (01, 05)
- `data/docs/`: six policy documents with a metadata header (02-05)
- `data/llm_cache.json`: saved Gemini answers used when there is no key or quota (04)
