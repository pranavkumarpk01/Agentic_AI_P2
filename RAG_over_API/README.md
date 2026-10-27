# RAG over API

A small project that teaches one idea:

> **An LLM answers best when you give it two kinds of knowledge:**
> 1. **Private, static knowledge** → fetched with **RAG** (our HR policy in a vector database)
> 2. **Public, live knowledge** → fetched from a **third-party API** (web search)

Then both are placed in one prompt and the LLM writes the final answer.

---

## 1. The problem

Ask: *"What is our maternity leave, and has the law changed recently?"*

| Source | Knows our HR policy? | Knows today's law? |
|---|---|---|
| LLM alone | No (never saw it) | Only up to its training date |
| RAG alone | Yes | No (documents are a snapshot) |
| API alone | No | Yes |
| **RAG + API + LLM** | **Yes** | **Yes** |

---

## 2. Architecture

```
  ONE-TIME INGESTION (ingest/ folder)
  data/hr_policy.md → fetch → preprocess → chunk → embed (Ollama) → Qdrant

 ───────────────────────────────────────────────────────────────────────────────
  AT QUESTION TIME  (integration.py controls this flow)

  User question
       │
       ▼
  STEP 1  RAG ── query.py ── embed question ── search Qdrant ──► policy chunks
       │                                                              │
       ▼                                                              │
  STEP 2  ROUTER ── router.py ── Gemini sees (question + policy chunks) ◄┘
       │             and answers:  DECISION: API  or  NO_API
       │
       ├── NO_API ───────────────────────────────┐
       │                                         │
       └── API ──► STEP 3  api.py ──► Tavily     │
                     live web results ───────────┤
                                                 ▼
                              STEP 4  ONE prompt (policy [+ web]) ──► Gemini ──► Final answer
```

**Key point for students:** the LLM is never "connected" to Qdrant or Tavily.
*Our code* fetches the data and pastes it into the prompt. The LLM only reads the prompt.
Even the API decision is just another prompt: Gemini replies `API` or `NO_API`, and *our code* acts on it.

---

## 2A. ⭐ How the system decides WHEN to call the API (`router.py`)

This is the "RAG over API" idea. The API is not called blindly; the LLM acts as a **router**.

### The idea in one sentence
> RAG runs first. Then Gemini looks at the **question** and at **what RAG found**, and answers one
> narrow question: *"Is the policy enough, or do we need live information from the web?"*

### Step by step (this matches the code in `router.py` and `integration.py`)

1. **RAG always runs first.** It is cheap, local and free, and its result is the best evidence of
   whether our own documents can answer the question.
2. **Router prompt.** `router.py` sends Gemini the question plus the retrieved policy chunks
   with clear rules (see `ROUTER_PROMPT` in the file):
   - policy fully answers it (leave counts, notice period, working hours) → `NO_API`
   - question asks for *current / latest / recent* information → `API`
   - policy only *refers to* an outside law or authority ("as per the EPFO rates") without the actual
     details → `API`
   - the answer isn't in the policy and it is public knowledge → `API`
3. **Fixed reply format.** Gemini must reply in exactly two lines:
   ```
   DECISION: API        (or NO_API)
   REASON: one short sentence
   ```
   A fixed format lets plain Python parse the reply. No complicated framework is needed.
4. **Temperature 0.** The router call uses `temperature=0` so the same question gets the same decision.
5. **Python acts on the decision.** In `integration.py`:
   ```python
   needs_api, reason = decide(question, rag_hits)   # router.py
   if needs_api:
       api_results = search_web(question)           # api.py  (Tavily)
   else:
       api_results = []                             # API skipped, no credit used
   ```
6. **Safe default.** If Gemini's reply can't be understood, we choose `API`. Extra information is
   safer than a wrong or empty answer.
7. **Answer prompt adapts.** `build_prompt()` only adds the "LIVE WEB INFORMATION" section when the API
   was actually called.

### Examples (what you should see)

| Question | What RAG finds | Router decision | Why |
|---|---|---|---|
| "How many casual leaves do I get?" | "12 casual leaves per year" | **NO_API** | Policy fully answers it |
| "What is the notice period for confirmed employees?" | "60 days" | **NO_API** | Policy fully answers it |
| "What is our maternity leave, and has the law changed recently?" | "as per the Maternity Benefit Act" | **API** | Policy points to an outside law; "changed recently" needs live info |
| "What is the latest EPFO provident fund contribution rate?" | "as per rates set by the EPFO" | **API** | Policy gives no number; "latest" needs live info |
| "Who won the cricket match yesterday?" | unrelated chunks | **API** | Not in the policy; live public info |

> LLM decisions are not perfect. Always show the **Reason** line to students and discuss borderline
> cases. That is a good lesson on why routers need testing.

### Test the router on its own (no API call is made)

```bash
python router.py "How many casual leaves do I get?"
python router.py "What is the latest EPFO provident fund contribution rate?"
```

### Why decide *after* RAG and not before?
- Deciding from the question alone would miss cases like maternity leave, where the question looks
  like a pure policy question but the policy itself points to an external law.
- RAG costs nothing, so running it first gives the router better evidence.

### Trade-offs (discussion points)
- The router costs **one extra Gemini call** per question (free-tier quota applies), but it saves
  Tavily credits and removes noisy web results from simple answers.
- A cheaper alternative is a keyword rule ("latest", "current", "today" → API). It is faster but misses
  cases like maternity leave. A hybrid (rules first, LLM for the unclear cases) is common in production.
- The more advanced version of this idea is **tool calling / function calling**, where the LLM is
  given the API as a "tool" and chooses it itself. Here we built the same idea by hand so every step
  is visible.

---

## 2B. Metadata filtering (`metadata_filtering.py`)

**Problem:** the question "How many days?" is vague. Vector search returns leave days, notice-period days,
reimbursement deadlines, all mixed together.

**Idea:** every chunk is stored with **metadata**. Before ranking by similarity, we tell Qdrant which chunks are allowed.

```
Ingestion (ingest/metadata.py)           Query (metadata_filtering.py)
chunk "Casual leaves 12 per year..."     Filter(category = "leave")
  + {section: "Casual and Sick Leave",     → Qdrant keeps only chunks where category = leave
     category: "leave", section_no: 2}     → then ranks THOSE by similarity
```

The key code (from `metadata_filtering.py`):

```python
Filter(must=[FieldCondition(key="category", match=MatchValue(value="leave"))])
hits = client.query_points(..., query_vector, query_filter=that_filter)
```

| Filter | Meaning |
|---|---|
| `category="leave"` | exactly this category |
| `category=["leave","exit"]` | either one (`MatchAny`) |
| `section_no=7` | one specific section |

- **Filter first, then rank.** Filtering removes wrong-topic chunks before similarity is even computed.
- **Where do filter values come from?** In this demo we pass them by hand. In a real app they come from the user's
  context (department, country, document year) or an LLM that reads the question and picks the filter.
- **Server note:** with a Qdrant *server* you also create a payload index on the field for speed. Local mode doesn't need it.

Run it: `python metadata_filtering.py`

---

## 2C. Re-ranking (`rerank.py`)

**Problem:** vector search embeds the question and each chunk *separately* and compares the numbers.
It is fast but approximate; the best chunk is not always #1.

**Idea:** a two-stage search.

```
Stage 1  Qdrant vector search ──► top 8 candidates      (fast, approximate, wide net)
Stage 2  cross-encoder reads (question + chunk) together
         ──► new score per chunk ──► keep best 3         (slower, more accurate, small list)
                                                │
                                                ▼
                                          sent to the LLM
```

| | Vector search (bi-encoder) | Re-ranker (cross-encoder) |
|---|---|---|
| Reads question and chunk | separately | together |
| Speed | very fast, millions of chunks | slow, so only ~10-50 chunks |
| Accuracy | good | better |

The key code (from `rerank.py`):

```python
pairs = [(question, hit["text"]) for hit in candidates]
scores = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict(pairs)
best = sorted(candidates, key=score, reverse=True)[:3]
```

- The model is open source and runs locally (no API key). It needs `sentence-transformers` (in `requirements.txt`).
- On a tiny 8-chunk document the order often stays the same. The benefit shows on larger document sets;
  be honest about this with students.
- **Metadata filtering and re-ranking combine well:** filter → vector search (wide) → re-rank (narrow) → LLM.
  They are kept as separate files here so each idea is easy to show. They are not wired into `integration.py`.

Run it: `python rerank.py`

---

## 3. Project structure and what each file does

```
RAG_over_API/
├── config.py            # all settings (models, paths, keys) in one place
├── ingest/              # everything about LOADING data into Qdrant
│   ├── fetch.py         # 1. read files from data/
│   ├── metadata.py      # 2. split by section + attach metadata (section, category)
│   ├── preprocess.py    # 2. clean the text
│   ├── chunk.py         # 3. split into small chunks
│   ├── embed.py         # 4. text → vector using Ollama
│   ├── vector_store.py  # 5. create collection, store vectors, search
│   └── ingest.py        # runs steps 1-5 in order
├── query.py             # RAG: question → top matching chunks
├── metadata_filtering.py # concept demo: search only chunks matching a metadata condition
├── rerank.py            # concept demo: re-score search results with a cross-encoder
├── router.py            # ⭐ DECISION: Gemini decides API or NO_API
├── api.py               # API: question → live web results (Tavily)
├── llm.py               # tiny helper that calls Gemini (used by router + integration)
├── integration.py       # RAG → router → (API if needed) → prompt → Gemini → answer
├── data/hr_policy.md    # sample HR policy (replace with yours)
├── requirements.txt
├── .env.example         # template for secrets
├── README.md            # this file
└── SETUP.md             # step-by-step setup
```

---

## 4. Key concepts (30-second versions)

- **Chunking** – splitting a long document into small pieces so retrieval is precise.
- **Embedding** – converting text into a list of numbers; similar meaning → similar numbers.
- **Vector database (Qdrant)** – stores embeddings and quickly finds the ones closest to a query.
- **Retrieval** – embed the question, find the nearest chunks.
- **Cosine similarity** – the "closeness" score (higher = more similar).
- **API call** – an HTTP request (with an API key for authentication) that returns JSON.
- **Context / prompt stuffing** – pasting retrieved data into the prompt so the LLM can use it.
- **Grounding** – telling the LLM to answer only from the provided context, to reduce hallucination.

---

## 5. Walk-through of one question

Question: *"What is our maternity leave, and has the law changed recently?"*

1. **RAG** – `query.py` embeds the question, Qdrant returns the policy chunk about
   "Maternity and Paternity Leave". It says leave is *"as per the Maternity Benefit Act"*,
   which points to an external law but doesn't say what the law currently is.
2. **Router** – `router.py` shows Gemini the question + that chunk. Gemini replies
   `DECISION: API` because the policy only refers to an outside law and the question says "recently".
3. **API** – `api.py` sends the question to Tavily, which returns current web articles.
4. **LLM** – `integration.py` puts both into one prompt. Gemini answers:
   company policy says X (source: policy), the law currently says Y (source: web).

For "How many casual leaves do I get?" the router replies `NO_API`, step 3 is skipped, and the
answer comes from the policy alone.

---

## 6. Honest limitations (good discussion points)

- **The router can be wrong.** It may skip the API when it was needed (stale answer) or call it
  when it wasn't (wasted credit). Test it with many sample questions.
- **Two Gemini calls per API-or-not question** (router + answer) use more of the free quota.
- **Web results can be wrong or irrelevant.** The LLM may trust them too much.
- **Quotas.** Tavily free tier = 1,000 credits/month; Gemini free tier has rate limits.
- **Free-tier Gemini data may be used to improve Google products.** Don't send confidential
  HR data through the free tier in real projects.
- **Fixed-size chunking** can split ideas in half; smarter chunking (by headings) is better.
- Qdrant local mode allows **one process at a time**; use a Qdrant server for real apps.

---

## 7. Ideas to extend later

- Add a keyword pre-check before the LLM router (cheaper), or switch to Gemini function calling.
- Add more APIs and let the router pick *which* one (e.g. web search vs leave-balance API).
- Replace Tavily with an internal API (e.g. leave-balance API) – same pattern, different URL.
- Add metadata filters in Qdrant (e.g. policy year, department).
- Wrap `ask()` in a FastAPI endpoint.

See **SETUP.md** to run everything.
