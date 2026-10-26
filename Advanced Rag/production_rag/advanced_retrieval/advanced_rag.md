# Advanced RAG — Course Notes

> **How to read these notes.** Each topic follows the same order: **the problem** (what goes wrong in basic RAG), **why it happens**, **the fix**, **a real-world example**, and **key points to remember**. Diagrams are plain text, so they display the same everywhere.
>
> The examples come from the two demos we ran in class:
> - **Issue → Fix pipeline** (loan rate PDF, refund policy, parental leave versions, stored in Qdrant)
> - **`advanced_retrieval/` scripts 01–05** (employee handbook tables, leave policies, VPN error codes, travel policy)

---

## Contents

1. [RAG in one page (recap)](#1-rag-in-one-page-recap)
2. [Where basic RAG breaks: the map](#2-where-basic-rag-breaks-the-map)
3. [Cleaning documents before chunking](#3-cleaning-documents-before-chunking)
4. [Chunking strategies](#4-chunking-strategies)
5. [Table-aware chunking](#5-table-aware-chunking)
6. [Quality gate: checking the parser's work](#6-quality-gate-checking-the-parsers-work)
7. [Metadata and metadata filtering](#7-metadata-and-metadata-filtering)
8. [Document versioning (old vs new policies)](#8-document-versioning-old-vs-new-policies)
9. [Embeddings and embedding management](#9-embeddings-and-embedding-management)
10. [The vector database (Qdrant)](#10-the-vector-database-qdrant)
11. [Keyword search: BM25, TF and IDF](#11-keyword-search-bm25-tf-and-idf)
12. [Vector (semantic) search](#12-vector-semantic-search)
13. [Hybrid retrieval: combining BM25 and vectors](#13-hybrid-retrieval-combining-bm25-and-vectors)
14. [Query rewriting (3 types)](#14-query-rewriting-3-types)
15. [Reranking](#15-reranking)
16. [Context management (context window optimisation)](#16-context-management-context-window-optimisation)
17. [Grounded generation](#17-grounded-generation)
18. [RAG + API: combining documents with live data](#18-rag--api-combining-documents-with-live-data)
19. [The complete production pipeline](#19-the-complete-production-pipeline)
20. [Cheat sheet, misconceptions and self-check](#20-cheat-sheet-misconceptions-and-self-check)

---

## 1. RAG in one page (recap)

**RAG = Retrieval-Augmented Generation.** Before the LLM answers, we *retrieve* the most relevant pieces of our own documents and put them in the prompt. The LLM then answers from those pieces instead of from memory.

**Why we need it:** an LLM does not know your company's leave policy, your loan rates or yesterday's price change. Retraining it is slow and expensive. Retrieval gives it the right facts at question time, and it can cite where each fact came from.

```text
                    INGESTION  (once, offline)
  ┌──────────┐   ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐
  │Documents │──►│ Clean  │──►│  Chunk   │──►│  Embed   │──►│ Vector DB    │
  │PDF, docs │   │        │   │+metadata │   │(vectors) │   │ (Qdrant)     │
  └──────────┘   └────────┘   └──────────┘   └──────────┘   └──────┬───────┘
                                                                   │
                    QUERY TIME  (every question)                   │ search
  ┌──────────┐   ┌──────────┐   ┌───────────────┐   ┌─────────┐    │
  │ Question │──►│  Embed   │──►│ Retrieve top-k│◄──┘         │    │
  └──────────┘   │ question │   │    chunks     │──►│  LLM    │──► Answer + sources
                 └──────────┘   └───────────────┘   └─────────┘
```

### Vocabulary

| Term | Meaning | Example |
|---|---|---|
| **Chunk** | A small piece of a document, the unit we search for | One paragraph of the leave policy |
| **Embedding / vector** | A list of numbers that represents the *meaning* of a text | 384 numbers (local MiniLM model) or 768 numbers (Gemini) |
| **Vector database** | Stores vectors and finds the closest ones fast | Qdrant, Chroma, Pinecone, pgvector |
| **Point** | One row in Qdrant: `id` + `vector` + `payload` | The L3 row of the leave table |
| **Payload / metadata** | Facts *about* a chunk: source, year, region, status | `{"region": "India", "year": 2025}` |
| **Top-k** | How many chunks we give the LLM | k = 3 |
| **Cosine similarity** | How close two vectors point in the same direction (−1 to 1) | 0.71 = quite similar |

**The one sentence to remember:** *The LLM can only be as correct as the chunks we hand it.* Every technique in these notes exists to hand it better chunks.

---

## 2. Where basic RAG breaks: the map

Basic RAG fails silently: it always returns *something*, and the LLM answers confidently from it. Each failure below lives at a specific stage, so each fix lives there too.

```text
 INGESTION                                          QUERY TIME
 ──────────────────────────────────────────         ─────────────────────────────────────────
 [3] Noise in text (headers/footers)                [14] Vague / follow-up / misspelt question
 [4] Fixed-size cuts split facts                    [11-13] Keyword OR vector alone misses
 [5] Tables lose their headers                      [7-8] Wrong region / old version returned
 [6] Parser silently drops numbers                  [15] Vector order is only approximate
 [7-8] No metadata / no version info                [16] Context missing neighbours / too long
 [9] Vectors don't know where a chunk came from     [17] LLM guesses when context is missing
 [10] Duplicate points on re-ingestion              [18] Question needs LIVE data (weather, prices)
```

| # | Failure | Fix | Section |
|---|---|---|---|
| 1 | Page headers/footers end up inside chunks | Clean text before chunking | 3 |
| 2 | Fixed-size cuts split numbers and sentences ("12.5%" → "12" + ".5%") | Sentence + structure-aware chunking | 4 |
| 3 | A rule and its exception land in different chunks | Split on headings, context line, neighbour expansion | 4, 16 |
| 4 | Table rows lose their column names | Table-aware chunking | 5 |
| 5 | Parser silently loses a number | Quality gate | 6 |
| 6 | India user gets the US or the 2024 policy | Metadata filtering | 7 |
| 7 | 2023, 2025, 2027 and a draft policy all match | Versioning + "current only" filter | 8 |
| 8 | Vector of "30 days" doesn't know which policy | Contextual embedding | 9 |
| 9 | Running ingestion twice doubles the data | Deterministic IDs + upsert | 10 |
| 10 | "ERR-4012" vs "ERR-4013" confused by vectors | Hybrid search (BM25 + vector) | 11–13 |
| 11 | "Work from my house" finds nothing by keyword | Hybrid search | 11–13 |
| 12 | "what about the US?" has no topic | Contextual query rewriting | 14 |
| 13 | Vector top-k is not the true best order | Reranking | 15 |
| 14 | LLM invents an answer | Grounded prompt: only context, "I don't know", citations | 17 |
| 15 | Question needs today's weather or price | RAG + API | 18 |

---

## 3. Cleaning documents before chunking

**Problem.** Exported documents carry noise: a header on every page (`ZENTRA RETAIL | INTERNAL`), page markers (`--- Page 3`), footers, page numbers. If we keep it, it goes into every chunk, every vector and every prompt.

**Why it hurts.**
- The noise appears in *many* chunks, so it pulls unrelated chunks closer to each other in vector space.
- It wastes tokens in the prompt.
- Keyword search can match on noise words instead of content.

**Fix.** Remove known noise lines *before* chunking:

```python
def clean(text):
    kept = []
    for line in text.splitlines():
        if line.startswith("ZENTRA RETAIL | INTERNAL"):   # repeated page header
            continue
        if line.startswith("--- Page"):                    # page-break marker
            continue
        kept.append(line)
    return "\n".join(kept)
```

**Key points**
- Clean first, chunk second. Noise you remove here never reaches the database.
- In production, detect lines that repeat on most pages (headers/footers) automatically.

---

## 4. Chunking strategies

A **chunk** should hold **one complete idea**. Too small and facts are split; too big and the chunk is about many things, so its vector is "blurry" and it wastes context.

### 4.1 Fixed-size chunking (the problem)

Cut the text every N characters, wherever that falls.

```python
chunks = [text[i : i + 300] for i in range(0, len(text), 300)]
```

**Real example from class (refund policy):**

```text
[chunk 16]  ... Electronics have a shorter window. Electronics may be returned within 1
[chunk 17]  5 days of delivery, and only if the item is unopened and sealed. ...
```

"15 days" became "1" and "5". In the loan PDF, "12.5%" became "12" and ".5%". The retriever finds *one* of these chunks, and the LLM reads a broken number.

> **Why not just change the chunk size?** Every size breaks *something*. The problem is **where** we cut, not how much.

### 4.2 Sentence-based chunking with overlap

Pack **whole sentences** into a chunk until the next sentence would go over the size limit. Never cut inside a sentence. Repeat the last sentence at the start of the next chunk (**overlap**), so a fact on the border appears in both.

```python
def sentence_chunks(text, max_chars=350, overlap=1):
    sentences = split_sentences(text)
    chunks, current = [], []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and len(candidate) > max_chars:     # next sentence would not fit
            chunks.append(" ".join(current))           # close this chunk
            current = current[-overlap:] if overlap else []   # carry the last sentence over
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks
```

**Example (employee handbook):**

```text
chunk 0:  ... Leave entitlement depends on your job grade. ⟨The table below lists the number of days for each grade.⟩
chunk 1:  ⟨The table below lists the number of days for each grade.⟩ Sick leave of more than two consecutive days ...
```
The sentence in ⟨ ⟩ is the overlap.

### 4.3 Structure-aware chunking (headings + context line)

Documents are already cut into sections by their author. Use those cuts.

1. **Split on headings** (`#`, `##`) and remember which heading each piece sits under.
2. **Merge tiny sections** (under ~60 characters, like a one-line "Contact" section) into the previous one.
3. **Pack whole sentences** up to ~500 characters inside each section.
4. **Store a context line** with every chunk: `Document title > Section heading`.

```text
context: Refund and Returns Policy > Exceptions for electronics
text:    Electronics have a shorter window. Electronics may be returned within 15 days
         of delivery, and only if the item is unopened and sealed.
```

The context line tells both the vector (section 9.3) and the LLM *where* the chunk came from. "15 days" alone is meaningless; "Refund policy > Electronics > 15 days" is an answer.

### 4.4 Comparison

| Strategy | How it cuts | Good | Bad |
|---|---|---|---|
| Fixed-size | Every N characters | Simple, predictable size | Splits numbers, sentences, rules |
| Sentence-based | At sentence ends, up to a limit | Never breaks a sentence | Ignores document sections |
| Sentence + overlap | Same, last sentence repeated | Border facts survive | Slight duplication |
| Structure-aware | At headings, then sentences | Keeps a rule with its section, adds context | Needs documents with headings |
| Table-aware | Tables handled separately | Rows keep their headers | Needs a table parser |

**Key points**
- Cut where the document is already cut (headings, paragraphs, sentences).
- Small overlap protects facts on boundaries.
- Keep a context line (document > section) with every chunk.

---

## 5. Table-aware chunking

### The problem

A PDF has no concept of a "table". It stores letters at x/y positions. A text extractor reads left-to-right, top-to-bottom, so a table comes out as plain lines:

```text
Table 1: Leave entitlement by grade
Grade Annual leave (days) Sick leave (days) Carry forward (days)
L1 18 8 5
L2 20 10 5
L3 24 12 10        ← which number is sick leave?
```

Then fixed-size chunking puts the header line in one chunk and `L3 24 12 10` in another:

```text
naive chunk 2:  ' 5\nL3 24 12 10\nL4 26 12 10\nL5 30 15 15\nSick leave of more than ...'
```

**Question:** *"How many sick leave days does an L3 employee get?"* The retriever finds chunk 2. The LLM sees four numbers and no column names, and it guesses.

Real-world: bank rate cards, insurance premium tables, price lists, salary bands, product spec sheets. The most valuable facts are usually *inside* tables.

### The fix: 4 steps

```text
   PDF page
      │
      ▼
 1. FIND tables ──► position (bbox) + caption + rows      page.find_tables()
      │
      ▼
 2. CONVERT each table into chunks that stand alone
      │   small table (≤ 4 rows) → ONE markdown chunk
      │   big table             → ONE chunk PER ROW, caption + headers repeated
      ▼
 3. REMOVE the table area from the page                    page.outside_bbox(bbox)
      │
      ▼
 4. SENTENCE-CHUNK the remaining text (with overlap)
```

**Step 1 — find tables** (pdfplumber detects them from grid lines):

```python
for t in page.find_tables():
    x0, top, x1, bottom = t.bbox                        # where the table is on the page
    caption = page.crop((0, top - 22, page.width, top)).extract_text()   # line just above it
    rows = t.extract()                                  # rows[0] = header, rest = data
```

**Step 2 — every row carries its own headers:**

```python
pairs = [f"{column}: {value}" for column, value in zip(header, row)]
chunk = caption + " | " + " | ".join(pairs)
```

```text
Table 1: Leave entitlement by grade | Grade: L3 | Annual leave (days): 24 |
Sick leave (days): 12 | Carry forward (days): 10
```

Small tables are kept whole as markdown (LLMs read markdown tables well):

```text
Table 2: Laptop allowance by role
| Role | Laptop | Accessory allowance (INR) |
|---|---|---|
| Engineer | MacBook Pro 14 | 15,000 |
| Designer | MacBook Pro 16 | 20,000 |
```

**Step 3 — remove the table** from the page text so the same numbers are not indexed twice (once with headers, once without).

**Step 4 — sentence-chunk** what is left (section 4.2).

### Tables that continue on the next page

In the loan rate guide, Table 1 continued on page 2 with the header row repeated. Rule: **if a table's header equals the previous table's header, it is the same table** — join the rows and drop the repeated header. Each row then becomes:

```text
Loan interest rates and fees - Loan type: Personal loan; Term: 5 years; Interest rate: 11.0%;
Processing fee: 1.5%; Maximum amount: Rs 15,00,000.
```

**Before vs after, same question:**

| | What the LLM receives for "sick leave for L3?" |
|---|---|
| Naive | `' 5\nL3 24 12 10\nL4 26 12 10 ...'` |
| Table-aware | `Table 1 ... | Grade: L3 | Annual leave (days): 24 | Sick leave (days): 12 | ...` |

**Limits:** borderless tables need a text-based strategy (`vertical_strategy="text"`); scanned PDFs need OCR; merged cells need extra handling. Production tools: Unstructured, Docling, LlamaParse, Azure Document Intelligence, AWS Textract.

**Key points**
- Find tables first, chunk them separately.
- A row chunk must make sense on its own: caption + column name + value.
- Remove the table from the text before sentence chunking.

---

## 6. Quality gate: checking the parser's work

**Problem.** Parsers fail silently. In the loan PDF, "12.5%" was cut in half and nobody noticed. In a bank, a wrong rate is worse than no answer.

**Fix.** Before embedding, check that **every number in the original PDF still exists in the chunks**. If one is missing, reject the file and say which number.

```python
def numbers_in(text):   # '5,00,000' -> '500000', '11.0%' -> '11.0'
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}

missing = numbers_in(original_pdf_text) - numbers_in(all_chunk_text)
if missing:
    print(f"REJECTED {source}: numbers missing from the chunks: {sorted(missing)}")
```

```text
REJECTED loan_rate_guide.pdf: 1 number(s) from the PDF are missing from the chunks: ['12.5']
OK       loan_rate_guide.pdf: all 28 numbers from the PDF are in the chunks
```

**Key points**
- Don't trust the parser; check it. A bad file should not enter the index.
- Numbers are the cheapest thing to verify and the most expensive thing to get wrong.

---

## 7. Metadata and metadata filtering

### 7.1 What metadata is

**Metadata** = structured facts *about* a chunk, stored next to its vector.

| Field | Example | Used for |
|---|---|---|
| `source` | `leave_policy_india_2025.txt` | Citations, debugging |
| `department` | `HR` | Restrict to a team's documents |
| `region` | `India` | Show the right country's rules |
| `year` / `version` | `2025` | Only current rules |
| `doc_type` | `policy`, `troubleshooting` | Narrow the search |
| `access_level` | `managers` | Security: who may see it |
| `chunk_type` | `table_row`, `text` | Table-aware retrieval |

**Where it comes from:** folder path, file properties, the source system (SharePoint, Confluence, a database), a manifest file, or extracted by an LLM. In class, each `.txt` file had a header:

```text
---
title: Leave Policy (India)
department: HR
region: India
year: 2025
doc_type: policy
---
Full-time employees in India receive 24 days of paid annual leave per calendar year. ...
```

### 7.2 Attach metadata to every chunk

```text
BEFORE  "Full-time employees in India receive 24 days of paid annual leave ..."

AFTER   {
          "id": "leave_policy_india_2025#0",
          "text": "Full-time employees in India receive 24 days of paid annual leave ...",
          "metadata": {"department": "HR", "region": "India", "year": 2025,
                       "source": "leave_policy_india_2025.txt"}
        }
```

Every chunk gets its **own copy** of its document's metadata. Store `year` as a **number**, not text, so `>= 2025` comparisons work.

### 7.3 The problem metadata filtering solves

User: **an employee in India, 2025.** Question: **"How many paid leave days do I get?"**

```text
India 2025:  "Full-time employees in India receive 24 days of paid annual leave per calendar year."
India 2024:  "Full-time employees in India receive 18 days of paid annual leave per calendar year."
US 2025:     "Full-time employees in the United States (US) receive 15 days of paid annual leave ..."
```

These sentences differ by one number and one country. Their vectors are almost identical, so similarity ranks them almost equally. Whichever wins, the LLM answers with *that* number: 24, 18 or 15.

> **Similarity measures topic, not validity.** No prompt can fix this, because the wrong chunk is already in the context.

### 7.4 The fix: filter first, then rank

```text
 All 15 chunks ──► FILTER (region ∈ {India, Global} AND year = 2025) ──► 11 chunks ──► RANK by similarity ──► top-k
                       ▲
                       └── values come from the LOGGED-IN USER's profile, not from the LLM
```

**Pre-filter vs post-filter**

| | How | Risk |
|---|---|---|
| **Pre-filter** (use this) | Keep only matching chunks, then rank them | None: you always get k valid results if they exist |
| Post-filter | Take top-k first, then throw away invalid ones | You can end up with 0 results |

**Filter operators** (same style as Chroma/Pinecone):

| Operator | Meaning | Example |
|---|---|---|
| plain value | equals | `{"year": 2025}` |
| `$in` | one of a list | `{"region": {"$in": ["India", "Global"]}}` |
| `$gte` / `$lte` | ≥ / ≤ | `{"year": {"$gte": 2025}}` |
| `$ne` | not equal | `{"doc_type": {"$ne": "draft"}}` |

**The same filter in Qdrant:**

```python
from qdrant_client import models

user_filter = models.Filter(must=[                     # must = AND
    models.FieldCondition(key="region", match=models.MatchAny(any=["India", "Global"])),
    models.FieldCondition(key="year",   match=models.MatchValue(value=2025)),
])
hits = qdrant.query_points("advanced_retrieval", query=question_vector,
                           query_filter=user_filter, limit=4).points
```

### 7.5 Who decides the filter?

| Source | OK? | Example |
|---|---|---|
| User profile / login | Yes | Region = India |
| App / UI (dropdown, current date) | Yes | Year = 2025 |
| LLM extracting it from the question ("self-query") | Only for convenience filters | "the 2024 policy" → year = 2024 |
| LLM for **access** rules | **Never** | A user must not unlock manager documents by asking |

**Key points**
- Attach metadata at ingestion; filter at query time.
- Filter **before** ranking.
- Security and validity rules come from code, not from the LLM.

---

## 8. Document versioning (old vs new policies)

**Problem.** The parental leave policy exists as **2023** (12 weeks), **2025** (18 weeks), **2027** (20 weeks, starts in the future) and an unapproved **draft note** ("26 weeks"). All four are about the same thing and score nearly the same. Nothing says which one is in force.

> "The model did not lie. We left the old rulebook on the desk next to the new one."

**Fix in 4 parts**

1. **Identity.** Every file gets `doc_key`, `version`, `effective_date` (from a manifest or your document system).

   ```json
   "leave_policy_2025.md": {"doc_key": "leave-policy", "version": "2025", "effective_date": "2025-01-01"}
   ```

2. **Status decided by code**, never by hand:

   ```text
   effective_date > today                          → upcoming
   newest effective_date that has already started → current
   any older version                               → superseded
   ```

3. **No identity, no entry.** A file without version information is not indexed:

   ```text
   WARNING: draft_notes.md has no version information, so it is NOT indexed
   ```

4. **Mandatory filter.** Every search applies `status = current`. A "history" question can opt in to `superseded`; `upcoming` is never returned.

**Time changes the answer automatically.** Set today's date to 2027-02-01 and re-ingest: 2027 becomes `current`, 2025 becomes `superseded`. Nothing else is edited.

**Production note:** the status is stored at ingestion time, so re-run ingestion (a nightly job) after an effective date passes.

**Key points**
- Every document needs an identity and a date.
- Let code decide current / superseded / upcoming.
- Make "current only" the default for every search.

---

## 9. Embeddings and embedding management

### 9.1 What an embedding is

An **embedding model** turns text into a fixed-length list of numbers (a **vector**). Texts with similar meaning get vectors that point in similar directions.

```text
"Employees may work remotely two days a week"   → [ 0.12, -0.08,  0.31, ... ]  (384 numbers)
"Can I work from home sometimes?"               → [ 0.10, -0.05,  0.29, ... ]  close  ✔
"ERR-4021: The VPN server is unreachable"        → [-0.22,  0.17, -0.03, ... ]  far    ✘
```
*(numbers illustrative)*

### 9.2 Cosine similarity

How close two vectors are is measured by the angle between them:

```text
                     A · B             sum of (a_i × b_i)
  cosine(A, B) = ─────────────  =  ───────────────────────────
                   |A| × |B|        length of A × length of B
```

- 1 = same direction (same meaning), 0 = unrelated, −1 = opposite.
- If vectors are **normalised** (length 1), cosine is just the **dot product**: `similarity = chunk_vectors @ query_vector`. That's what our code does, and it compares the query with every chunk in one line.

### 9.3 Contextual embedding

**Problem.** The chunk "Electronics may be returned within 15 days" embedded alone doesn't know it belongs to the *refund* policy.

**Fix.** Put the context line in front of the text **before embedding**:

```python
texts_to_embed = [chunk["context"] + "\n" + chunk["text"] for chunk in chunks]
# "Refund and Returns Policy > Exceptions for electronics
#  Electronics may be returned within 15 days of delivery ..."
```

The vector now carries "refund policy" + "electronics" + "15 days" together.

### 9.4 Embedding management rules

| Rule | Why | In our code |
|---|---|---|
| **Same model for documents and questions** | Vectors from different models live in different "spaces" and can't be compared | MiniLM for both, or Gemini for both |
| **Use task types where the model supports them** | Some models embed documents and queries slightly differently for better matching | Gemini: `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for questions |
| **Fixed dimension** | The collection is created for one vector size; a different size is rejected | 384 (MiniLM), 768 (Gemini with `output_dimensionality=768`) |
| **Changing the model = re-embed everything** | Old and new vectors can't be mixed | Delete the collection, re-ingest |
| **Record which model built the index** | So you know when re-indexing is needed | Store model name in config / collection name |
| **Normalise vectors** | Cosine = dot product, faster and consistent | `normalize_embeddings=True` |
| **Embed in batches** | One API call for many chunks is faster and cheaper | `embed_content(contents=[...all chunks...])` |
| **Don't re-embed unchanged text** | Saves cost on re-ingestion | Deterministic IDs + upsert (section 10) |

### 9.5 Local vs API embedding models

| | Local (`all-MiniLM-L6-v2`) | API (Gemini `gemini-embedding-001`) |
|---|---|---|
| Cost | Free | Free tier, then paid |
| Internet | Only for the first download | Every call |
| Size | 384 numbers | Up to 3072, can be shortened (we used 768) |
| Quality | Good for demos and small apps | Stronger, multilingual |
| Used in | `advanced_retrieval/` scripts | Issue/fix pipeline |

**Key points**
- Embedding = meaning as numbers; cosine = how close two meanings are.
- One model per index. Changing models means re-embedding everything.
- Add the context line before embedding.

---

## 10. The vector database (Qdrant)

### 10.1 Vocabulary

```text
 Qdrant
 └── collection "advanced_retrieval"         (like a table)  size=384, distance=COSINE
      ├── point id=5                         (like a row)
      │    ├── vector  [0.012, -0.087, ... 384 numbers]     ← used for SEARCH
      │    └── payload {text, source, region: "India",      ← used for FILTERS
      │                 year: 2025, chunk_type: "text"}       and for showing SOURCES
      └── point id=17
           ├── vector  [...]
           └── payload {text: "Table 1 ... | Grade: L3 | ... Sick leave (days): 12 ...",
                        source: "employee_handbook.pdf", chunk_type: "table_row", page: 1}
```

### 10.2 The main operations

| Operation | What it does | Code |
|---|---|---|
| Create collection | Fix vector size + distance | `create_collection(name, vectors_config=VectorParams(size=384, distance=Distance.COSINE))` |
| Upsert | Insert, or overwrite if the id exists | `upsert(name, points=[PointStruct(id, vector, payload)])` |
| Scroll | Read stored points (no search) | `scroll(name, limit=100, with_payload=True, with_vectors=True)` |
| Query | Vector search, optional filter | `query_points(name, query=vector, query_filter=f, limit=k)` |
| Count | How many points | `count(name).count` |

### 10.3 Duplicates on re-ingestion

**Problem.** Random IDs mean every ingestion run *adds* the same chunks again: 17 points become 34. Search then returns the same chunk twice.

**Fix.** Build the ID from what identifies the chunk (file + position), so the same chunk always gets the same ID and `upsert` overwrites it:

```python
point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['source']}-{chunk['chunk_no']}"))
```

Run ingestion twice: issue version 34 points, fix version 19 points.

**Key points**
- Point = id + vector + payload.
- Search uses the vector; filters and citations use the payload.
- Deterministic IDs + upsert = re-ingestion is safe.

---

## 11. Keyword search: BM25, TF and IDF

### 11.1 The idea

**BM25** ("Best Matching 25") scores a chunk by the **words it shares with the question**, weighted so that rare, specific words count more than common ones. It is the algorithm behind Elasticsearch and OpenSearch, and it is still a strong baseline.

### 11.2 Step 1: tokenize

```python
def tokenize(text):
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())   # keeps "err-4012" as ONE word
    return [w for w in words if w not in STOPWORDS]                  # drop "how", "the", "my"...
```

```text
"How do I fix ERR-4012?"  →  ['fix', 'err-4012']
```

**Stopwords** are very common words that carry no search meaning.

### 11.3 Step 2: TF — term frequency

**TF** = how many times a query word appears in the chunk. More often → more relevant, but with **diminishing returns** (saturation, controlled by `k1`). A chunk that says "VPN" 10 times is not 10× more about VPNs.

```text
TF part = tf × (k1 + 1) / (tf + k1 × length_norm)          k1 = 1.5

 tf = 1  →  1.00
 tf = 2  →  1.43
 tf = 3  →  1.67      ← grows, but slower and slower
 tf = 10 →  2.17
```
*(values for a chunk of average length)*

### 11.4 Step 3: IDF — inverse document frequency

**IDF** = how **rare** a word is across all chunks. A word found in only one chunk is very informative; a word found everywhere tells us little.

```text
IDF(word) = ln( 1 + (N − df + 0.5) / (df + 0.5) )

 N  = number of chunks
 df = number of chunks containing the word
```

**Real numbers from our knowledge base (N = 15 chunks):**

| Word | In how many chunks (df) | IDF | Meaning |
|---|---|---|---|
| `err-4012` | 1 | **2.37** | Very specific → strong signal |
| `certificate` | 1 | 2.37 | Very specific |
| `expired` | 2 | 1.86 | Fairly specific |
| `vpn` | 4 | 1.27 | Somewhat common |
| `employees` | 5 | 1.07 | Common |
| `days` | 7 | 0.76 | Very common → weak signal |

`err-4012` counts more than twice as much as `employees`.

### 11.5 Step 4: length normalisation

Long chunks contain more words, so they match more by chance. BM25 gives them a small penalty (controlled by `b = 0.75`):

```text
length_norm = 1 − b + b × (chunk_length / average_chunk_length)
```

A chunk of average length has `length_norm = 1`; a longer chunk has more than 1, which lowers its TF part.

### 11.6 The full BM25 formula

```text
                       tf × (k1 + 1)
BM25 = Σ   IDF(w) × ─────────────────────────────────────────────
       w ∈ query     tf + k1 × (1 − b + b × chunk_len / avg_len)

defaults: k1 = 1.5 (TF saturation), b = 0.75 (length penalty)
```

### 11.7 Worked example

Question: **"How do I fix ERR-4012?"** → tokens `['fix', 'err-4012']`

Chunk #11: *"ERR-4012: The VPN certificate on your laptop has expired. Open the IT Self-Service app, click Renew Certificate, then restart the VPN client."* → 16 tokens (average 14.87).

```text
'fix'       not in chunk                                     → 0
'err-4012'  IDF = 2.37, tf = 1
            length_norm = 1 − 0.75 + 0.75 × 16/14.87 = 1.057
            TF part     = 1 × 2.5 / (1 + 1.5 × 1.057)  = 0.967
            score       = 2.37 × 0.967                 = 2.29
                                                    BM25 = 2.29
```

Every other chunk scores **0** (none contains `err-4012`), so BM25 returns exactly one chunk — the right one.

### 11.8 Strengths and weaknesses

| Strong at | Weak at |
|---|---|
| Exact codes, IDs, names, product numbers (`ERR-4012`, `SKU-88231`) | Synonyms: "house" ≠ "home" ≠ "remotely" |
| Rare technical terms | Paraphrases: "do my job from my house" finds **nothing** |
| Explainable: you can see why it matched | Typos: `err4012` ≠ `err-4012` |
| Fast, no model needed | Word order and meaning |

---

## 12. Vector (semantic) search

```text
question ──embed──► query vector ──cosine with every chunk vector──► sort ──► top-k
```

```python
query_vector = embed([question])[0]
similarities = chunk_vectors @ query_vector       # cosine for all chunks at once
best = np.argsort(-similarities)[:top_k]
```

| Strong at | Weak at |
|---|---|
| Meaning and paraphrases: "work from my house" → "perform duties remotely" | Exact codes: ERR-4012, ERR-4013, ERR-4021 all look like "a VPN error and its fix" |
| Synonyms, other languages | Numbers and IDs |
| Vague questions | Near-identical texts (table rows L3 vs L4, 2024 vs 2025 policy) |

**Real example from class:**

| Query | BM25 | Vector |
|---|---|---|
| "How do I fix ERR-4012?" | ✔ Exactly one match | ✘ Three error codes score almost the same |
| "Can I do my job from my house sometimes?" | ✘ **No match at all** (job, house, sometimes appear nowhere) | ✔ Finds the remote work policy |

> Keyword search knows **words**, not meaning. Vector search knows **meaning**, not exact words. Each fails exactly where the other is strong.

---

## 13. Hybrid retrieval: combining BM25 and vectors

### 13.1 Architecture

```text
                       ┌─────────────────────┐
                ┌─────►│ BM25 keyword search │──► ranked list A  (by word match)
                │      └─────────────────────┘                     │
  question ─────┤                                                  ├──► FUSION (RRF) ──► top-k
                │      ┌─────────────────────┐                     │
                └─────►│ Vector search       │──► ranked list B  (by meaning)
                       └─────────────────────┘
```

### 13.2 Why we can't just add the scores

| | Score range | Example |
|---|---|---|
| BM25 | 0 to ~10+ (no upper limit) | 2.29 |
| Cosine | −1 to 1 | 0.61 |

Adding 2.29 + 0.61 lets BM25 dominate just because its numbers are bigger. So we combine **ranks**, not scores.

### 13.3 Reciprocal Rank Fusion (RRF) — the formula we used

```text
                    1
RRF(chunk) = Σ  ──────────────          summed over every list the chunk appears in
           lists   k + rank             k = 60 (standard constant), rank starts at 1
```

```python
def rrf(ranked_lists, k=60):
    fused = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            fused[chunk_id] += 1 / (k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])
```

### 13.4 Worked example

Query: **"ERR-4012 popped up, how do I get my VPN working again?"**

| Chunk | BM25 rank | Vector rank | RRF score |
|---|---|---|---|
| ERR-4012 fix | 1 | 2 | 1/61 + 1/62 = 0.0164 + 0.0161 = **0.0325** |
| ERR-4013 fix | 4 | 1 | 1/64 + 1/61 = 0.0156 + 0.0164 = 0.0320 |
| Telecommuting VPN rule | 2 | 3 | 1/62 + 1/63 = 0.0161 + 0.0159 = 0.0320 |
| PTO roll-over (matched "up") | 3 | — | 1/63 = 0.0159 |

*(ranks illustrative; the script prints the real ones)*

A chunk that is **high in both lists** wins. A chunk found by only one list gets only one part. The noise in each list (PTO matched the word "up"; vector put ERR-4013 first) cancels out.

**What k = 60 does:** it softens the gap between neighbouring ranks. With k = 60, rank 1 (1/61) and rank 2 (1/62) are close, so agreement between lists matters more than being #1 in one list.

### 13.5 Alternative: weighted score fusion

```text
hybrid = α × normalised_BM25 + (1 − α) × cosine         normalised = (s − min) / (max − min)
α = 0.5 → equal weight;  α higher → trust keywords more
```

| | RRF | Weighted score fusion |
|---|---|---|
| Needs score normalisation | No | Yes |
| Needs tuning | Hardly (k = 60 works) | Yes (α) |
| Uses score gaps | No (ranks only) | Yes |
| Typical use | Default choice | When you can tune on real questions |

**In real tools:** Elasticsearch/OpenSearch (BM25 + kNN), Qdrant / Weaviate / Pinecone (sparse + dense vectors), LangChain `EnsembleRetriever` (uses RRF).

**Key points**
- BM25 = precision on exact words. Vectors = recall on meaning. Hybrid = both.
- Merge by rank with RRF: `Σ 1/(60 + rank)`.
- Agreement between the two lists is the strongest signal.

---

## 14. Query rewriting (3 types)

### 14.1 Why

Retrieval can only match what the user typed, and users don't type search queries — they type messages. Rewriting fixes the **input** before it reaches the retriever.

```text
 user message ──► LLM rewrite ──► clean search query(s) ──► hybrid retrieval ──► top-k ──► LLM answer
                    (do NOT answer, only rewrite)
```

### 14.2 The three types

| # | Type | What it does | When | Example |
|---|---|---|---|---|
| 1 | **Simple query rewriting** | AI rewrites the question to make it **clearer**: fix spelling, remove filler, use official terms and code formats | Messy, short or informal questions | `vpn not wrking!!! says err4012 pls help` → `VPN not working error ERR-4012` |
| 2 | **Contextual query rewriting** ⭐ | Uses the **chat history** to turn a follow-up into a **standalone** question | **Chatbots** — the most important type, because follow-ups are everywhere in conversation | History: "India leave days?" + `what about the US?` → `How many paid leave days do employees in the US get?` |
| 3 | **Multi-query rewriting** | Generates **several** search queries from one question, searches with each, merges with RRF | Vague questions, vocabulary mismatch | `Can I expense my cab to the airport?` → `taxi fare reimbursement for airport trips` / `how to submit a travel expense claim` / `business trip transport costs policy` |

### 14.3 Type 1 — Simple rewriting

```text
Before:  tokens ['vpn', 'not', 'wrking', 'says', 'err4012', 'pls', 'help', 'asap']
         BM25 match for the code: []    ← "err4012" ≠ "err-4012"
After:   tokens ['vpn', 'not', 'working', 'error', 'err-4012']
         exact keyword match → both retrievers agree → ERR-4012 fix is #1
```

```text
Prompt: Fix spelling, remove filler words, and write error codes in the official
        format ERR-NNNN. Do not answer the question.
        Question: {messy}
        Return JSON: {"rewritten_query": "..."}
```

### 14.4 Type 2 — Contextual rewriting (chatbots)

```text
  user:      How many paid leave days do employees in India get?
  assistant: Employees in India get 24 days of paid annual leave per year.
  user:      what about the US?          ← searched alone: no topic, wrong chunks
                    │
                    ▼  LLM + conversation history
  "How many paid leave days do employees in the US get?"   ← finds the US policy (15 days PTO)
```

```text
Prompt: Use the conversation history to turn the follow-up into a complete,
        standalone question. Keep names, codes and numbers exactly.
        Do not answer the question.
        Conversation: {history}
        Follow-up question: {follow_up}
        Return JSON: {"rewritten_query": "..."}
```

Every real chatbot needs this: "and for managers?", "what about last year?", "how do I apply for it?" all depend on earlier messages.

### 14.5 Type 3 — Multi-query rewriting

```text
                         ┌─► "Can I expense my cab to the airport?"   ─► search ─► list 1 ┐
 "Can I expense my  ──LLM┼─► "taxi fare reimbursement for airport..." ─► search ─► list 2 ├─► RRF ─► top-k
  cab to the airport?"   ├─► "how to submit a travel expense claim"   ─► search ─► list 3 │
                         └─► "business trip transport costs policy"   ─► search ─► list 4 ┘
```

The user says "cab" and "expense"; the policy says "taxi", "reimbursed" and "expense claim". Each variant catches different wording; chunks found by **many** variants rise to the top. Result: the taxi rule **and** how to submit the claim.

> Hybrid search = **one** query, **two** retrievers. Multi-query = **many** queries, one retriever. Both merge with RRF.

### 14.6 Prompt rules for every rewrite

- "**Do not answer the question**" — the LLM only rewrites; answers come from documents.
- Keep names, codes and numbers exactly.
- Ask for JSON and use temperature 0 so the output is predictable and parseable.

### 14.7 Related techniques (for reference)

| Technique | Idea |
|---|---|
| HyDE | Generate a hypothetical answer and search with *its* embedding |
| Step-back | Ask a more general question first ("what is the leave policy?") |
| Decomposition | Split "Compare India and US leave" into two questions |

### 14.8 Trade-offs

- Every rewrite = **one extra LLM call** → more latency and cost.
- The LLM can **drift** the meaning or add details → log the rewritten query; consider searching with both the original and the rewrite.
- Clear questions often need **no** rewrite.

---

## 15. Reranking

**Problem.** Vector search compares two lists of numbers. It is fast but approximate: the true best chunk is often #3 or #4, not #1.

**Fix.** Two stages:

```text
 retrieve MANY candidates (e.g. 10, fast)  ──►  RERANK: read question + each chunk together, score 0-10  ──►  keep best k (e.g. 2)
```

```python
prompt = ("Score how well each passage answers the question, from 0 (useless) to 10 (answers it directly).\n"
          "Return a JSON list with exactly one number per passage, in the same order.\n\n"
          f"Question: {question}\n\nPassages:\n{passages}")
```

| | Vector search (bi-encoder) | Reranker (cross-encoder / LLM) |
|---|---|---|
| Reads | Question and chunk **separately** | Question and chunk **together** |
| Speed | Very fast (millions of chunks) | Slow (only a few candidates) |
| Accuracy | Approximate | High |
| Examples | MiniLM, Gemini embeddings | `cross-encoder/ms-marco-MiniLM-L-6-v2`, Cohere Rerank, an LLM |

**Metadata filter vs reranking:** a filter is a hard yes/no rule **before** ranking; reranking re-scores the top results **after** retrieval.

---

## 16. Context management (context window optimisation)

The **context window** is everything the LLM reads: instructions + retrieved passages + question. What we put in it, how much, and in what order decides the answer.

### 16.1 Too little context: neighbour expansion

**Problem.** The refund rule ("30 days") is in chunk #0 and the electronics exception ("15 days, unopened") is in chunk #1. Search finds only one.

**Fix.** After search, also fetch the chunk **before and after** each hit (same file, `chunk_no ± 1`):

```text
the model will read 3 passages:
  refund_policy.md #0  (neighbour)         ← the standard 30-day rule
  refund_policy.md #1  (found by search)   ← the electronics exception
  refund_policy.md #2  (neighbour)
```

### 16.2 Too much context

More is not better:
- **Cost and latency** grow with every token.
- **Lost in the middle:** LLMs use information at the start and end of a long context better than in the middle.
- **Distraction:** near-miss chunks (the 2024 policy) can be picked up as the answer.

### 16.3 Techniques

| Technique | What | Example |
|---|---|---|
| Choose k carefully | Enough to answer, not more | Retrieve 10, rerank, keep 2–5 |
| Filter first | Remove invalid chunks before they compete | Region + year + "current only" |
| Rerank | Put the best passages first | Section 15 |
| Neighbour expansion | Add the chunks around each hit | ±1 chunk |
| De-duplicate | Remove repeated text (overlaps, duplicate points) | Same `source + chunk_no` once |
| Keep reading order | Sort passages by position in the document | `sorted(passages)` |
| Context line | Tell the LLM where each passage came from | `Refund policy > Electronics` |
| Number the passages | Lets the LLM cite `[1]`, `[2]` | Section 17 |
| Compress | Summarise or trim long passages | Only for very long documents |

**Key point:** the goal is the **smallest context that fully answers the question**, with its sources.

---

## 17. Grounded generation

**Problem.** A bare prompt ("Answer this question: ...") lets the model fill gaps from memory. Ask "Who is the CEO of Zentra Retail?" with no such fact in the documents, and it may invent a name.

**Fix.** Ground the model:

```text
Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I don't know based on the documents."
After the answer, cite the passage numbers you used, like [1].

Context:
[1] Refund and Returns Policy > Standard returns: Most items can be returned within 30 days ...
[2] Refund and Returns Policy > Exceptions for electronics: Electronics may be returned within 15 days ...

Question: How many days do I have to return unopened electronics?
```

```text
ANSWER: Unopened electronics can be returned within 15 days of delivery [2].
```

**Three rules:** only the context · say "I don't know" · cite sources.

---

## 18. RAG + API: combining documents with live data

### 18.1 Why documents alone are not enough

RAG searches **documents you indexed earlier**. Some answers change every hour and can't live in a document:

| Question needs | Comes from | Example |
|---|---|---|
| **Stable knowledge** (rules, manuals, policies) | **RAG** (vector DB) | "What does our policy say about outdoor events in rain?" |
| **Live data** (weather, stock price, order status, exchange rate) | **API** | "What is the rain forecast in Bangalore tomorrow?" |
| **Both** | **RAG + API** | "Can we hold tomorrow's outdoor team event in Bangalore?" |

> Never put live data in the vector DB. Today's weather embedded today is wrong tomorrow. Call the API at question time instead.

### 18.2 Architecture

```text
                                 ┌──────────────────────────┐
                           ┌────►│ RAG: vector DB search    │──► policy passages + sources
                           │     │ (hybrid + filters)       │                 │
 user question ──► ROUTER ─┤     └──────────────────────────┘                 │
   (LLM or rules)          │     ┌──────────────────────────┐                 ▼
   decides what            └────►│ API call (live data)     │──► JSON  ──► CONTEXT BUILDER ──► LLM ──► answer
   is needed                     │ e.g. weather forecast    │      (+ time)   (passages + live      (with sources
                                 └──────────────────────────┘                   facts + question)    and timestamp)
```

**The router** decides which sources the question needs:

| Question | RAG | API |
|---|---|---|
| "What is the refund window for electronics?" | ✔ | |
| "Will it rain in Bangalore tomorrow?" | | ✔ |
| "Can we hold tomorrow's outdoor event in Bangalore?" | ✔ (event policy) | ✔ (forecast) |

Routers can be simple **rules** (keywords like "weather", "today", "price"), an **LLM classifier** that returns JSON (`{"needs_docs": true, "needs_weather": true, "city": "Bangalore"}`), or **LLM tool/function calling**, where the model itself asks for `get_weather(city)` and `search_policies(query)` (this is the step towards agentic RAG).

### 18.3 Real-world example: outdoor event and the weather

**Company document (indexed in the vector DB)** — *Event Safety Policy, HR, 2025* (example document):

```text
Outdoor team events must be moved indoors if the chance of rain is above 60%
or the maximum temperature is above 38°C. The event owner must inform
attendees at least 12 hours before the event.
```

**User:** *"Can we hold tomorrow's outdoor team event in Bangalore?"*

**Step 1 — Router** → needs the policy (RAG) **and** tomorrow's forecast for Bangalore (API).

**Step 2 — RAG** retrieves the Event Safety Policy chunk (filtered: department HR, status current).

**Step 3 — API.** Open-Meteo is a free weather API (no key for non-commercial use):

```text
GET https://api.open-meteo.com/v1/forecast
    ?latitude=12.97&longitude=77.59
    &daily=precipitation_probability_max,temperature_2m_max
    &timezone=Asia/Kolkata&forecast_days=2
```

```json
{
  "daily": {
    "time": ["2026-10-24", "2026-10-25"],
    "precipitation_probability_max": [40, 75],
    "temperature_2m_max": [29.1, 27.4]
  }
}
```
*(values illustrative)*

**Step 4 — Combine** into one grounded prompt:

```text
Answer using ONLY the policy passages and the live data below.
Cite policy passages as [1] and live data as [LIVE]. Say when the live data was fetched.

Policy passages:
[1] Event Safety Policy > Outdoor events: Outdoor team events must be moved indoors if the
    chance of rain is above 60% or the maximum temperature is above 38°C. ...

Live data [LIVE] (Open-Meteo, fetched 2026-10-24 00:50 IST):
Bangalore, 2026-10-25: max rain probability 75%, max temperature 27.4°C

Question: Can we hold tomorrow's outdoor team event in Bangalore?
```

**Step 5 — Answer:**

```text
No. Tomorrow's forecast shows a 75% chance of rain in Bangalore [LIVE], which is above the
60% limit in the Event Safety Policy, so the event must be moved indoors [1]. Inform
attendees at least 12 hours before the event [1]. (Forecast fetched 2026-10-24 00:50 IST.)
```

Neither source alone could answer: the API doesn't know your policy, and the policy doesn't know tomorrow's weather.

### 18.4 Code sketch

```python
import requests

def get_forecast(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon,
              "daily": "precipitation_probability_max,temperature_2m_max",
              "timezone": "Asia/Kolkata", "forecast_days": 2}
    data = requests.get(url, params=params, timeout=10).json()["daily"]
    return {"date": data["time"][1],                                   # tomorrow
            "rain_probability": data["precipitation_probability_max"][1],
            "max_temp": data["temperature_2m_max"][1]}

def answer(question):
    route = route_question(question)            # LLM returns {"needs_docs":..., "needs_weather":..., "city":...}
    passages, live = [], None
    if route["needs_docs"]:
        passages = hybrid_search(question, filter={"status": "current"})   # RAG
    if route["needs_weather"]:
        live = get_forecast(*CITY_COORDS[route["city"]])                   # API
    prompt = build_prompt(question, passages, live)   # grounded: only context, cite [n] and [LIVE]
    return llm(prompt)
```

### 18.5 Other real-world RAG + API combinations

| Domain | RAG (documents) | API (live) | Combined question |
|---|---|---|---|
| E-commerce | Return policy | Order status / delivery date | "Can I still return my order #4521?" |
| Banking | Loan product rules | Customer's credit score, live rates | "Am I eligible for the 5-year personal loan?" |
| Travel | Company travel policy | Flight prices | "Which flight to Delhi is within policy for Friday?" |
| IT helpdesk | Troubleshooting guides | System status page | "VPN shows ERR-4021 — is the server down?" |
| Healthcare admin | Insurance coverage docs | Claim status | "Is my claim covered, and where is it now?" |

### 18.6 Things to get right

| Concern | What to do |
|---|---|
| API fails or is slow | Timeout, retry once, then answer from documents and say live data is unavailable |
| Freshness | Always show **when** the live data was fetched |
| Rate limits / cost | Cache API results for a short time (e.g. weather for 15 minutes) |
| Trust | Treat API text as **data**, not instructions to the LLM |
| Security | API keys in environment variables; never let the LLM choose arbitrary URLs |
| Citations | Separate document sources `[1]` from live sources `[LIVE]` |

**Key points**
- RAG = what you **know** (stable documents). API = what is **happening now**.
- A router decides which source(s) a question needs.
- Combine both in one grounded prompt with sources and a timestamp.

---

## 19. The complete production pipeline

Everything in these notes, in order:

```text
INGESTION ─────────────────────────────────────────────────────────────────────────────────
 Documents
   │ 1. Fetch + identity (doc_key, version, effective_date) ── no identity → not indexed
   │ 2. Status by code: current / superseded / upcoming
   │ 3. Clean: remove headers, footers, page markers
   │ 4. Tables: find → row chunks with headers → remove from text
   │ 5. Text: split on headings → whole sentences (+ overlap) → context line
   │ 6. Quality gate: every PDF number present? else REJECT
   │ 7. Embed (context line + text), same model for docs and queries
   │ 8. Store in Qdrant: deterministic id + vector + payload (metadata)
   ▼
 Vector DB  +  keyword (BM25) index

QUERY TIME ────────────────────────────────────────────────────────────────────────────────
 Question
   │ 1. Router: documents? live API? both?
   │ 2. Rewrite: simple / contextual (chat history) / multi-query
   │ 3. Filter: user profile (region, access) + status = current
   │ 4. Hybrid retrieve: BM25 + vector → RRF  (many candidates)
   │ 5. Rerank → keep best k
   │ 6. Context: neighbours, de-duplicate, order, context lines, numbered passages
   │ 7. API call for live data (if routed)
   │ 8. Grounded prompt: only context, "I don't know", cite [n] and [LIVE]
   ▼
 Answer + sources
```

---

## 20. Cheat sheet, misconceptions and self-check

### 20.1 One line per concept

| Concept | Remember it as |
|---|---|
| Chunking | Cut where the document is already cut; keep sentences whole |
| Overlap | Repeat the last sentence so border facts survive |
| Context line | Every chunk knows its document > section |
| Table-aware chunking | Every row carries its column names |
| Quality gate | Don't trust the parser; check the numbers |
| Metadata | Facts about a chunk, stored as payload |
| Metadata filtering | Similarity measures topic, not validity: filter before ranking |
| Versioning | Identity + date → code decides current; search current only |
| Embedding | Meaning as numbers; one model per index |
| Cosine similarity | Angle between two meanings; dot product when normalised |
| TF | More repeats help, with diminishing returns |
| IDF | Rare words matter more: `ln(1 + (N − df + 0.5)/(df + 0.5))` |
| BM25 | IDF × saturated TF × length penalty, summed over query words |
| Hybrid + RRF | Run both, merge by rank: `Σ 1/(60 + rank)` |
| Query rewriting | Fix the input: simple, contextual (chatbots), multi-query |
| Reranking | Retrieve many fast, re-score few carefully |
| Context management | Smallest context that fully answers, with sources |
| Grounding | Only context, "I don't know", cite |
| RAG + API | Documents for knowledge, APIs for live data, a router to choose |

### 20.2 Common misconceptions

| Misconception | Reality |
|---|---|
| "A bigger embedding model fixes retrieval." | It can't add headers that chunking removed or know which year is current. |
| "Just increase the chunk size." | Every size breaks something; fix *where* you cut. |
| "BM25 is outdated." | Exact matching is why every major vector DB now offers hybrid search. |
| "Add BM25 and cosine scores." | Different scales; combine ranks (RRF) or normalise first. |
| "More context = better answers." | Cost, latency and lost-in-the-middle; send the smallest complete context. |
| "The LLM can decide what the user may see." | Access and validity filters must come from code. |
| "Put live data in the vector DB." | It goes stale; call the API at question time. |
| "Query rewriting always helps." | It adds a call and can change meaning; use it for vague or follow-up questions. |

### 20.3 Self-check questions (answers below)

1. A chunk contains `L3 24 12 10` but not the header row. What is the fix?
2. Why does an India employee sometimes get the US leave policy from a pure vector search?
3. Pre-filter or post-filter — which one, and why?
4. A word appears in 1 of 15 chunks; another in 7 of 15. Which has the higher IDF?
5. Why do we merge BM25 and vector results by rank instead of adding their scores?
6. A chunk is rank 1 in BM25 and rank 3 in vector search. What is its RRF score?
7. Which query rewriting type does a chatbot need most, and why?
8. What is the difference between hybrid search and multi-query?
9. Why do we embed `context line + text` instead of the text alone?
10. A user asks "Should we hold the outdoor event tomorrow?" What sources are needed, and how are they combined?

<details>
<summary>Answers</summary>

1. Table-aware chunking: write each row as `caption | Column: value | ...` so it carries its headers.
2. Embeddings measure topic, not validity; the sentences differ only by a number and a country. Fix: metadata filter on region (and year).
3. Pre-filter: keep only valid chunks, then rank. Post-filter can leave zero results.
4. The word in 1 chunk (IDF 2.37 vs 0.76). Rare words are more informative.
5. Scores are on different scales (BM25 0–10+, cosine −1–1); adding them lets BM25 dominate.
6. 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323.
7. Contextual rewriting: follow-ups like "what about the US?" have no topic without the chat history.
8. Hybrid: one query, two retrievers. Multi-query: many queries, one retriever. Both merge with RRF.
9. So the vector knows which document and section the chunk belongs to ("15 days" → refund policy > electronics).
10. RAG for the event policy + weather API for tomorrow's forecast; a router decides, both go into one grounded prompt with citations and the fetch time.

</details>
