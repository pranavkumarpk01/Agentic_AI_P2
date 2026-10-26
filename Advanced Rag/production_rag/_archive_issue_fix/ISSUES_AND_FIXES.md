# The three failures: what goes wrong, why, and how it is fixed

This is the explanation to read before class. `README.md` is the run order; this file is the "why".
Each failure follows the same pattern: the story, the problem you can see, why it happens, the fix, the code to open, and what to say.

---

## Failure 1. The table never made it in

**The story.** A loan rate guide is a PDF with tables. The most valuable questions (rates, fees, limits) live inside them, and this table runs across two pages.

**Three things go wrong**

1. The PDF text is read as one flat stream, so the column names sit once at the top, far from most numbers.
2. The table continues on page 2 with a repeated header row, which lands in the middle of a chunk.
3. A fixed 300-character cut can slice a number in half.

**Show the problem.** Run `python -m ingest_issue.main --reset`, stage 3, find `loan_rate_guide.pdf`:

```text
[chunk 3] loan_rate_guide.pdf #0  (300 characters)
  | Table 1: Loan interest rates and fees
  | Loan type Term Interest rate Processing fee Maximum amount
  | Personal loan 1 year 12
[chunk 4] loan_rate_guide.pdf #1  (300 characters)
  | .5% 2.0% Rs 5,00,000
  | Personal loan 3 years 11.5% 1.75% Rs 10,00,000
  | ...
  | Table 1 (continued)
  | Loan type Term Interest rate Processing fee Maximum amount
  | Home loan 30 years
[chunk 5] loan_rate_guide.pdf #2
  | 9.0% 0.5% Rs 2,00,00,000
```

- "12.5%" is cut into "12" and ".5%".
- The repeated header sits in the middle of chunk 4.
- The row "Home loan 30 years" is cut off from its numbers.

Ask: `python -m query_issue.main "What is the interest rate for a 5-year personal loan?" --k 2 --no-answer`
The model sees "11.0% 1.5%" and cannot tell the interest rate from the fee.

**Why it matters.** In a bank, a wrong rate is worse than no answer. The system returned a confident-looking number with nothing to check it against.

**The fix has two parts**

*Part 1, read the table as a table (`ingest_fix/fetch.py`, `read_pdf_with_tables()`):*

1. `page.find_tables()` finds the grids on each page.
2. `outside_bbox` keeps the text that is not inside a table, separately.
3. If a table's header equals the previous table's header, it is the same table continuing: join the rows and drop the repeated header.
4. Each row becomes `Table title - Column: value; Column: value.`

```text
Loan interest rates and fees - Loan type: Personal loan; Term: 5 years; Interest rate: 11.0%; Processing fee: 1.5%; Maximum amount: Rs 15,00,000.
```

Every row now carries its own column names, so a row can be understood on its own.

*Part 2, the quality gate (`ingest_fix/quality_gate.py`):* compare the numbers found in the PDF with the numbers found in the chunks. If any PDF number is missing, reject the file and say which one.

```bash
python gate_demo.py
```
```text
REJECTED loan_rate_guide.pdf: 1 number(s) from the PDF are missing from the chunks: ['12.5']
```

On the fix ingestion the same check passes: `OK loan_rate_guide.pdf: all 28 numbers from the PDF are in the chunks`.

**Say:** "We do not trust the parser. We check its work. If a number from the PDF is not in the chunks, the file does not go in."

**Limit:** pdfplumber finds tables with ruled lines. Scanned or borderless tables need another tool.

---

## Failure 2. The answer got cut in half

**The story.** The returns policy says most items can be returned within 30 days, with an exception for electronics. The rule and its exception must travel together.

**What goes wrong.** Cutting every 300 characters ignores the document's structure. Chunks start and end mid-sentence, and a fact can be split across two chunks.

**Show the problem.** Issue ingestion, stage 3, `refund_policy.md`:

```text
[chunk 16] refund_policy.md #1  (300 characters)
  | ing days. Store credit is offered ... Ask in shop.
  | ...
  | ## Exceptions for electronics
  | Electronics have a shorter window. Electronics may be returned within 1
[chunk 17] refund_policy.md #2
  | 5 days of delivery, and only if the item is unopened and sealed. ...
```

The number 15 is split into "1" and "5". Ask:

```bash
python -m query_issue.main "How many days do I have to return unopened electronics?" --k 1 --no-answer
```

Stage 10 ("the model will read 1 passages") shows one broken chunk.

**Why a bigger chunk size does not fix it.** Every size breaks something. Too small splits facts, too big dilutes the search. The cut position is the problem, not the size.

**The fix has four parts**

*In `ingest_fix/chunking.py`, `split_into_chunks()`:*
1. Split on headings and remember the heading path.
2. Merge a tiny section (under 60 characters) into the one before it, such as the one-line "Contact" section.
3. Pack whole sentences up to 500 characters, never cutting inside a sentence.
4. Store a context line with each chunk, for example `Refund and Returns Policy > Exceptions for electronics`. It is also put in front of the text before embedding (`embedding.py`), so the vector knows where the chunk came from.

*In `query_fix/retrieval.py`, `expand_with_neighbours()`:*
5. After search, also fetch the chunk before and after each hit, so a fact on a boundary is not lost.

Result:

```bash
python -m query_fix.main "How many days do I have to return unopened electronics?" --k 1 --no-answer
```
```text
the model will read 3 passages:
  refund_policy.md #0  (neighbour)         <- the standard 30-day rule
  refund_policy.md #1  (found by search)   <- the electronics exception
  refund_policy.md #2  (neighbour)
```

See the context line in Qdrant with `python show_db.py fix`.

**Say:** "Cut where the document is already cut, and bring the neighbours along."

---

## Failure 3. Old and new versions share one index

**The story.** The parental leave policy changed from 12 weeks (2023) to 18 weeks (2025). A new version, 20 weeks, starts in 2027. There is also a draft note nobody approved ("26 weeks").

**What goes wrong.** All four documents are about the same thing, so they all score nearly the same for the question. Nothing in the index says which one is in force.

**Show the problem.**

```bash
python -m query_issue.main "How many weeks of paid parental leave do employees get?" --k 3 --no-answer
```

The candidates include the 2023, 2025 and 2027 chunks and the draft note. Their scores are almost identical. Each hit is labelled `(no version info stored)`.

**Say:** "The model did not lie. We left the old rulebook on the desk next to the new one."

**The fix has four parts**

1. *Identity.* `data/manifest.json` gives every file a `doc_key`, `version` and `effective_date`. In real life this comes from your document system.
2. *Status decided by code* (`ingest_fix/fetch.py`, `add_status()`): the newest version whose date has passed is `current`, older ones are `superseded`, future ones are `upcoming`.
3. *No identity, no entry.* A file with no manifest entry is not indexed, and a warning is printed.

```text
WARNING: draft_notes.md has no version information, so it is NOT indexed
leave_policy_2023.md  ... 'status': 'superseded'
leave_policy_2025.md  ... 'status': 'current'
leave_policy_2027.md  ... 'status': 'upcoming'
```

4. *Mandatory filter* (`query_fix/retrieval.py`): "current only" is applied in every search, so no caller can forget it.

```bash
python -m query_fix.main "How many weeks of paid parental leave do employees get?" --k 3 --no-answer
```

Only `leave-policy v2025 (current)` comes back.

**A history question opts in on purpose.** Upcoming versions are never returned:

```bash
python -m query_fix.main "How many weeks of parental leave did employees get in the past?" --k 3 --history --no-answer
```

**Watch the status change with time:**

```bash
AS_OF_DATE=2027-02-01 python -m ingest_fix.main --reset
```

The 2027 policy is now `current` and 2025 is `superseded`. Nothing was edited except the date. Re-run `python -m ingest_fix.main --reset` afterwards to go back.

**Production note.** The status is decided at ingestion time and stored in Qdrant. When an effective date passes, run ingestion again (a nightly job is enough) or the labels go stale.

---

## Smaller issues (each a pair of files to compare)

| Label | File | Problem | Fix |
| --- | --- | --- | --- |
| X1 | `preprocessing.py` | Page headers and footers ("ZENTRA RETAIL \| INTERNAL", "--- Page") stay inside chunks and pollute search | `clean()` removes them |
| X2 | `embedding.py` | The vector sees only the chunk text, so "30 days" has no idea which policy it belongs to | The context line is put in front of the text before embedding. Stage 5 shows what is sent to Gemini |
| X3 | `database.py` | Random IDs, so running ingestion twice doubles the points (34) | IDs built from file name and chunk number, so a second run overwrites (19). Run each `main` twice without `--reset` |
| Q1 | `rewrite.py` | The question is searched exactly as typed, with the user's words, not the policy's | Gemini rewrites it in the words a policy would use. Try `"can I send back a gadget I haven't opened yet?"` |
| Q2 | `rerank.py` | The vector order is trusted, but it is only approximate | Gemini scores each candidate for the question and the best are kept. Try `"What is the maximum I can spend on a hotel per night?"` |
| Q3 | `generation.py` | A bare prompt lets the model guess when context is missing | Answer only from the context, say "I don't know based on the documents.", cite passage numbers. Try `"Who is the CEO of Zentra Retail?" --show-prompt` |

Q1 to Q3 call Gemini, so whether the difference is visible depends on the real model. Try each once before class.
