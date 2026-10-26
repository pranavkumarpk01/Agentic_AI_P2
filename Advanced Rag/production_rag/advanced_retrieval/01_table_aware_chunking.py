"""
01 - TABLE-AWARE CHUNKING
=========================
Question we answer:  "How many SICK leave days does an L3 employee get?"
The answer (12) sits inside a TABLE in data/employee_handbook.pdf.

  BEFORE  naive fixed-size chunking -> the table is flattened and cut into
          pieces; the row "L3 24 12 10" loses its column headers.
  AFTER   1. find the tables  2. turn every table into self-describing chunks
          3. chunk the remaining text by whole sentences.

Run:  python 01_table_aware_chunking.py            (Enter between steps)
      python 01_table_aware_chunking.py --no-pause
"""

import json
from pathlib import Path

import pdfplumber

from display import console, fixed, note, pause, problem, step, table, title
from rag_core import BM25, sentence_chunks

PDF_PATH = Path(__file__).parent / "data" / "employee_handbook.pdf"
OUTPUT_PATH = Path(__file__).parent / "output" / "chunks_table_aware.json"
QUESTION = "How many sick leave days does an L3 employee get?"
SMALL_TABLE_ROWS = 4    # tables with <= 4 data rows are kept whole

title("01 - Table-aware chunking")
console.print(f"Document : {PDF_PATH.name}\nQuestion : [bold]{QUESTION}[/bold]")

# ---------------------------------------------------------------------------
step(1, "BEFORE - naive approach: extract all text, cut every 200 characters")
# ---------------------------------------------------------------------------
with pdfplumber.open(PDF_PATH) as pdf:
    full_text = "\n".join(page.extract_text() for page in pdf.pages)

console.print("[dim]Raw text that pdfplumber gives us (tables become plain lines):[/dim]")
console.print(full_text)

CHUNK_SIZE = 200
naive_chunks = [full_text[i : i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]

table(
    "Naive chunks (fixed 200 characters)",
    ["#", "chunk text"],
    [[i, repr(c)] for i, c in enumerate(naive_chunks)],
)
problem(
    "Look at the table rows: '[bold]L3 24 12 10[/bold]' is just four numbers.\n"
    "The header 'Annual / Sick / Carry forward' is in a DIFFERENT chunk (or cut in half).\n"
    "Even if we retrieve the right chunk, the LLM cannot tell which number is sick leave."
)
pause()

# ---------------------------------------------------------------------------
step(2, "Find every table on every page")
# ---------------------------------------------------------------------------
found_tables = []   # each item: {"page", "bbox", "caption", "rows"}

with pdfplumber.open(PDF_PATH) as pdf:
    for page_number, page in enumerate(pdf.pages, start=1):
        for t in page.find_tables():          # detects tables from their grid lines
            x0, top, x1, bottom = t.bbox
            # The caption ("Table 1: ...") is the line of text just ABOVE the table.
            caption_area = page.crop((0, max(0, top - 22), page.width, top))
            caption = (caption_area.extract_text() or "").strip()
            found_tables.append(
                {"page": page_number, "bbox": t.bbox, "caption": caption, "rows": t.extract()}
            )

for i, t in enumerate(found_tables, start=1):
    console.print(f"\n[bold]Table {i}[/bold] on page {t['page']}  |  caption: [yellow]{t['caption']}[/yellow]")
    console.print(f"[dim]position (x0, top, x1, bottom) = {tuple(round(v) for v in t['bbox'])}[/dim]")
    table(None, t["rows"][0], t["rows"][1:])    # row 0 = header, the rest = data
pause()

# ---------------------------------------------------------------------------
step(3, "Turn each table into chunks that make sense on their own")
# ---------------------------------------------------------------------------
note(
    "Rule we use:\n"
    f" - SMALL table (<= {SMALL_TABLE_ROWS} rows) -> keep it WHOLE as one markdown chunk\n"
    " - BIG table -> ONE chunk PER ROW, and repeat the caption + headers in every row chunk"
)

table_chunks = []
for t in found_tables:
    header, data_rows = t["rows"][0], t["rows"][1:]

    if len(data_rows) <= SMALL_TABLE_ROWS:
        # Whole table as markdown - an LLM reads markdown tables very well.
        lines = [t["caption"], "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(row) + " |" for row in data_rows]
        table_chunks.append(
            {"text": "\n".join(lines), "metadata": {"type": "table", "page": t["page"], "caption": t["caption"]}}
        )
    else:
        # One self-describing chunk per row: "Grade: L3 | Sick leave (days): 12 | ..."
        for row in data_rows:
            pairs = [f"{column}: {value}" for column, value in zip(header, row)]
            table_chunks.append(
                {
                    "text": f"{t['caption']} | " + " | ".join(pairs),
                    "metadata": {"type": "table_row", "page": t["page"], "caption": t["caption"]},
                }
            )

table(
    "Table chunks",
    ["#", "type", "chunk text"],
    [[i, c["metadata"]["type"], c["text"]] for i, c in enumerate(table_chunks)],
)
fixed("The L3 row now carries its own headers: '... Sick leave (days): 12 ...'")
pause()

# ---------------------------------------------------------------------------
step(4, "Remove the tables from the page, then chunk the remaining TEXT by sentences")
# ---------------------------------------------------------------------------
text_chunks = []
with pdfplumber.open(PDF_PATH) as pdf:
    for page_number, page in enumerate(pdf.pages, start=1):
        text_only = page
        for t in [t for t in found_tables if t["page"] == page_number]:
            x0, top, x1, bottom = t["bbox"]
            # cut out the table AND its caption so they are not indexed twice
            text_only = text_only.outside_bbox((x0, max(0, top - 22), x1, bottom))
        page_text = text_only.extract_text()

        console.print(f"[dim]Page {page_number} text with tables removed:[/dim]")
        console.print(page_text)

        for chunk in sentence_chunks(page_text, max_chars=300, overlap=1):
            text_chunks.append({"text": chunk, "metadata": {"type": "text", "page": page_number}})

table(
    "Text chunks (whole sentences, max ~300 chars, 1 sentence overlap)",
    ["#", "chunk text"],
    [[i, c["text"]] for i, c in enumerate(text_chunks)],
)
note("Notice: no chunk ends in the middle of a sentence, and the last sentence of a\n"
     "chunk is repeated at the start of the next one (overlap).")
pause()

# ---------------------------------------------------------------------------
step(5, "Proof - search both chunk sets with the same question")
# ---------------------------------------------------------------------------
smart_chunks = table_chunks + text_chunks

naive_hit = BM25(naive_chunks).search(QUESTION, top_k=1)[0]
smart_hit = BM25([c["text"] for c in smart_chunks]).search(QUESTION, top_k=1)[0]

table(
    f"Top-1 chunk for: {QUESTION}",
    ["approach", "what the LLM would receive"],
    [
        ["BEFORE (naive)", repr(naive_chunks[naive_hit[0]])],
        ["AFTER (table-aware)", smart_chunks[smart_hit[0]]["text"]],
    ],
)
fixed("With the table-aware chunk the LLM can answer '12 days' with confidence.")

OUTPUT_PATH.parent.mkdir(exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(smart_chunks, indent=2))
console.print(f"\nSaved {len(smart_chunks)} chunks to {OUTPUT_PATH.relative_to(Path(__file__).parent)}")
