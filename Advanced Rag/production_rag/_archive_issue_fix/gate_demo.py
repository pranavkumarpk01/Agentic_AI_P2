"""gate_demo.py - show the quality gate rejecting what the ISSUE pipeline would have stored.

It builds the chunks the ISSUE way (cut every 300 characters), then runs the FIX quality gate on them.
No Gemini and no database are used.

    python gate_demo.py
"""
import contextlib
import io

from config import qdrant
from ingest_fix.quality_gate import file_passes
from ingest_issue.chunking import chunk_documents
from ingest_issue.fetch import fetch_documents

with contextlib.redirect_stdout(io.StringIO()):  # keep the output short
    chunks = chunk_documents(fetch_documents())

pdf_chunks = [chunk for chunk in chunks if chunk["source"] == "loan_rate_guide.pdf"]
print(f"The ISSUE pipeline cut the loan PDF into {len(pdf_chunks)} chunks. Now the FIX gate checks them:\n")
file_passes("loan_rate_guide.pdf", pdf_chunks)

qdrant.close()  # close the connection politely
