"""[FIX version] quality_gate.py - STAGE 4 of ingestion: check the chunks before anything is embedded or stored.

For a PDF the rule is simple: every number in the original PDF must appear in the final chunks.
If one is missing, the file is rejected (not indexed) and we say which numbers are missing.
Rates, fees and limits are numbers. A missing number is a wrong answer waiting to happen.
"""
import re

from ingest_fix.fetch import read_pdf_plain


def numbers_in(text):
    """All numbers in a text, without thousands separators: '5,00,000' becomes '500000', '11.0%' becomes '11.0'."""
    return {number.replace(",", "") for number in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def file_passes(source, chunks_of_file):
    """Helper for FIX T2. Only PDFs are checked here."""
    if not source.endswith(".pdf"):
        return True
    original = numbers_in(read_pdf_plain(chunks_of_file[0]["path"]))
    stored = numbers_in(" ".join(chunk["text"] for chunk in chunks_of_file))
    missing = sorted(original - stored)
    if missing:
        print(f"  REJECTED {source}: {len(missing)} number(s) from the PDF are missing from the chunks: {missing}")
        return False
    print(f"  OK {source}: all {len(original)} numbers from the PDF are in the chunks")
    return True


def check_chunks(chunks):

    # ===== FIX T2: reject a PDF if any of its numbers is missing from the chunks =====
    accepted = []
    for source in sorted({chunk["source"] for chunk in chunks}):
        chunks_of_file = [chunk for chunk in chunks if chunk["source"] == source]
        if file_passes(source, chunks_of_file):
            accepted += chunks_of_file

    print(f"  {len(accepted)} of {len(chunks)} chunks go on to the next stage")
    return accepted
