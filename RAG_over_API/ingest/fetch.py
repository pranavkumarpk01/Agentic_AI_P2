"""
fetch.py  -  STEP 1 of ingestion: get the raw documents.
Reads every .txt / .md / .pdf file inside the data/ folder.
"""
from pathlib import Path


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_documents(folder: str = "data") -> list[dict]:
    """Return a list like [{"source": "hr_policy.md", "text": "..."}]"""
    docs = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        else:
            continue  # ignore other file types
        docs.append({"source": path.name, "text": text})
    return docs
