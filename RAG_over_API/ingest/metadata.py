"""
metadata.py  -  STEP 2b of ingestion: attach METADATA to every chunk.

Metadata = extra facts about a chunk (which section? which topic?).
Qdrant stores it next to the vector (in the "payload") so that at query time
we can say "only search chunks where category = leave".

Our HR policy is written as sections:   ## 2. Casual and Sick Leave
So we (1) split the document by section, and (2) tag each section.
"""
import re

# Simple keyword -> category rules. Edit these for your own policy.
CATEGORY_RULES = {
    "leave": "leave",
    "holiday": "leave",
    "working hours": "work",
    "provident": "benefits",
    "reimbursement": "benefits",
    "notice": "exit",
    "resignation": "exit",
    "full and final": "exit",
    "work from home": "work",
    "attendance": "work",
    "insurance": "benefits",
    "bonus": "benefits",
    "learning": "benefits",
    "payroll": "benefits",
    # anything else (e.g. Code of Conduct, Performance Review) becomes "general"
}


def get_category(title: str) -> str:
    title = title.lower()
    for keyword, category in CATEGORY_RULES.items():
        if keyword in title:
            return category
    return "general"


def split_sections(raw_text: str) -> list[dict]:
    """
    Split on lines that start with '## '.
    Returns [{"section_no": 2, "section": "Casual and Sick Leave", "category": "leave", "body": "..."}]
    """
    sections = []
    # parts[0] is the text before the first '## ' (the document title) - we skip it
    parts = re.split(r"^## ", raw_text, flags=re.MULTILINE)[1:]
    for part in parts:
        heading, _, body = part.partition("\n")
        match = re.match(r"(\d+)\.\s*(.*)", heading.strip())
        section_no = int(match.group(1)) if match else 0
        title = match.group(2) if match else heading.strip()
        sections.append({
            "section_no": section_no,
            "section": title,
            "category": get_category(title),
            "body": body,
        })
    return sections
