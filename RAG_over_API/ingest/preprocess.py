"""
preprocess.py  -  STEP 2 of ingestion: clean the text.
Keep it simple: remove markdown symbols and extra whitespace.
"""
import re


def preprocess(text: str) -> str:
    text = re.sub(r"[#*_>`|]", " ", text)   # remove markdown symbols
    text = re.sub(r"\s+", " ", text)        # collapse spaces / newlines
    return text.strip()
