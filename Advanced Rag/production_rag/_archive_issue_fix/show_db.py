"""show_db.py - look inside Qdrant at any time.

    python show_db.py issue     # what ingest_issue stored
    python show_db.py fix       # what ingest_fix stored
"""
import importlib
import sys

from config import COLLECTION_FIX, COLLECTION_ISSUE, qdrant

version = sys.argv[1] if len(sys.argv) > 1 else "fix"
if version not in ("issue", "fix"):
    sys.exit("usage: python show_db.py issue|fix")
collection = COLLECTION_ISSUE if version == "issue" else COLLECTION_FIX

if qdrant.collection_exists(collection):
    importlib.import_module(f"ingest_{version}.database").show_stored()
    print(f"\nTotal points in '{collection}': {qdrant.count(collection).count}")
else:
    print(f"Nothing stored yet. Run: python -m ingest_{version}.main --reset")
qdrant.close()
