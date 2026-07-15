"""Index all existing journal entries into Pinecone for RAG retrieval.

Run once after enabling RAG (setting PINECONE_API_KEY), or any time entries
exist that predate indexing. Idempotent - upsert is keyed by entry id, so
re-running is safe.

    python scripts/backfill_journal_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import journal_index
from app.config import settings
from app.db import init_db
from app.tools.journal import list_journal_entries


def main() -> None:
    if not settings.rag_enabled:
        print("RAG is not enabled (set PINECONE_API_KEY and OPENAI_API_KEY). Nothing to do.")
        return
    init_db()
    entries = list_journal_entries()
    count = journal_index.backfill(entries)
    print(f"Indexed {count} of {len(entries)} journal entries into '{settings.pinecone_index_name}'.")


if __name__ == "__main__":
    main()
