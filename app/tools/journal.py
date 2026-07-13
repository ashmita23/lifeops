"""Journal entry storage tool."""

from datetime import datetime, timezone

from app.db import connection_scope
from app.schemas import ParsedIntent


def create_journal_entry(intent: ParsedIntent) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    title = intent.title
    content = intent.description or intent.raw_text
    # ParsedIntent has no dedicated mood/tags fields yet; leave unset for now
    # so this tool doesn't guess at data the parser hasn't produced.
    mood = None
    tags = None

    with connection_scope() as conn:
        cursor = conn.execute(
            """
            INSERT INTO journal_entries (title, content, mood, tags, created_at, raw_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title, content, mood, tags, created_at, intent.raw_text),
        )
        record_id = cursor.lastrowid

    return {
        "id": record_id,
        "title": title,
        "content": content,
        "mood": mood,
        "tags": tags,
        "created_at": created_at,
        "raw_text": intent.raw_text,
    }


def list_journal_entries() -> list[dict]:
    with connection_scope() as conn:
        rows = conn.execute("SELECT * FROM journal_entries ORDER BY created_at DESC").fetchall()

    return [dict(row) for row in rows]


def delete_journal_entry(entry_id: int) -> bool:
    with connection_scope() as conn:
        cursor = conn.execute("DELETE FROM journal_entries WHERE id = ?", (entry_id,))

    return cursor.rowcount > 0
