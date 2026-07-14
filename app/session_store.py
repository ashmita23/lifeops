"""SQLite-backed conversation/pending-confirmation storage.

Replaces the in-memory SESSIONS/PENDING_CONFIRMATIONS dicts that used to
live in app.agent - those were lost on every process restart. Same
connection pattern as app/tools/*.py, reusing app.db.connection_scope.

Conversation history is an APPEND-ONLY log (one row per message in the
conversation_messages table), not a single blob rewritten each turn. That
distinction is the whole point: a blob rewrite is a read-modify-write, so
two concurrent turns on one session race and one silently loses its
messages. An append is a plain INSERT the database serializes for us, so
concurrent turns both land - nothing is lost. It also drops the per-turn
write cost from O(whole history) to O(new messages).
"""

import json
from datetime import datetime, timezone

from app.db import connection_scope


def get_session_messages(session_id: str) -> list[dict] | None:
    """Full ordered message history for a session, or None if it has none.
    Ordered by the autoincrement id, i.e. insertion order."""
    with connection_scope() as conn:
        rows = conn.execute(
            "SELECT message FROM conversation_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    if not rows:
        return None
    return [json.loads(row["message"]) for row in rows]


def append_messages(session_id: str, messages: list[dict]) -> None:
    """Appends new messages as individual rows in one transaction. Only the
    messages produced this turn should be passed - never the full history -
    since existing rows are already durably stored and must not be rewritten."""
    if not messages:
        return
    now = datetime.now(timezone.utc).isoformat()
    with connection_scope() as conn:
        conn.executemany(
            "INSERT INTO conversation_messages (session_id, message, created_at) VALUES (?, ?, ?)",
            [(session_id, json.dumps(message), now) for message in messages],
        )


def get_pending_confirmation(session_id: str) -> dict | None:
    with connection_scope() as conn:
        row = conn.execute(
            "SELECT tool_name, arguments FROM pending_confirmations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return {"tool": row["tool_name"], "arguments": json.loads(row["arguments"])}


def set_pending_confirmation(session_id: str, tool_name: str, arguments: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection_scope() as conn:
        conn.execute(
            """
            INSERT INTO pending_confirmations (session_id, tool_name, arguments, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                tool_name = excluded.tool_name, arguments = excluded.arguments, created_at = excluded.created_at
            """,
            (session_id, tool_name, json.dumps(arguments), now),
        )


def pop_pending_confirmation(session_id: str) -> dict | None:
    """Reads and deletes in one call - matches the old dict.pop() semantics
    used for the "snapshot then clear stale state" logic in app.agent."""
    pending = get_pending_confirmation(session_id)
    if pending is not None:
        with connection_scope() as conn:
            conn.execute("DELETE FROM pending_confirmations WHERE session_id = ?", (session_id,))
    return pending
