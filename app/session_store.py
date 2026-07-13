"""SQLite-backed conversation/pending-confirmation storage.

Replaces the in-memory SESSIONS/PENDING_CONFIRMATIONS dicts that used to
live in app.agent - those were lost on every process restart. Same
connection pattern as app/tools/*.py, reusing app.db.connection_scope.
"""

import json
from datetime import datetime, timezone

from app.db import connection_scope


def get_session_messages(session_id: str) -> list[dict] | None:
    with connection_scope() as conn:
        row = conn.execute(
            "SELECT messages FROM conversations WHERE session_id = ?", (session_id,)
        ).fetchone()
    return json.loads(row["messages"]) if row else None


def save_session_messages(session_id: str, messages: list[dict]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection_scope() as conn:
        conn.execute(
            """
            INSERT INTO conversations (session_id, messages, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET messages = excluded.messages, updated_at = excluded.updated_at
            """,
            (session_id, json.dumps(messages), now),
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
