"""Reminder storage tool."""

from datetime import datetime, timezone

from app.db import connection_scope
from app.schemas import ParsedIntent


def create_reminder(intent: ParsedIntent) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    title = intent.title or "Untitled reminder"
    description = intent.description
    due_date = intent.due_date
    priority = intent.priority or "medium"

    with connection_scope() as conn:
        cursor = conn.execute(
            """
            INSERT INTO reminders (title, description, due_date, priority, created_at, raw_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title, description, due_date, priority, created_at, intent.raw_text),
        )
        record_id = cursor.lastrowid

    return {
        "id": record_id,
        "title": title,
        "description": description,
        "due_date": due_date,
        "priority": priority,
        "completed": False,
        "created_at": created_at,
        "raw_text": intent.raw_text,
    }


def list_reminders(include_completed: bool = False) -> list[dict]:
    query = "SELECT * FROM reminders"
    if not include_completed:
        query += " WHERE completed = 0"
    query += " ORDER BY due_date IS NULL, due_date ASC"

    with connection_scope() as conn:
        rows = conn.execute(query).fetchall()

    return [_row_to_dict(row) for row in rows]


def update_reminder(
    reminder_id: int,
    title: str | None = None,
    due_date: str | None = None,
    priority: str | None = None,
    description: str | None = None,
) -> dict | None:
    fields = {
        "title": title,
        "due_date": due_date,
        "priority": priority,
        "description": description,
    }
    fields = {key: value for key, value in fields.items() if value is not None}

    with connection_scope() as conn:
        if fields:
            set_clause = ", ".join(f"{key} = ?" for key in fields)
            conn.execute(
                f"UPDATE reminders SET {set_clause} WHERE id = ?",
                (*fields.values(), reminder_id),
            )
        row = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()

    return _row_to_dict(row) if row else None


def complete_reminder(reminder_id: int) -> dict | None:
    with connection_scope() as conn:
        conn.execute("UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,))
        row = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()

    return _row_to_dict(row) if row else None


def delete_reminder(reminder_id: int) -> bool:
    with connection_scope() as conn:
        cursor = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))

    return cursor.rowcount > 0


def _row_to_dict(row) -> dict:
    data = dict(row)
    data["completed"] = bool(data["completed"])
    return data
