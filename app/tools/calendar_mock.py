"""Mock calendar tool. Stands in for a real Google Calendar integration."""

from datetime import datetime, timezone

from app.db import connection_scope
from app.schemas import ParsedIntent


def create_calendar_event(intent: ParsedIntent) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    title = intent.title or "Untitled event"
    description = intent.description
    start_time = intent.start_time
    end_time = intent.end_time
    duration_minutes = intent.duration_minutes

    with connection_scope() as conn:
        cursor = conn.execute(
            """
            INSERT INTO calendar_events
                (title, description, start_time, end_time, duration_minutes, created_at, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                description,
                start_time,
                end_time,
                duration_minutes,
                created_at,
                intent.raw_text,
            ),
        )
        record_id = cursor.lastrowid

    return {
        "id": record_id,
        "title": title,
        "description": description,
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": duration_minutes,
        "created_at": created_at,
        "raw_text": intent.raw_text,
    }


def list_calendar_events() -> list[dict]:
    with connection_scope() as conn:
        rows = conn.execute(
            "SELECT * FROM calendar_events ORDER BY start_time IS NULL, start_time ASC"
        ).fetchall()

    return [dict(row) for row in rows]


def update_calendar_event(
    event_id: int,
    title: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    duration_minutes: int | None = None,
    description: str | None = None,
) -> dict | None:
    fields = {
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": duration_minutes,
        "description": description,
    }
    fields = {key: value for key, value in fields.items() if value is not None}

    with connection_scope() as conn:
        if fields:
            set_clause = ", ".join(f"{key} = ?" for key in fields)
            conn.execute(
                f"UPDATE calendar_events SET {set_clause} WHERE id = ?",
                (*fields.values(), event_id),
            )
        row = conn.execute("SELECT * FROM calendar_events WHERE id = ?", (event_id,)).fetchone()

    return dict(row) if row else None


def delete_calendar_event(event_id: int) -> bool:
    with connection_scope() as conn:
        cursor = conn.execute("DELETE FROM calendar_events WHERE id = ?", (event_id,))

    return cursor.rowcount > 0
