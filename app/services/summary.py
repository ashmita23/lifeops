"""Today's reminders, calendar events, and journal entries.

Backs the agent's get_daily_summary tool. Split out from the old
app.services.router (deleted with the pre-agent command path) since the
tool-calling agent is the only remaining caller.
"""

from datetime import date

from app.db import connection_scope


def get_daily_summary() -> dict:
    today_str = date.today().isoformat()

    with connection_scope() as conn:
        reminders = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM reminders WHERE due_date LIKE ? OR created_at LIKE ?",
                (f"{today_str}%", f"{today_str}%"),
            ).fetchall()
        ]
        events = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM calendar_events WHERE start_time LIKE ? OR created_at LIKE ?",
                (f"{today_str}%", f"{today_str}%"),
            ).fetchall()
        ]
        journal_entries = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM journal_entries WHERE created_at LIKE ?",
                (f"{today_str}%",),
            ).fetchall()
        ]

    return {
        "date": today_str,
        "reminders": reminders,
        "calendar_events": events,
        "journal_entries": journal_entries,
    }
