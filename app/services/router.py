"""Routes a parsed intent to the right tool and builds the AgentResponse."""

from datetime import date

from app.db import connection_scope
from app.parser import parse_user_command
from app.schemas import AgentResponse, ParsedIntent
from app.tools.calendar_mock import create_calendar_event
from app.tools.journal import create_journal_entry
from app.tools.reminders import create_reminder


def process_command(
    input_text: str, input_type: str = "text", timezone: str = "America/Chicago"
) -> AgentResponse:
    intent = parse_user_command(input_text, timezone=timezone)

    if intent.needs_clarification:
        return AgentResponse(
            success=False,
            message=intent.clarification_question or "I need more information to proceed.",
            intent=intent,
            stored_record=None,
        )

    handlers = {
        "reminder": _handle_reminder,
        "calendar_event": _handle_calendar_event,
        "journal_entry": _handle_journal_entry,
        "daily_summary": _handle_daily_summary,
    }
    handler = handlers.get(intent.intent_type, _handle_unknown)
    return handler(intent)


def _handle_reminder(intent: ParsedIntent) -> AgentResponse:
    record = create_reminder(intent)
    return AgentResponse(
        success=True,
        message=f"Reminder set: \"{record['title']}\"" + (f" for {record['due_date']}" if record["due_date"] else ""),
        intent=intent,
        stored_record=record,
    )


def _handle_calendar_event(intent: ParsedIntent) -> AgentResponse:
    record = create_calendar_event(intent)
    return AgentResponse(
        success=True,
        message=f"Event scheduled: \"{record['title']}\"" + (f" at {record['start_time']}" if record["start_time"] else ""),
        intent=intent,
        stored_record=record,
    )


def _handle_journal_entry(intent: ParsedIntent) -> AgentResponse:
    record = create_journal_entry(intent)
    return AgentResponse(
        success=True,
        message="Journal entry saved.",
        intent=intent,
        stored_record=record,
    )


def _handle_daily_summary(intent: ParsedIntent) -> AgentResponse:
    summary = get_daily_summary()
    return AgentResponse(
        success=True,
        message="Here's your summary for today.",
        intent=intent,
        stored_record=summary,
    )


def _handle_unknown(intent: ParsedIntent) -> AgentResponse:
    return AgentResponse(
        success=False,
        message=(
            "I couldn't figure out what you want me to do. Try phrases like "
            "\"remind me to...\", \"schedule a meeting...\", or \"journal: ...\"."
        ),
        intent=intent,
        stored_record=None,
    )


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
