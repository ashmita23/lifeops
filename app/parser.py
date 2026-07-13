"""Turns free-form user text into a structured ParsedIntent.

Tries the real LLM first (app.llm_client.call_llm). If no API key is
configured, or the LLM call/response fails for any reason, falls back to a
deterministic local parser so the app is always usable offline.
"""

import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.llm_client import LLMUnavailableError, call_llm
from app.schemas import ParsedIntent

_INTENT_TYPES = {"reminder", "calendar_event", "journal_entry", "daily_summary", "unknown"}
_PRIORITIES = {"low", "medium", "high"}

_SYSTEM_PROMPT = """You are an intent-extraction engine for a personal productivity agent.
Given a user's natural language command, extract a single JSON object with exactly these fields:

- intent_type: one of "reminder", "calendar_event", "journal_entry", "daily_summary", "unknown"
- title: short string or null
- description: string or null
- due_date: best-effort ISO-like date/time string or null
- start_time: best-effort ISO-like date/time string or null
- end_time: best-effort ISO-like date/time string or null
- duration_minutes: integer or null
- priority: one of "low", "medium", "high", or null
- needs_clarification: boolean
- clarification_question: string or null
- raw_text: the original user text, unmodified

Rules:
- If the user says "remind me", intent_type is "reminder".
- If the user says "block", "schedule", "calendar", or "meeting", intent_type is "calendar_event".
- If the user says "journal", "note", "I felt", or uses reflective language, intent_type is "journal_entry".
- If the user is asking what they have today / for a summary, intent_type is "daily_summary".
- If a date/time is missing for a reminder or calendar_event, set needs_clarification to true and
  provide a short clarification_question.
- Parse natural phrases like "tomorrow at 9", "Friday afternoon", "in 2 hours" into best-effort
  ISO-like strings (YYYY-MM-DDTHH:MM:SS).
- Always return ONLY the JSON object, no other text.

User's timezone: {timezone}
User command: {text}
"""


def parse_user_command(text: str, timezone: str = "America/Chicago") -> ParsedIntent:
    try:
        raw_response = call_llm(_SYSTEM_PROMPT.format(timezone=timezone, text=text))
        return _parse_llm_json(raw_response, text)
    except LLMUnavailableError:
        return _local_parse(text, timezone)
    except (json.JSONDecodeError, ValueError, TypeError):
        # LLM returned something unusable; degrade gracefully instead of crashing.
        return _local_parse(text, timezone)


def _parse_llm_json(raw_response: str, raw_text: str) -> ParsedIntent:
    cleaned = raw_response.strip()
    # Strip markdown code fences some models wrap JSON in.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.IGNORECASE)

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM response was not a JSON object")

    intent_type = data.get("intent_type")
    if intent_type not in _INTENT_TYPES:
        intent_type = "unknown"

    priority = data.get("priority")
    if priority not in _PRIORITIES:
        priority = None

    return ParsedIntent(
        intent_type=intent_type,
        title=data.get("title"),
        description=data.get("description"),
        due_date=data.get("due_date"),
        start_time=data.get("start_time"),
        end_time=data.get("end_time"),
        duration_minutes=data.get("duration_minutes"),
        priority=priority,
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_question=data.get("clarification_question"),
        raw_text=data.get("raw_text") or raw_text,
    )


# --- Deterministic local parser (demo mode / LLM fallback) -----------------

_REMINDER_TRIGGERS = ("remind me", "reminder", "remind")
_CALENDAR_TRIGGERS = ("block", "schedule", "calendar", "meeting")
_JOURNAL_TRIGGERS = ("journal", "note", "i felt", "i feel", "felt", "feeling")
_SUMMARY_TRIGGERS = ("daily summary", "today's summary", "summary", "what do i have today", "agenda")

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_TIME_OF_DAY = {"morning": (9, 0), "afternoon": (14, 0), "evening": (18, 0), "night": (20, 0)}


def _local_parse(text: str, timezone: str) -> ParsedIntent:
    lowered = text.lower()

    intent_type = _detect_intent(lowered)
    title = _extract_title(text, lowered, intent_type)
    when = _resolve_datetime_phrase(lowered, timezone)
    duration_minutes = _extract_duration_minutes(lowered)
    priority = _extract_priority(lowered)

    due_date = when if intent_type == "reminder" else None
    start_time = when if intent_type == "calendar_event" else None
    end_time = None
    if start_time and duration_minutes:
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_time = (start_dt + timedelta(minutes=duration_minutes)).isoformat()
        except ValueError:
            end_time = None

    needs_clarification = False
    clarification_question = None
    if intent_type == "reminder" and not due_date:
        needs_clarification = True
        clarification_question = "What date/time should I set for this reminder?"
    elif intent_type == "calendar_event" and not start_time:
        needs_clarification = True
        clarification_question = "What date/time should this event start?"

    return ParsedIntent(
        intent_type=intent_type,
        title=title,
        description=text if intent_type == "journal_entry" else None,
        due_date=due_date,
        start_time=start_time,
        end_time=end_time,
        duration_minutes=duration_minutes,
        priority=priority,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        raw_text=text,
    )


def _detect_intent(lowered: str) -> str:
    if any(trigger in lowered for trigger in _REMINDER_TRIGGERS):
        return "reminder"
    if any(trigger in lowered for trigger in _CALENDAR_TRIGGERS):
        return "calendar_event"
    if any(trigger in lowered for trigger in _SUMMARY_TRIGGERS):
        return "daily_summary"
    if any(trigger in lowered for trigger in _JOURNAL_TRIGGERS):
        return "journal_entry"
    return "unknown"


def _extract_title(original: str, lowered: str, intent_type: str) -> str | None:
    stripped = original.strip()
    for trigger in (*_REMINDER_TRIGGERS, *_CALENDAR_TRIGGERS, *_JOURNAL_TRIGGERS):
        pattern = re.compile(re.escape(trigger), re.IGNORECASE)
        stripped = pattern.sub("", stripped, count=1)
    stripped = re.sub(r"\s+", " ", stripped).strip(" .,-to")
    if not stripped:
        return None
    return stripped[:1].upper() + stripped[1:] if stripped else None


def _extract_priority(lowered: str) -> str | None:
    if "urgent" in lowered or "asap" in lowered or "high priority" in lowered:
        return "high"
    if "low priority" in lowered or "whenever" in lowered:
        return "low"
    return None


def _extract_duration_minutes(lowered: str) -> int | None:
    match = re.search(r"for (\d+)\s*(minute|min|hour|hr)s?", lowered)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    return amount * 60 if unit in ("hour", "hr") else amount


def _resolve_datetime_phrase(lowered: str, timezone: str) -> str | None:
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)

    relative_match = re.search(r"in (\d+)\s*(minute|min|hour|hr|day)s?", lowered)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        if unit in ("minute", "min"):
            delta = timedelta(minutes=amount)
        elif unit in ("hour", "hr"):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)
        return (now + delta).replace(microsecond=0).isoformat()

    base_date = now.date()
    found_date = False
    if "tomorrow" in lowered:
        base_date = (now + timedelta(days=1)).date()
        found_date = True
    elif "today" in lowered:
        found_date = True
    else:
        for i, weekday_name in enumerate(_WEEKDAYS):
            if weekday_name in lowered:
                days_ahead = (i - now.weekday()) % 7
                days_ahead = days_ahead or 7
                base_date = (now + timedelta(days=days_ahead)).date()
                found_date = True
                break

    hour, minute = 9, 0
    found_time = False
    time_match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", lowered)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        meridiem = time_match.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        found_time = True
    else:
        for phrase, (h, m) in _TIME_OF_DAY.items():
            if phrase in lowered:
                hour, minute = h, m
                found_time = True
                break

    if not found_date and not found_time:
        return None

    return datetime(
        base_date.year, base_date.month, base_date.day, hour, minute, tzinfo=tz
    ).replace(microsecond=0).isoformat()
