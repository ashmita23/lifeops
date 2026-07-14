"""Planner specialist: decides WHEN to put something on the calendar.

The supervisor (the main agent) delegates scheduling decisions to this
specialist via the plan_schedule tool. The intelligence - finding a free slot
that fits, keeps a buffer from other events, and respects a time-of-day
preference - lives in the pure, unit-tested `find_free_slots`; the tool
handler just feeds it the day's events and returns proposed times for the
supervisor to confirm and create. Deterministic on purpose: a live demo of
"find me an hour tomorrow morning" should never depend on an LLM guessing at
arithmetic.
"""

from datetime import datetime, time, timedelta

from app.tools.calendar_mock import list_calendar_events

_PREFERENCE_WINDOWS = {
    "morning": (time(0, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(17, 0)),
    "evening": (time(17, 0), time(23, 59)),
    "any": (time(0, 0), time(23, 59)),
}

# Default hours the planner is willing to schedule within when the caller
# doesn't pin an explicit window.
_DAY_START = time(8, 0)
_DAY_END = time(20, 0)


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def find_free_slots(
    busy: list[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
    duration_minutes: int,
    *,
    buffer_minutes: int = 15,
    preference: str = "any",
    max_results: int = 3,
    step_minutes: int = 15,
) -> list[datetime]:
    """Return up to max_results start times where a `duration_minutes` block
    fits inside [window_start, window_end], keeps `buffer_minutes` clear of
    every busy block, and starts within the `preference` time-of-day window.

    busy blocks may be unsorted/overlapping; they're padded by the buffer so a
    candidate simply must not overlap a padded block."""
    duration = timedelta(minutes=duration_minutes)
    buffer = timedelta(minutes=buffer_minutes)
    step = timedelta(minutes=step_minutes)
    padded = [(s - buffer, e + buffer) for s, e in busy]

    pref_start, pref_end = _PREFERENCE_WINDOWS.get(preference or "any", _PREFERENCE_WINDOWS["any"])

    results: list[datetime] = []
    cursor = window_start
    while cursor + duration <= window_end and len(results) < max_results:
        slot_end = cursor + duration
        in_pref = pref_start <= cursor.time() < pref_end
        clear = not any(_overlaps(cursor, slot_end, bs, be) for bs, be in padded)
        if in_pref and clear:
            results.append(cursor)
            cursor = slot_end  # jump past the chosen slot, don't emit adjacent dupes
        else:
            cursor += step
    return results


def _event_bounds(event: dict) -> tuple[datetime, datetime] | None:
    """Parse a calendar row into (start, end) datetimes, or None if it has no
    usable start time. End defaults to start + duration, else +60 min."""
    start_raw = event.get("start_time")
    if not start_raw:
        return None
    try:
        start = datetime.fromisoformat(start_raw)
    except ValueError:
        return None
    end_raw = event.get("end_time")
    if end_raw:
        try:
            return start, datetime.fromisoformat(end_raw)
        except ValueError:
            pass
    minutes = event.get("duration_minutes") or 60
    return start, start + timedelta(minutes=minutes)


def plan_schedule(args: dict, raw_text: str) -> dict:
    """Tool handler. args: title, date (ISO date), duration_minutes,
    preference (morning|afternoon|evening|any). Returns proposed start times;
    the supervisor confirms with the user and creates the event.

    Reads the app's local calendar. When the real Google Calendar (MCP) is
    connected, events live there instead, so the planner won't see them yet -
    parsing live MCP events into busy blocks is a documented stretch
    (docs/tech-debt.md). `considered_events` in the result reports how many it
    actually accounted for, so the caller can be transparent about it."""
    title = args.get("title") or "Untitled"
    date_str = args.get("date")
    duration_minutes = int(args.get("duration_minutes") or 60)
    preference = (args.get("preference") or "any").lower()

    try:
        day = datetime.fromisoformat(date_str).date() if date_str else datetime.now().date()
    except ValueError:
        return {"error": f"Could not parse date '{date_str}'. Use YYYY-MM-DD."}

    window_start = datetime.combine(day, _DAY_START)
    window_end = datetime.combine(day, _DAY_END)

    busy = []
    for event in list_calendar_events():
        bounds = _event_bounds(event)
        if bounds and bounds[0].date() == day:
            busy.append(bounds)

    slots = find_free_slots(
        busy, window_start, window_end, duration_minutes, preference=preference
    )

    return {
        "title": title,
        "date": day.isoformat(),
        "duration_minutes": duration_minutes,
        "preference": preference,
        "proposed_slots": [s.isoformat() for s in slots],
        "considered_events": len(busy),
    }
