"""Per-user Google Calendar access via the REST API.

Phases 1-3 gave every request a signed-in user and stored that user's calendar
refresh token (encrypted). This module is the multi-user replacement for the
single global MCP calendar (app/mcp_client.py): it calls the Google Calendar
REST API directly with the *current* user's access token, so each person acts
on their OWN calendar.

- CALENDAR_TOOLS: OpenAI function schemas offered to the model. Names and shapes
  mirror the @cocal/google-calendar-mcp tools the agent already knows
  (create-event, list-events, ...), so the planner/prompt behave identically -
  only the execution backend changes.
- call_calendar_tool(): dispatch one tool call for current_user_id(), returning
  the same {"result": [...], "is_error": bool} shape as mcp_client.call_mcp_tool
  so the agent consumes it unchanged.

Tool results are JSON strings (as the MCP server returned), so the model reads
them the same way. Destructive names keep the word "delete" so the agent's
existing approval gate (_requires_approval) still catches delete-event.
"""

import json
import logging
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from app.config import settings
from app.db import current_user_id
from app.tokens import CalendarAuthError, get_access_token

logger = logging.getLogger(__name__)

_API = "https://www.googleapis.com/calendar/v3"

# The subset of calendar operations the agent actually uses. Schemas match the
# MCP server's so the model produces identical arguments.
CALENDAR_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get-current-time",
            "description": "Get the current date and time. Call this FIRST before creating, "
            "updating, or searching for events so relative dates resolve correctly.",
            "parameters": {
                "type": "object",
                "properties": {"timeZone": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list-calendars",
            "description": "List all of the user's calendars.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list-events",
            "description": "List events from a calendar within a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendarId": {"type": "string", "description": "Use 'primary' for the main calendar."},
                    "timeMin": {"type": "string", "description": "RFC3339 lower bound, e.g. 2026-07-17T00:00:00-05:00."},
                    "timeMax": {"type": "string", "description": "RFC3339 upper bound."},
                    "timeZone": {"type": "string"},
                },
                "required": ["calendarId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search-events",
            "description": "Search events in a calendar by free-text query within a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendarId": {"type": "string"},
                    "query": {"type": "string"},
                    "timeMin": {"type": "string"},
                    "timeMax": {"type": "string"},
                    "timeZone": {"type": "string"},
                },
                "required": ["calendarId", "query", "timeMin", "timeMax"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create-event",
            "description": "Create a new calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendarId": {"type": "string", "description": "Use 'primary' unless the user names another calendar."},
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "start": {"type": "string", "description": "RFC3339 start, e.g. 2026-07-17T12:00:00-05:00."},
                    "end": {"type": "string", "description": "RFC3339 end."},
                    "timeZone": {"type": "string", "description": "IANA zone, e.g. America/Chicago."},
                    "location": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["calendarId", "summary", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update-event",
            "description": "Update fields of an existing calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendarId": {"type": "string"},
                    "eventId": {"type": "string"},
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "timeZone": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["calendarId", "eventId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete-event",
            "description": "Delete a calendar event. Requires the user's confirmation first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendarId": {"type": "string"},
                    "eventId": {"type": "string"},
                },
                "required": ["calendarId", "eventId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get-freebusy",
            "description": "Query free/busy windows across calendars to find open slots. "
            "Time range max 3 months.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calendars": {"type": "array", "items": {"type": "object"},
                                  "description": "List of {\"id\": \"primary\"} objects."},
                    "timeMin": {"type": "string"},
                    "timeMax": {"type": "string"},
                    "timeZone": {"type": "string"},
                },
                "required": ["calendars", "timeMin", "timeMax"],
            },
        },
    },
]

CALENDAR_TOOL_NAMES = {t["function"]["name"] for t in CALENDAR_TOOLS}


def is_calendar_tool(name: str) -> bool:
    return name in CALENDAR_TOOL_NAMES


def _default_tz() -> str:
    return settings.default_timezone


def _start_end_obj(value: str, time_zone: str) -> dict:
    """Google wants {dateTime, timeZone} (or {date} for all-day). We always get
    an RFC3339 dateTime string from the model."""
    return {"dateTime": value, "timeZone": time_zone}


def _pathsafe(value: str) -> str:
    """URL-encode a calendarId/eventId for use in the REST path. Calendar IDs
    like the US-holidays calendar contain '#'/'@' which otherwise break the URL
    (404), so secondary/shared calendars would silently fail without this."""
    return quote(str(value), safe="")


def _dispatch(name: str, args: dict, token: str) -> dict:
    """Execute one calendar op against Google's REST API. Returns the JSON the
    model should read. Raises httpx.HTTPStatusError on API failure."""
    headers = {"Authorization": f"Bearer {token}"}
    tz = args.get("timeZone") or _default_tz()

    with httpx.Client(timeout=30, headers=headers) as client:
        if name == "get-current-time":
            now = datetime.now(ZoneInfo(tz))
            return {"currentTime": now.isoformat(), "timeZone": tz}

        if name == "list-calendars":
            r = client.get(f"{_API}/users/me/calendarList")
            r.raise_for_status()
            items = r.json().get("items", [])
            return {"calendars": [{"id": c["id"], "summary": c.get("summary")} for c in items]}

        if name in ("list-events", "search-events"):
            cal = _pathsafe(args["calendarId"])
            params = {"singleEvents": "true", "orderBy": "startTime", "timeZone": tz}
            for k in ("timeMin", "timeMax"):
                if args.get(k):
                    params[k] = args[k]
            if name == "search-events":
                params["q"] = args["query"]
            r = client.get(f"{_API}/calendars/{cal}/events", params=params)
            r.raise_for_status()
            events = [_slim_event(e) for e in r.json().get("items", [])]
            return {"events": events}

        if name == "create-event":
            body = _event_body(args, tz)
            r = client.post(f"{_API}/calendars/{_pathsafe(args['calendarId'])}/events", json=body)
            r.raise_for_status()
            return {"event": _slim_event(r.json())}

        if name == "update-event":
            body = _event_body(args, tz, partial=True)
            r = client.patch(
                f"{_API}/calendars/{_pathsafe(args['calendarId'])}/events/{_pathsafe(args['eventId'])}",
                json=body,
            )
            r.raise_for_status()
            return {"event": _slim_event(r.json())}

        if name == "delete-event":
            r = client.delete(
                f"{_API}/calendars/{_pathsafe(args['calendarId'])}/events/{_pathsafe(args['eventId'])}"
            )
            r.raise_for_status()
            return {"deleted": True, "eventId": args["eventId"]}

        if name == "get-freebusy":
            body = {
                "timeMin": args["timeMin"],
                "timeMax": args["timeMax"],
                "timeZone": tz,
                "items": args["calendars"],
            }
            r = client.post(f"{_API}/freeBusy", json=body)
            r.raise_for_status()
            return {"calendars": r.json().get("calendars", {})}

    raise ValueError(f"Unknown calendar tool: {name}")


def _event_body(args: dict, tz: str, partial: bool = False) -> dict:
    body: dict = {}
    for key in ("summary", "description", "location"):
        if args.get(key) is not None:
            body[key] = args[key]
    if args.get("attendees"):
        body["attendees"] = args["attendees"]
    if args.get("start"):
        body["start"] = _start_end_obj(args["start"], tz)
    if args.get("end"):
        body["end"] = _start_end_obj(args["end"], tz)
    return body


def _slim_event(e: dict) -> dict:
    """Trim Google's verbose event object to what the model needs to answer."""
    return {
        "id": e.get("id"),
        "summary": e.get("summary"),
        "start": e.get("start"),
        "end": e.get("end"),
        "location": e.get("location"),
        "htmlLink": e.get("htmlLink"),
        "status": e.get("status"),
    }


def call_calendar_tool(name: str, args: dict) -> dict:
    """Dispatch a calendar tool for the current signed-in user. Mirrors
    mcp_client.call_mcp_tool's return shape: {"result": [json_str], "is_error"}.
    Never raises - failures come back as an error result the agent surfaces."""
    user_id = current_user_id()
    try:
        token = get_access_token(user_id)
        payload = _dispatch(name, args, token)
        return {"result": [json.dumps(payload)], "is_error": False}
    except CalendarAuthError as exc:
        logger.warning("Calendar auth error for user %s: %s", user_id, exc)
        return {
            "result": ["Calendar isn't connected for this account. Ask the user to reconnect "
                       "Google Calendar (sign out and back in)."],
            "is_error": True,
        }
    except httpx.HTTPStatusError as exc:
        logger.warning("Calendar API error (%s): %s", name, exc.response.text[:200])
        return {"result": [f"Calendar API error: {exc.response.status_code} {exc.response.text[:200]}"],
                "is_error": True}
    except Exception as exc:  # noqa: BLE001 - a calendar failure must not crash the turn
        logger.exception("Unexpected calendar error (%s)", name)
        return {"result": [f"Calendar error: {exc}"], "is_error": True}
