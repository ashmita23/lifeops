"""Tool-calling agent with a bounded multi-step loop and a clarification flow.

This hands the LLM real tool definitions and lets it decide which ones to
call, in what order, and with what arguments - up to MAX_TOOL_ITERATIONS
per user turn, so a single message like "list my reminders, mark the tax
one done, and create a calendar event to celebrate" can complete in one
turn instead of needing several round trips.

Rules this loop follows (agreed with the user):
1. Up to MAX_TOOL_ITERATIONS tool calls may be attempted per turn.
2. Each tool result is appended to message history before the next call.
3. No tool call in a response means that response is the final answer.
4. Every attempted action is tracked in AgentTurnResult.actions.
5-8. Destructive tools (name matches _DESTRUCTIVE_KEYWORDS: delete/remove/
     cancel/trash) always pause for a human confirmation turn before
     executing, even mid-chain. The exact proposed call is stored; the next
     turn confirms via CONFIRM_TOOL_NAME, which replays that stored call -
     nothing is ever regenerated or argument-matched. A confirmed action
     executes and then processing CONTINUES so the rest of a multi-part
     request in the same turn still gets handled. A stale/unconfirmed
     pending confirmation is cleared, not left dangling.
9. If the exact same (tool, arguments) repeats in one turn, stop early -
   the model is stuck, not making progress.
10. Hitting the cap forces one final text-only summary of what got done.
11. The system prompt tells the model to resolve vague references (id-less
    reminders/events) via a list/search tool before acting on them.
12. Local tool schemas use OpenAI's strict structured-outputs mode, so
    malformed/extra arguments are rejected by the API itself, not just
    validated after the fact.

Requires a real OPENAI_API_KEY - there is no regex fallback here, since a
tool-calling decision loop isn't something the local parser can approximate.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

# app.config must be imported (and its load_dotenv() run) before langfuse,
# or Langfuse may initialize its singleton client with missing credentials.
from app import google_calendar, guardrails, journal_index, mcp_client, session_store, tokens
from app.config import settings

from langfuse import get_client, observe, propagate_attributes

from app.budget import BudgetExceededError
from app.db import user_scope
from app.llm_client import LLMUnavailableError, call_llm_with_tools, session_scope
from app.schemas import AgentTurnResult, ParsedIntent
from app.services.summary import get_daily_summary
from app.tools.calendar_mock import (
    create_calendar_event,
    delete_calendar_event,
    list_calendar_events,
    update_calendar_event,
)
from app.tools.journal import create_journal_entry, delete_journal_entry, list_journal_entries
from app.tools.reminders import (
    complete_reminder,
    create_reminder,
    delete_reminder,
    list_reminders,
    update_reminder,
)
from app.specialists.booking import book_reservation
from app.specialists.planner import plan_schedule

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5

# Cap on how many messages (excluding the system message) a conversation
# keeps - unbounded growth means more tokens sent per call and more DB
# read/write on every turn for a long-running conversation. No
# summarization, just a bound.
MAX_HISTORY_MESSAGES = 40

# Conversation history and pending delete-confirmations are persisted in
# SQLite (app.session_store) so they survive a process restart - see
# app/db.py's conversations/pending_confirmations tables.

# Local mock-calendar tools, dropped from the merged tool list whenever the
# real Google Calendar MCP server is connected, so the model isn't choosing
# between two competing "create an event" tools.
_LOCAL_CALENDAR_TOOL_NAMES = {
    "create_calendar_event",
    "list_calendar_events",
    "update_calendar_event",
    "delete_calendar_event",
}

CONFIRM_TOOL_NAME = "respond_to_pending_confirmation"

# The only way a destructive action ever actually executes. Deliberately
# never asks the model to regenerate/re-supply the original arguments -
# confirming just replays the EXACT call that was stored when the action
# first paused (see PENDING_CONFIRMATIONS handling below), so there's
# nothing for drifting/re-generated arguments to mismatch against.
_CONFIRM_TOOL = {
    "type": "function",
    "function": {
        "name": CONFIRM_TOOL_NAME,
        "description": (
            "Respond to a pending destructive action that's awaiting the user's "
            "confirmation (you'll see your own previous message asking them to "
            "confirm). Call this instead of calling the original delete/remove/"
            "cancel tool again yourself - this executes the exact original "
            "pending action directly, so its arguments never need to be "
            "reconstructed or guessed a second time."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "confirmed": {
                    "type": "boolean",
                    "description": "true if the user's latest message confirms proceeding, false if they decline",
                },
            },
            "required": ["confirmed"],
            "additionalProperties": False,
        },
    },
}

# OpenAI's strict structured-outputs mode requires every property to be
# listed in "required" (optional fields are expressed as nullable types,
# not by omission) and additionalProperties: false on every object.
_LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Create a reminder for the user at a specific date/time.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title of the reminder"},
                    "due_date": {
                        "type": "string",
                        "description": "ISO 8601 date/time the reminder is due, e.g. 2026-07-09T17:00:00",
                    },
                    "description": {"type": ["string", "null"]},
                    "priority": {"type": ["string", "null"], "enum": ["low", "medium", "high", None]},
                },
                "required": ["title", "due_date", "description", "priority"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Schedule a calendar event with a start time.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {
                        "type": "string",
                        "description": "ISO 8601 start date/time, e.g. 2026-07-10T14:00:00",
                    },
                    "end_time": {"type": ["string", "null"], "description": "ISO 8601 end date/time"},
                    "duration_minutes": {"type": ["integer", "null"]},
                    "description": {"type": ["string", "null"]},
                },
                "required": ["title", "start_time", "end_time", "duration_minutes", "description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_journal_entry",
            "description": "Save a reflective journal entry for the user.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The journal entry text"},
                    "title": {"type": ["string", "null"]},
                },
                "required": ["content", "title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_summary",
            "description": "Get today's reminders, calendar events, and journal entries.",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List reminders so you can find the id of one the user is referring to.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "include_completed": {
                        "type": ["boolean", "null"],
                        "description": "Include already-completed reminders. Defaults to false.",
                    },
                },
                "required": ["include_completed"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_reminder",
            "description": "Update fields on an existing reminder. Only include fields that should change.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The reminder's id"},
                    "title": {"type": ["string", "null"]},
                    "due_date": {"type": ["string", "null"]},
                    "priority": {"type": ["string", "null"], "enum": ["low", "medium", "high", None]},
                    "description": {"type": ["string", "null"]},
                },
                "required": ["id", "title", "due_date", "priority", "description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_reminder",
            "description": "Mark a reminder as completed/done.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "The reminder's id"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_reminder",
            "description": "Permanently delete a reminder. Confirm with the user before calling this.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "The reminder's id"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": "List calendar events so you can find the id of one the user is referring to.",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": "Update fields on an existing calendar event. Only include fields that should change.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The event's id"},
                    "title": {"type": ["string", "null"]},
                    "start_time": {"type": ["string", "null"]},
                    "end_time": {"type": ["string", "null"]},
                    "duration_minutes": {"type": ["integer", "null"]},
                    "description": {"type": ["string", "null"]},
                },
                "required": ["id", "title", "start_time", "end_time", "duration_minutes", "description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Permanently delete a calendar event. Confirm with the user before calling this.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "The event's id"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_journal_entries",
            "description": "List journal entries so you can find the id of one the user is referring to.",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_journal_entry",
            "description": "Permanently delete a journal entry. Confirm with the user before calling this.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "The journal entry's id"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_journal",
            "description": (
                "Semantic search over the user's past journal entries by meaning (not exact "
                "words). Call this to answer reflective or recall questions about what they've "
                "journaled - e.g. 'what have I been worried about?', 'summarize my mood this "
                "month' - then ground your answer in the returned entries."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, in natural language"},
                    "k": {"type": ["integer", "null"], "description": "How many entries to retrieve (default 5)"},
                },
                "required": ["query", "k"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_schedule",
            "description": (
                "Delegate to the planner specialist to find free time and propose WHEN to "
                "schedule something. Use this when the user wants help deciding a time "
                "(e.g. 'find an hour tomorrow morning for deep work'), NOT when they give an "
                "explicit time. Returns proposed start times; confirm one with the user, then "
                "create the event with the calendar tool."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "What to schedule"},
                    "date": {"type": "string", "description": "ISO date (YYYY-MM-DD) to schedule on"},
                    "duration_minutes": {"type": "integer", "description": "Length of the block in minutes"},
                    "preference": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "evening", "any"],
                        "description": "Preferred time of day",
                    },
                },
                "required": ["title", "date", "duration_minutes", "preference"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_reservation",
            "description": (
                "Delegate to the booking specialist to reserve a restaurant table. This is a "
                "high-stakes action requiring the user's explicit approval AND ID verification "
                "(the guest name) before it executes - expect it to pause for confirmation."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant": {"type": "string", "description": "Restaurant name"},
                    "datetime": {"type": "string", "description": "ISO date/time of the reservation"},
                    "party_size": {"type": "integer", "description": "Number of guests"},
                    "guest_name": {"type": "string", "description": "Name on the reservation, used for ID verification"},
                },
                "required": ["restaurant", "datetime", "party_size", "guest_name"],
                "additionalProperties": False,
            },
        },
    },
]


def _dispatch_create_reminder(args: dict, raw_text: str) -> dict:
    intent = ParsedIntent(
        intent_type="reminder",
        title=args.get("title"),
        description=args.get("description"),
        due_date=args.get("due_date"),
        priority=args.get("priority"),
        raw_text=raw_text,
    )
    return create_reminder(intent)


def _dispatch_create_calendar_event(args: dict, raw_text: str) -> dict:
    intent = ParsedIntent(
        intent_type="calendar_event",
        title=args.get("title"),
        description=args.get("description"),
        start_time=args.get("start_time"),
        end_time=args.get("end_time"),
        duration_minutes=args.get("duration_minutes"),
        raw_text=raw_text,
    )
    return create_calendar_event(intent)


def _dispatch_create_journal_entry(args: dict, raw_text: str) -> dict:
    intent = ParsedIntent(
        intent_type="journal_entry",
        title=args.get("title"),
        description=args.get("content"),
        raw_text=raw_text,
    )
    record = create_journal_entry(intent)
    # Index for RAG retrieval. Best-effort: an embedding/Pinecone failure (or no
    # Pinecone configured) must never fail the write. No-op when RAG is off.
    try:
        journal_index.index_entry(record["id"], record.get("content") or "")
    except Exception:
        logger.warning("Failed to index journal entry %s for RAG.", record.get("id"), exc_info=True)
    return record


def _dispatch_search_journal(args: dict, raw_text: str) -> dict:
    if not settings.rag_enabled:
        return {"status": "unavailable", "reason": "journal search (RAG) is not configured"}
    results = journal_index.retrieve(args["query"], args.get("k") or 5)
    return {"results": results}


def _dispatch_get_daily_summary(args: dict, raw_text: str) -> dict:
    return get_daily_summary()


def _dispatch_list_reminders(args: dict, raw_text: str) -> dict:
    return {"reminders": list_reminders(include_completed=bool(args.get("include_completed", False)))}


def _dispatch_update_reminder(args: dict, raw_text: str) -> dict:
    record = update_reminder(
        reminder_id=args["id"],
        title=args.get("title"),
        due_date=args.get("due_date"),
        priority=args.get("priority"),
        description=args.get("description"),
    )
    return record if record else {"error": f"No reminder found with id {args.get('id')}"}


def _dispatch_complete_reminder(args: dict, raw_text: str) -> dict:
    record = complete_reminder(reminder_id=args["id"])
    return record if record else {"error": f"No reminder found with id {args.get('id')}"}


def _dispatch_delete_reminder(args: dict, raw_text: str) -> dict:
    deleted = delete_reminder(reminder_id=args["id"])
    return {"deleted": deleted, "id": args.get("id")}


def _dispatch_list_calendar_events(args: dict, raw_text: str) -> dict:
    return {"calendar_events": list_calendar_events()}


def _dispatch_update_calendar_event(args: dict, raw_text: str) -> dict:
    record = update_calendar_event(
        event_id=args["id"],
        title=args.get("title"),
        start_time=args.get("start_time"),
        end_time=args.get("end_time"),
        duration_minutes=args.get("duration_minutes"),
        description=args.get("description"),
    )
    return record if record else {"error": f"No calendar event found with id {args.get('id')}"}


def _dispatch_delete_calendar_event(args: dict, raw_text: str) -> dict:
    deleted = delete_calendar_event(event_id=args["id"])
    return {"deleted": deleted, "id": args.get("id")}


def _dispatch_list_journal_entries(args: dict, raw_text: str) -> dict:
    return {"journal_entries": list_journal_entries()}


def _dispatch_delete_journal_entry(args: dict, raw_text: str) -> dict:
    deleted = delete_journal_entry(entry_id=args["id"])
    return {"deleted": deleted, "id": args.get("id")}


TOOL_DISPATCH = {
    "create_reminder": _dispatch_create_reminder,
    "create_calendar_event": _dispatch_create_calendar_event,
    "create_journal_entry": _dispatch_create_journal_entry,
    "get_daily_summary": _dispatch_get_daily_summary,
    "list_reminders": _dispatch_list_reminders,
    "update_reminder": _dispatch_update_reminder,
    "complete_reminder": _dispatch_complete_reminder,
    "delete_reminder": _dispatch_delete_reminder,
    "list_calendar_events": _dispatch_list_calendar_events,
    "update_calendar_event": _dispatch_update_calendar_event,
    "delete_calendar_event": _dispatch_delete_calendar_event,
    "list_journal_entries": _dispatch_list_journal_entries,
    "delete_journal_entry": _dispatch_delete_journal_entry,
    "search_journal": _dispatch_search_journal,
    # Specialist tools the supervisor delegates to (app/specialists/).
    "plan_schedule": plan_schedule,
    "book_reservation": book_reservation,
}


@observe(as_type="tool", capture_input=False)
def _execute_local_tool(tool_name: str, args: dict, input_text: str) -> dict:
    get_client().update_current_span(name=tool_name, input=args)
    return TOOL_DISPATCH[tool_name](args, input_text)


_DESTRUCTIVE_KEYWORDS = ("delete", "remove", "cancel", "trash")

# Non-destructive but high-stakes tools that also need explicit human approval
# before executing (e.g. booking spends money / makes a commitment in the real
# world). Listed by exact name since they don't share a destructive verb.
_APPROVAL_REQUIRED_TOOLS = frozenset({"book_reservation"})


def _is_destructive_tool(name: str) -> bool:
    # Name-based heuristic - covers our own tools (all "delete_*") and the
    # current MCP server's "delete-event". Broadened beyond just "delete"
    # since we don't control MCP tool naming and a future server version
    # could use a different verb (e.g. "remove-event", "cancel-event")
    # without us noticing the safety gate silently stopped applying.
    lowered = name.lower()
    return any(keyword in lowered for keyword in _DESTRUCTIVE_KEYWORDS)


def _requires_approval(name: str) -> bool:
    """Any tool that must pause for explicit human confirmation before it
    executes - destructive tools plus the named high-stakes ones. This is the
    single gate both the tool-hiding logic and the pause logic consult."""
    return _is_destructive_tool(name) or name in _APPROVAL_REQUIRED_TOOLS


def _record_status(record: dict) -> tuple[str, str | None]:
    if isinstance(record, dict):
        if record.get("error"):
            return "error", str(record["error"])
        if record.get("is_error"):
            return "error", "; ".join(str(item) for item in record.get("result", [])) or "MCP tool reported an error"
    return "success", None


_FALLBACK_MESSAGES = {
    "confirm": "Please confirm: should I go ahead with that?",
    "repeat": "I tried the same action twice with identical arguments and stopped to avoid looping.",
    "cap": "I've made several changes this turn and stopped at the per-turn action limit.",
}

_DECLINE_MESSAGE = "Okay, I won't go ahead with that."

# Deterministic backstop for the "confirm" synthesis call: if the model's
# phrasing still claims completion despite _CONFIRM_CONTEXT_NOTE, it's
# discarded for the canned fallback instead. This does NOT depend on the
# note working - it's a second, independent guardrail against the synthesis
# step narrating a false "done" for an action that hasn't executed yet.
#
# Deliberately only PAST-TENSE outcome words ("deleted", not "delete"). The
# earlier version also blocked "done"/"successfully"/"all set", which show up
# in perfectly valid confirmation questions ("Once you confirm, this will be
# done - proceed?") and caused false positives. A "should I ...?" question
# uses present-tense verbs ("delete it?", "cancel it?"), so past-tense words
# are a much tighter signal for an actual (false) completion claim.
_FALSE_COMPLETION_MARKERS = (
    "deleted", "removed", "cancelled", "canceled", "has been", "have been",
)

_CONFIRM_CONTEXT_NOTE = {
    "role": "system",
    "content": (
        "Reminder: the destructive action you just proposed has NOT executed yet - it is only "
        "awaiting the user's yes/no confirmation. Do not say it is done, deleted, removed, "
        "cancelled, or completed. Only ask the user to confirm."
    ),
}


def _trim_history(messages: list[dict]) -> list[dict]:
    """Keeps the system message plus at most MAX_HISTORY_MESSAGES more,
    never cutting into the middle of an assistant-tool_calls/tool sequence
    (which would leave a "tool" message whose tool_call_id was never
    declared - API-invalid). Only starts the truncated history at a "user"
    message boundary."""
    if len(messages) <= MAX_HISTORY_MESSAGES + 1:  # +1 for the system message
        return messages

    system_message = messages[0]
    rest = messages[1:]
    cutoff = len(rest) - MAX_HISTORY_MESSAGES
    while cutoff < len(rest) and rest[cutoff].get("role") != "user":
        cutoff += 1
    return [system_message] + rest[cutoff:]


def _final_text(messages: list[dict], tools: list[dict], reason: str) -> str:
    """Makes one text-only (tool_choice='none') call so the model can phrase
    a natural confirmation/summary, with a canned fallback if unavailable.

    reason == "confirm" means nothing has executed yet - only a confirmation
    question is being asked. _CONFIRM_CONTEXT_NOTE is appended ephemerally
    (never persisted to real history) to steer the model away from implying
    completion. As an independent, deterministic backstop that doesn't rely
    on the note actually working, the resulting text is also scanned for
    false-completion language and discarded for the canned fallback if found -
    this is what protects against the synthesis step lying about success
    even if some unrelated bug produces a weird tool result later."""
    # Trim to a bounded window for the model (the full history stays in the
    # durable log); the ephemeral confirm note is added after trimming so it
    # is never dropped by the window.
    call_messages = _trim_history(messages)
    if reason == "confirm":
        call_messages = call_messages + [_CONFIRM_CONTEXT_NOTE]

    try:
        synthesis_message = call_llm_with_tools(call_messages, tools, tool_choice="none")
        final_text = synthesis_message.content or _FALLBACK_MESSAGES.get(reason, "Done.")
    except (LLMUnavailableError, BudgetExceededError):
        # No model available (no key) or budget/rate cap hit during synthesis -
        # fall back to the canned phrasing rather than crashing the turn.
        final_text = _FALLBACK_MESSAGES.get(reason, "Done.")

    if reason == "confirm" and any(marker in final_text.lower() for marker in _FALSE_COMPLETION_MARKERS):
        final_text = _FALLBACK_MESSAGES["confirm"]

    messages.append({"role": "assistant", "content": final_text})
    return final_text


def _user_calendar_active() -> bool:
    """True when the signed-in user has their own Google Calendar connected, so
    the agent should act on THEIR calendar (app/google_calendar.py) rather than
    the single global MCP calendar. Decided here and in dispatch off the same
    condition so the offered tools and the executor never disagree."""
    from app.db import current_user_id

    return settings.google_login_enabled and tokens.has_calendar_credentials(current_user_id())


def _build_tools(include_confirm_tool: bool) -> tuple[list[dict], bool]:
    """Merges local tools with the active calendar backend's tools. Returns
    (tools, calendar_active).

    Calendar backend, in priority order:
      1. the signed-in user's own Google Calendar (multi-user), or
      2. the single global Google Calendar MCP server (single-user), or
      3. none -> the local mock calendar tools stay in the list.

    include_confirm_tool: only offer CONFIRM_TOOL_NAME when there's an
    actual pending confirmation for this session - otherwise it's a
    meaningless option that could confuse the model into calling it
    without cause."""
    if _user_calendar_active():
        calendar_tools = google_calendar.CALENDAR_TOOLS
    else:
        calendar_tools = mcp_client.get_mcp_tools()
    calendar_active = bool(calendar_tools)

    local_tools = _LOCAL_TOOLS
    if calendar_active:
        local_tools = [
            tool for tool in _LOCAL_TOOLS if tool["function"]["name"] not in _LOCAL_CALENDAR_TOOL_NAMES
        ]

    tools = local_tools + calendar_tools

    if include_confirm_tool:
        # While an action is pending confirmation, hide ALL approval-required
        # tools (destructive + high-stakes like booking) from the model for
        # this turn - the only way to actually execute one is via the confirm
        # tool below, which replays the exact original pending call. This is a
        # hard, code-enforced guarantee, not just a prompt instruction.
        tools = [t for t in tools if not _requires_approval(t["function"]["name"])]
        tools = tools + [_CONFIRM_TOOL]

    return tools, calendar_active


def _system_prompt(timezone: str, calendar_active: bool) -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz).replace(microsecond=0).isoformat()

    return (
        "You are LifeOps Agent, a personal productivity assistant. "
        f"The current date/time is {now} in timezone {timezone}. "
        "Resolve any relative dates the user mentions (e.g. 'tomorrow', 'Friday', "
        "'in 2 hours') against this current date/time, and pass absolute ISO 8601 "
        "date/times to tools.\n\n"
        "You may call multiple tools across several steps in one turn (up to "
        f"{MAX_TOOL_ITERATIONS}) to fully complete requests that span more than one "
        "action - e.g. listing something, then updating or creating based on what "
        "you found. Keep going until the request is fully handled or you need "
        "information only the user can provide. If required information is missing "
        "(e.g. a reminder with no date/time), do NOT call a tool - instead ask a "
        "short, specific clarifying question in plain text.\n\n"
        "When a single message contains more than one distinct request (e.g. "
        "'schedule X and also delete Y'), address every part of it - never silently "
        "act on only one and drop the rest. If one part pauses for confirmation "
        "(e.g. a destructive action), your reply must explicitly say what you're "
        "still waiting to do, so it isn't forgotten once the conversation moves on to "
        "confirming the first part.\n\n"
        "Never state a fact about the user's schedule (e.g. 'you have no other "
        "meetings', 'your day is free') unless you have an actual, current tool "
        "result in front of you that covers the FULL scope of what you're claiming. "
        "A list/search call scoped to a narrow window (e.g. just tomorrow morning, "
        "to find a free slot) only supports a claim about that narrow window - it "
        "does not support a claim about the rest of the day or anything you haven't "
        "actually queried. If asked about the whole day and you've only checked part "
        "of it, call the list tool again for the full range before answering.\n\n"
        "After a tool result comes back, answer the user directly and usefully using "
        "that data - don't just confirm the tool ran. For example, if get_daily_summary "
        "comes back empty, tell them they're free and offer to schedule something; if "
        "asked about availability, reason over the returned events/reminders instead of "
        "just repeating them.\n\n"
        "You can also list, update, complete, and delete reminders"
        + (" and journal entries." if calendar_active else ", calendar events, and journal entries.")
        + " If the user refers to something without a known "
        "id (e.g. 'delete my dentist reminder', 'mark the tax thing done'), call the "
        "matching list tool first to find its id before acting on it. Always confirm "
        "with the user before deleting anything, since deletion is irreversible - "
        "expect a delete request to pause and ask for confirmation before it executes. "
        "When you see your own previous message asking to confirm a pending destructive "
        f"action, and the user's new message responds to it, call {CONFIRM_TOOL_NAME} "
        "with confirmed=true or confirmed=false - do NOT call the original delete/"
        "remove/cancel tool again yourself; that tool isn't available right now for "
        "exactly this reason, and respond_to_pending_confirmation executes the exact "
        "original pending action directly if confirmed.\n\n"
        "Reminders and calendar events are two distinct, separate things - a reminder is "
        "a private to-do tracked only in this app (title + due date), a calendar event is "
        "a real, shareable, time-blocked entry on the user's actual calendar. Pick exactly "
        "ONE based on the user's wording: 'remind me to X' means create_reminder only; "
        "'schedule/set up/add a calendar event/invite/meeting' means the calendar tool "
        "only. Only create both if the user explicitly asks for both (e.g. 'remind me AND "
        "put it on my calendar') - do not create a calendar event just because a reminder "
        "was requested, or vice versa.\n\n"
        "You delegate two kinds of work to specialists. When the user wants help DECIDING "
        "when to do something rather than giving an explicit time (e.g. 'find me an hour "
        "tomorrow morning for deep work', 'when can I fit in a workout?'), call plan_schedule "
        "to get proposed free slots, then present a slot and, once the user picks one, create "
        "the calendar event. When the user wants to reserve a restaurant table, call "
        "book_reservation - it needs a guest name for ID verification and will pause for the "
        "user's explicit approval before booking (respond via respond_to_pending_confirmation "
        "just like a delete).\n\n"
        "For reflective or recall questions about the user's past journal entries (e.g. 'what "
        "have I been anxious about?', 'summarize my mood this month'), call search_journal "
        "first and ground your answer in the entries it returns - do not answer from memory."
        + (
            "\n\nYou have access to the user's real Google Calendar through the connected "
            "calendar tools - use those (not any local mock) for scheduling, listing, "
            "updating, or deleting calendar events. When searching/checking for an existing "
            "event by name (e.g. to cancel or reference it) and a specific keyword search "
            "returns no matches, do not immediately conclude it doesn't exist - list/search "
            "more broadly (e.g. all events in the relevant date range) before telling the "
            "user nothing was found, since titles are often worded differently than how the "
            "user refers to them in conversation."
            if calendar_active
            else "\n\nCalendar events are currently stored locally only (no real Google "
            "Calendar connected yet)."
        )
    )


@observe(name="run_agent_turn", as_type="agent", capture_input=False, capture_output=False)
def run_agent_turn(
    session_id: str | None,
    input_text: str,
    timezone: str = "America/Chicago",
    user_id: str | None = None,
) -> AgentTurnResult:
    # Resolve the session id up front so every branch below (including the
    # demo-mode early return) can be tagged and grouped into the same
    # Langfuse Session - this is a multi-turn conversation app, so session_id
    # is the right thing to group traces by (see app.session_store).
    session_id = session_id or str(uuid.uuid4())

    # session_scope makes session_id available to call_llm_with_tools (for
    # per-session budget/usage) without adding it as a parameter that would
    # break the mocked signature in tests. user_scope does the same for the
    # signed-in user id so the per-user data tools (reminders, journal, RAG)
    # scope their queries - None means single-user (DEFAULT_USER_ID).
    # propagate_attributes groups the Langfuse trace by the same session.
    with user_scope(user_id), session_scope(session_id), propagate_attributes(session_id=session_id, trace_name="run_agent_turn"):
        # Set explicit trace input/output instead of letting @observe capture
        # raw function args (which would include internal params like
        # timezone) - keep the trace readable and scoped to what matters.
        get_client().update_current_span(input=input_text)

        result = _run_agent_turn_body(session_id, input_text, timezone)
        # An explicit application-level "this run is really done" signal -
        # Langfuse's own timestamps/observation list can look plausible while
        # more child observations are still being indexed, so app.trace_export
        # checks for this marker on the root span rather than inferring
        # completeness purely from structure.
        get_client().update_current_span(
            output=result.message, metadata={"lifeops_export_marker": "complete"}
        )
        result.trace_id = get_client().get_current_trace_id()
        return result


@dataclass
class _TurnState:
    """All state that accumulates across one user turn's tool-calling loop.

    Pulled into an explicit object (instead of a stack of closures mutating
    enclosing locals) so each mutation is a named method call you can follow -
    and so the per-call logic can live in a free function that just takes this
    state, rather than being trapped inside one 200-line function."""

    session_id: str
    input_text: str
    tools: list[dict]
    messages: list[dict]
    persisted_count: int
    pending_before: dict | None
    actions: list[dict] = field(default_factory=list)
    seen_calls: set = field(default_factory=set)
    last_tool_name: str | None = None
    last_record: dict | None = None

    def append_assistant_tool_calls(self, tool_calls) -> None:
        # One assistant message carrying ALL tool calls from this response,
        # matching what the API actually returned - not one message per call.
        # Every id here needs a matching "tool" result message before the next
        # LLM call, real or a synthetic "not executed" placeholder.
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"},
                }
                for c in tool_calls
            ],
        })

    def append_tool_result(self, call_id: str, content: dict) -> None:
        # Fence any injection-shaped text in the result (e.g. a calendar event
        # titled "ignore previous instructions...") so it re-enters context as
        # data, not a command. No-op for clean results.
        content = guardrails.fence_if_untrusted(content)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(content)})

    def record_action(self, tool_name: str, args: dict, record: dict) -> None:
        status, error = _record_status(record)
        self.actions.append(
            {"tool": tool_name, "arguments": args, "status": status, "result": record, "error": error}
        )
        self.last_tool_name, self.last_record = tool_name, record

    def finish(
        self, *, done: bool, message: str, stored_record: dict | None = None, tool_called: str | None = None
    ) -> AgentTurnResult:
        # The single choke point every return path goes through: persist only
        # the messages produced this turn (append-only, never rewriting an
        # existing row - that's what keeps concurrent turns from clobbering
        # each other) and build the result.
        session_store.append_messages(self.session_id, self.messages[self.persisted_count:])
        return AgentTurnResult(
            session_id=self.session_id,
            done=done,
            message=message,
            stored_record=stored_record,
            tool_called=tool_called,
            actions=self.actions,
        )


@dataclass
class _Outcome:
    """What processing one tool call tells the batch loop to do next:
    - "continue": keep processing the rest of the batch
    - "stop":     stop the batch with `reason`, handled after the loop
    - "return":   end the whole turn immediately with `result`"""

    kind: str
    reason: str | None = None
    tool_name: str | None = None
    result: AgentTurnResult | None = None


_CONTINUE = _Outcome("continue")


def _process_confirmation(state: _TurnState, call, args: dict) -> _Outcome:
    """Handle the meta-tool that satisfies a pending destructive confirmation
    by replaying the EXACT stored call - never the arguments this call carries
    (it only has `confirmed`), so nothing can drift or mismatch."""
    confirmed = bool(args.get("confirmed"))
    state.append_tool_result(call.id, {"status": "confirmed" if confirmed else "declined"})

    if not confirmed:
        return _Outcome("return", result=state.finish(done=False, message=_DECLINE_MESSAGE))

    if state.pending_before is None:
        # Shouldn't happen (the tool is only offered when a pending
        # confirmation exists), but don't crash if it does.
        return _CONTINUE

    tool_name = state.pending_before["tool"]
    tool_args = state.pending_before["arguments"]
    if tool_name in TOOL_DISPATCH:
        record = _execute_local_tool(tool_name, tool_args, state.input_text)
    elif _user_calendar_active() and google_calendar.is_calendar_tool(tool_name):
        record = google_calendar.call_calendar_tool(tool_name, tool_args)
    else:
        record = mcp_client.call_mcp_tool(tool_name, tool_args)

    state.record_action(tool_name, tool_args, record)
    state.pending_before = None  # consumed - a later call in this batch can't reuse it
    return _CONTINUE


def _process_one_call(state: _TurnState, call) -> _Outcome:
    """Process a single tool call from a (possibly batched) response and say
    what the batch loop should do next. Each guard is one branch - cap,
    repeat, confirm meta-tool, unknown tool, destructive pause, execute."""
    tool_name = call.function.name

    if len(state.actions) >= MAX_TOOL_ITERATIONS:
        state.append_tool_result(call.id, {"status": "not_executed", "reason": "cap_reached"})
        return _Outcome("stop", reason="cap")

    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    # Rule 9: same (tool, args) repeating this turn means the model is stuck -
    # stop rather than burning the rest of the cap.
    signature = (tool_name, json.dumps(args, sort_keys=True))
    if signature in state.seen_calls:
        state.append_tool_result(call.id, {"status": "not_executed", "reason": "repeat_detected"})
        return _Outcome("stop", reason="repeat")
    state.seen_calls.add(signature)

    # The confirm tool is a meta-tool, not a dispatchable resource action -
    # handle it before the unknown-tool check (it's never in TOOL_DISPATCH or
    # the MCP tool list) and before the destructive gate (it's how that gate
    # gets satisfied).
    if tool_name == CONFIRM_TOOL_NAME:
        return _process_confirmation(state, call, args)

    # Calendar routing mirrors _build_tools: if the signed-in user has their own
    # calendar connected, calendar tool calls execute against THEIR calendar;
    # otherwise they fall through to the global MCP calendar.
    user_calendar = _user_calendar_active()
    is_known_local = tool_name in TOOL_DISPATCH
    is_known_user_calendar = (
        not is_known_local and user_calendar and google_calendar.is_calendar_tool(tool_name)
    )
    is_known_mcp = (
        not is_known_local and not user_calendar and mcp_client.is_mcp_tool(tool_name)
    )

    if not is_known_local and not is_known_user_calendar and not is_known_mcp:
        state.actions.append(
            {"tool": tool_name, "arguments": args, "status": "error", "result": None, "error": "unknown tool"}
        )
        state.append_tool_result(call.id, {"error": f"Unknown tool '{tool_name}'"})
        return _Outcome("stop", reason="unknown", tool_name=tool_name)

    # Rules 5-8: proposing an approval-required tool (destructive OR high-stakes
    # like booking) always pauses for an explicit human confirmation turn before
    # it executes - no exceptions. The ONLY execution path is via
    # CONFIRM_TOOL_NAME, which replays the stored call exactly, so this never
    # argument-matches.
    if _requires_approval(tool_name):
        session_store.set_pending_confirmation(state.session_id, tool_name, args)
        state.append_tool_result(call.id, {"status": "awaiting_confirmation"})
        return _Outcome("stop", reason="confirm", tool_name=tool_name)

    if is_known_local:
        record = _execute_local_tool(tool_name, args, state.input_text)
    elif is_known_user_calendar:
        record = google_calendar.call_calendar_tool(tool_name, args)
    else:
        record = mcp_client.call_mcp_tool(tool_name, args)
    state.append_tool_result(call.id, record)
    state.record_action(tool_name, args, record)
    # A confirmed/normal execution falls through and the batch keeps going, so
    # the model can pick back up the rest of a multi-part request this turn.
    return _CONTINUE


def _finish_stopped(state: _TurnState, stop_reason: str, stop_tool_name: str | None) -> AgentTurnResult:
    """Build the terminal result for a batch that stopped early. `unknown` and
    `confirm` haven't completed the user's request (done=False); `cap` has done
    as much as it will (done=True); `repeat` is done iff anything ran."""
    if stop_reason == "unknown":
        return state.finish(
            done=False,
            message=f"Model tried to call unknown tool '{stop_tool_name}'.",
            tool_called=stop_tool_name,
        )
    if stop_reason == "confirm":
        return state.finish(
            done=False,
            message=_final_text(state.messages, state.tools, reason="confirm"),
            tool_called=stop_tool_name,
        )
    if stop_reason == "repeat":
        return state.finish(
            done=bool(state.actions),
            message=_final_text(state.messages, state.tools, reason="repeat"),
            stored_record=state.last_record,
            tool_called=state.last_tool_name,
        )
    # cap
    return state.finish(
        done=True,
        message=_final_text(state.messages, state.tools, reason="cap"),
        stored_record=state.last_record,
        tool_called=state.last_tool_name,
    )


def _run_agent_turn_body(session_id: str, input_text: str, timezone: str) -> AgentTurnResult:
    if settings.demo_mode:
        return AgentTurnResult(
            session_id=session_id,
            done=False,
            message=(
                "Agent mode requires OPENAI_API_KEY to be set (tool-calling needs a "
                "real model). Use /command for the offline regex-based demo."
            ),
            stored_record=None,
            tool_called=None,
            actions=[],
        )

    # Flag injection-shaped user input for observability. Not blocked - the
    # user is the principal and may legitimately say "ignore that"; the real
    # defense is fencing untrusted TOOL DATA (see _TurnState.append_tool_result)
    # plus the code-enforced approval gate.
    flagged = guardrails.scan(input_text)
    if flagged:
        logger.warning("guardrail: user input matched injection patterns %s", flagged)

    # Snapshot (and clear) any pending delete-confirmation for this session.
    # Rule 8: if this turn doesn't re-request confirmation, the stale pending
    # state is gone - it never lingers past one follow-up turn.
    pending_before = session_store.pop_pending_confirmation(session_id)

    tools, calendar_active = _build_tools(include_confirm_tool=pending_before is not None)

    messages = session_store.get_session_messages(session_id)
    if messages is None:
        # Brand-new session: the system message is freshly built and NOT yet
        # persisted, so persisted_count is 0 - finish() must write it too.
        messages = [{"role": "system", "content": _system_prompt(timezone, calendar_active)}]
        persisted_count = 0
    else:
        # Loaded from the append-only log: every row is already durable, so
        # only messages produced from here on are new.
        persisted_count = len(messages)

    messages.append({"role": "user", "content": input_text})

    state = _TurnState(
        session_id=session_id,
        input_text=input_text,
        tools=tools,
        messages=messages,
        persisted_count=persisted_count,
        pending_before=pending_before,
    )

    while len(state.actions) < MAX_TOOL_ITERATIONS:
        try:
            response_message = call_llm_with_tools(_trim_history(state.messages), tools)
        except LLMUnavailableError:
            return state.finish(
                done=bool(state.actions),
                message="Agent mode requires OPENAI_API_KEY to be set.",
                stored_record=state.last_record,
                tool_called=state.last_tool_name,
            )
        except BudgetExceededError as exc:
            # Per-session budget/rate cap hit. Stop cleanly with what's already
            # done rather than spending more - done=True since we won't retry.
            return state.finish(
                done=True,
                message=str(exc),
                stored_record=state.last_record,
                tool_called=state.last_tool_name,
            )

        tool_calls = getattr(response_message, "tool_calls", None)
        if not tool_calls:
            assistant_text = response_message.content or "Could you clarify what you'd like me to do?"
            state.messages.append({"role": "assistant", "content": assistant_text})
            return state.finish(
                done=bool(state.actions),
                message=assistant_text,
                stored_record=state.last_record,
                tool_called=state.last_tool_name,
            )

        # The model may batch multiple actions into one response (confirmed
        # live: 2-4 tool calls at once). Process them all in this round-trip.
        state.append_assistant_tool_calls(tool_calls)

        stop_reason: str | None = None
        stop_tool_name: str | None = None
        for call in tool_calls:
            if stop_reason is not None:
                # Already stopping this turn - every remaining call still needs
                # a result so the message history stays valid, without
                # executing anything further.
                state.append_tool_result(call.id, {"status": "not_executed", "reason": stop_reason})
                continue

            outcome = _process_one_call(state, call)
            if outcome.kind == "return":
                return outcome.result
            if outcome.kind == "stop":
                stop_reason = outcome.reason
                stop_tool_name = outcome.tool_name

        if stop_reason is not None:
            return _finish_stopped(state, stop_reason, stop_tool_name)
        # Clean batch - loop back for the model's next decision or final answer.

    # Rule 10: cap hit without an in-batch cap stop above (defensive fallback).
    return _finish_stopped(state, "cap", None)
