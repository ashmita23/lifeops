"""Tool-calling agent with a bounded multi-step loop and a clarification flow.

Unlike app.parser/app.services.router (single-shot classify-then-route),
this hands the LLM real tool definitions and lets it decide which ones to
call, in what order, and with what arguments - up to MAX_TOOL_ITERATIONS
per user turn, so a single message like "list my reminders, mark the tax
one done, and create a calendar event to celebrate" can complete in one
turn instead of needing several round trips.

Rules this loop follows (agreed with the user):
1. Up to MAX_TOOL_ITERATIONS tool calls may be attempted per turn.
2. Each tool result is appended to message history before the next call.
3. No tool call in a response means that response is the final answer.
4. Every attempted action is tracked in AgentTurnResult.actions.
5-8. Destructive tools (name contains "delete") always pause for a human
     confirmation turn before executing, even mid-chain; a confirmed delete
     executes and ends the turn immediately (no further chaining); an
     unconfirmed/stale pending confirmation is cleared, not left dangling.
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
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

# app.config must be imported (and its load_dotenv() run) before langfuse,
# or Langfuse may initialize its singleton client with missing credentials.
from app import mcp_client, session_store
from app.config import settings

from langfuse import get_client, observe, propagate_attributes

from app.llm_client import LLMUnavailableError, call_llm_with_tools
from app.schemas import AgentTurnResult, ParsedIntent
from app.services.router import get_daily_summary
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

MAX_TOOL_ITERATIONS = 5

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
    return create_journal_entry(intent)


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
}


@observe(as_type="tool", capture_input=False)
def _execute_local_tool(tool_name: str, args: dict, input_text: str) -> dict:
    get_client().update_current_span(name=tool_name, input=args)
    return TOOL_DISPATCH[tool_name](args, input_text)


def _is_destructive_tool(name: str) -> bool:
    return "delete" in name.lower()


def _record_status(record: dict) -> tuple[str, str | None]:
    if isinstance(record, dict):
        if record.get("error"):
            return "error", str(record["error"])
        if record.get("is_error"):
            return "error", "; ".join(str(item) for item in record.get("result", [])) or "MCP tool reported an error"
    return "success", None


def _assistant_tool_call_message(call, tool_name: str, raw_args: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": tool_name, "arguments": raw_args}}
        ],
    }


_FALLBACK_MESSAGES = {
    "confirm": "Please confirm: should I go ahead with that?",
    "repeat": "I tried the same action twice with identical arguments and stopped to avoid looping.",
    "cap": "I've made several changes this turn and stopped at the per-turn action limit.",
}


def _final_text(messages: list[dict], tools: list[dict], reason: str) -> str:
    """Makes one text-only (tool_choice='none') call so the model can phrase
    a natural confirmation/summary, with a canned fallback if unavailable."""
    try:
        synthesis_message = call_llm_with_tools(messages, tools, tool_choice="none")
        final_text = synthesis_message.content or _FALLBACK_MESSAGES.get(reason, "Done.")
    except LLMUnavailableError:
        final_text = _FALLBACK_MESSAGES.get(reason, "Done.")
    messages.append({"role": "assistant", "content": final_text})
    return final_text


def _build_tools() -> tuple[list[dict], bool]:
    """Merges local tools with whatever the Google Calendar MCP server
    exposes (if connected). Returns (tools, mcp_active)."""
    mcp_tools = mcp_client.get_mcp_tools()
    mcp_active = bool(mcp_tools)

    local_tools = _LOCAL_TOOLS
    if mcp_active:
        local_tools = [
            tool for tool in _LOCAL_TOOLS if tool["function"]["name"] not in _LOCAL_CALENDAR_TOOL_NAMES
        ]

    return local_tools + mcp_tools, mcp_active


def _system_prompt(timezone: str, mcp_active: bool) -> str:
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
        "After a tool result comes back, answer the user directly and usefully using "
        "that data - don't just confirm the tool ran. For example, if get_daily_summary "
        "comes back empty, tell them they're free and offer to schedule something; if "
        "asked about availability, reason over the returned events/reminders instead of "
        "just repeating them.\n\n"
        "You can also list, update, complete, and delete reminders"
        + (" and journal entries." if mcp_active else ", calendar events, and journal entries.")
        + " If the user refers to something without a known "
        "id (e.g. 'delete my dentist reminder', 'mark the tax thing done'), call the "
        "matching list tool first to find its id before acting on it. Always confirm "
        "with the user before deleting anything, since deletion is irreversible - "
        "expect a delete request to pause and ask for confirmation before it executes.\n\n"
        "Reminders and calendar events are two distinct, separate things - a reminder is "
        "a private to-do tracked only in this app (title + due date), a calendar event is "
        "a real, shareable, time-blocked entry on the user's actual calendar. Pick exactly "
        "ONE based on the user's wording: 'remind me to X' means create_reminder only; "
        "'schedule/set up/add a calendar event/invite/meeting' means the calendar tool "
        "only. Only create both if the user explicitly asks for both (e.g. 'remind me AND "
        "put it on my calendar') - do not create a calendar event just because a reminder "
        "was requested, or vice versa."
        + (
            "\n\nYou have access to the user's real Google Calendar through the connected "
            "calendar tools - use those (not any local mock) for scheduling, listing, "
            "updating, or deleting calendar events. When searching/checking for an existing "
            "event by name (e.g. to cancel or reference it) and a specific keyword search "
            "returns no matches, do not immediately conclude it doesn't exist - list/search "
            "more broadly (e.g. all events in the relevant date range) before telling the "
            "user nothing was found, since titles are often worded differently than how the "
            "user refers to them in conversation."
            if mcp_active
            else "\n\nCalendar events are currently stored locally only (no real Google "
            "Calendar connected yet)."
        )
    )


@observe(name="run_agent_turn", as_type="agent", capture_input=False, capture_output=False)
def run_agent_turn(
    session_id: str | None, input_text: str, timezone: str = "America/Chicago"
) -> AgentTurnResult:
    # Resolve the session id up front so every branch below (including the
    # demo-mode early return) can be tagged and grouped into the same
    # Langfuse Session - this is a multi-turn conversation app, so session_id
    # is the right thing to group traces by (see app.session_store).
    session_id = session_id or str(uuid.uuid4())

    with propagate_attributes(session_id=session_id, trace_name="run_agent_turn"):
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

    tools, mcp_active = _build_tools()

    messages = session_store.get_session_messages(session_id)
    if messages is None:
        messages = [{"role": "system", "content": _system_prompt(timezone, mcp_active)}]

    messages.append({"role": "user", "content": input_text})

    def _finish(result: AgentTurnResult) -> AgentTurnResult:
        session_store.save_session_messages(session_id, messages)
        return result

    # Snapshot (and clear) any pending delete-confirmation for this session.
    # Rule 8: if this turn doesn't re-request the exact same action, the
    # stale pending state is gone - it never lingers past one follow-up turn.
    pending_before = session_store.pop_pending_confirmation(session_id)

    actions: list[dict] = []
    seen_calls: set[tuple] = set()
    last_tool_name: str | None = None
    last_record: dict | None = None

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response_message = call_llm_with_tools(messages, tools)
        except LLMUnavailableError:
            return _finish(AgentTurnResult(
                session_id=session_id,
                done=bool(actions),
                message="Agent mode requires OPENAI_API_KEY to be set.",
                stored_record=last_record,
                tool_called=last_tool_name,
                actions=actions,
            ))

        tool_calls = getattr(response_message, "tool_calls", None)
        if not tool_calls:
            assistant_text = response_message.content or "Could you clarify what you'd like me to do?"
            messages.append({"role": "assistant", "content": assistant_text})
            return _finish(AgentTurnResult(
                session_id=session_id,
                done=bool(actions),
                message=assistant_text,
                stored_record=last_record,
                tool_called=last_tool_name,
                actions=actions,
            ))

        call = tool_calls[0]
        tool_name = call.function.name
        raw_args = call.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}

        # Rule 9: same (tool, args) repeating this turn means the model is
        # stuck - stop early rather than burning the rest of the cap.
        call_signature = (tool_name, json.dumps(args, sort_keys=True))
        if call_signature in seen_calls:
            final_text = _final_text(messages, tools, reason="repeat")
            return _finish(AgentTurnResult(
                session_id=session_id,
                done=bool(actions),
                message=final_text,
                stored_record=last_record,
                tool_called=last_tool_name,
                actions=actions,
            ))
        seen_calls.add(call_signature)

        handler = TOOL_DISPATCH.get(tool_name)
        is_known_local = handler is not None
        is_known_mcp = not is_known_local and mcp_client.is_mcp_tool(tool_name)

        if not is_known_local and not is_known_mcp:
            actions.append(
                {"tool": tool_name, "arguments": args, "status": "error", "result": None, "error": "unknown tool"}
            )
            return _finish(AgentTurnResult(
                session_id=session_id,
                done=False,
                message=f"Model tried to call unknown tool '{tool_name}'.",
                stored_record=None,
                tool_called=tool_name,
                actions=actions,
            ))

        # Rules 5-8: destructive tools always require an explicit human
        # confirmation turn before they actually execute.
        if _is_destructive_tool(tool_name):
            already_confirmed = (
                pending_before is not None
                and pending_before["tool"] == tool_name
                and pending_before["arguments"] == args
            )
            if not already_confirmed:
                session_store.set_pending_confirmation(session_id, tool_name, args)
                messages.append(_assistant_tool_call_message(call, tool_name, raw_args))
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"status": "awaiting_confirmation"})}
                )
                confirmation_text = _final_text(messages, tools, reason="confirm")
                return _finish(AgentTurnResult(
                    session_id=session_id,
                    done=False,
                    message=confirmation_text,
                    stored_record=None,
                    tool_called=tool_name,
                    actions=actions,
                ))
            # Already confirmed on a prior turn - fall through and execute.

        if is_known_local:
            record = _execute_local_tool(tool_name, args, input_text)
        else:
            record = mcp_client.call_mcp_tool(tool_name, args)

        status, error = _record_status(record)

        messages.append(_assistant_tool_call_message(call, tool_name, raw_args))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(record)})

        actions.append({"tool": tool_name, "arguments": args, "status": status, "result": record, "error": error})
        last_tool_name, last_record = tool_name, record

        # Rule 7 (revised): only *proposing* an unconfirmed destructive call
        # pauses the turn (handled above). Once confirmed and executed, it
        # falls through here like any other tool call - the loop continues
        # so the model can pick back up the rest of a multi-part request
        # ("...and also remind me to call back next week") in this same
        # turn, instead of requiring a manual follow-up nudge.

    # Rule 10: cap hit - force one final summary of everything done.
    final_text = _final_text(messages, tools, reason="cap")
    return _finish(AgentTurnResult(
        session_id=session_id,
        done=True,
        message=final_text,
        stored_record=last_record,
        tool_called=last_tool_name,
        actions=actions,
    ))
