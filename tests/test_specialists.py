"""Tests for the planner (free-slot finding) and booking (approval gate)
specialists."""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from app import agent, config, db, session_store
from app.specialists.booking import book_reservation
from app.specialists.planner import find_free_slots, plan_schedule


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "database_path", str(tmp_path / "spec.db"))
    monkeypatch.setattr(config.settings, "openai_api_key", "sk-test")
    db.init_db()
    yield


def _dt(h, m=0):
    return datetime(2026, 7, 14, h, m)


# ---- planner: pure slot-finder ----

def test_find_free_slots_avoids_busy_blocks_with_buffer():
    busy = [(_dt(9), _dt(10))]  # 9-10am busy
    slots = find_free_slots(
        busy, _dt(8), _dt(12), duration_minutes=60, buffer_minutes=15, preference="any", max_results=2
    )
    # 8:00 slot ends 9:00 but needs 15m buffer before the 9:00 block -> blocked.
    # First valid start is 10:15 (15m after the 10:00 end).
    assert slots[0] == _dt(10, 15)


def test_find_free_slots_respects_preference_window():
    slots = find_free_slots(
        [], _dt(8), _dt(20), duration_minutes=30, preference="afternoon", max_results=1
    )
    assert slots[0].hour >= 12  # afternoon only


def test_find_free_slots_returns_empty_when_no_room():
    busy = [(_dt(8), _dt(20))]  # whole window busy
    slots = find_free_slots(busy, _dt(8), _dt(20), duration_minutes=60, preference="any")
    assert slots == []


def test_plan_schedule_proposes_around_existing_event(monkeypatch):
    from app.schemas import ParsedIntent
    from app.specialists import planner
    from app.tools.calendar_mock import create_calendar_event

    # Force the local-calendar path: this test seeds a local event and asserts
    # the planner plans around it. (When a real Google Calendar is connected the
    # planner prefers get-freebusy; that path is covered by a live check.)
    monkeypatch.setattr(planner, "_busy_from_google", lambda day: None)

    create_calendar_event(
        ParsedIntent(
            intent_type="calendar_event", title="Standup", raw_text="standup",
            start_time="2026-07-14T09:00:00", end_time="2026-07-14T10:00:00",
        )
    )
    result = plan_schedule(
        {"title": "Deep work", "date": "2026-07-14", "duration_minutes": 60, "preference": "morning"},
        "find me an hour",
    )
    assert result["considered_events"] == 1
    assert result["proposed_slots"]  # found at least one morning slot
    # none of the proposals should collide with the 9-10 standup
    for iso in result["proposed_slots"]:
        assert not iso.startswith("2026-07-14T09")


# ---- booking: approval gate ----

def test_book_reservation_returns_confirmation():
    out = book_reservation(
        {"restaurant": "Alinea", "datetime": "2026-07-20T19:00:00", "party_size": 2, "guest_name": "Ashmita"},
        "book it",
    )
    assert out["status"] == "booked"
    assert out["confirmation_number"].startswith("LO-")


def test_book_reservation_pauses_for_approval(monkeypatch):
    """A proposed booking must pause (like a delete), not execute immediately."""
    booking_args = {"restaurant": "Alinea", "datetime": "2026-07-20T19:00:00", "party_size": 2, "guest_name": "Ashmita"}
    responses = [
        SimpleNamespace(content=None, tool_calls=[
            SimpleNamespace(id="c1", function=SimpleNamespace(name="book_reservation", arguments=json.dumps(booking_args)))
        ]),
        SimpleNamespace(content="This reservation needs your approval and ID (Ashmita). Confirm?", tool_calls=None),
    ]
    monkeypatch.setattr(agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0))

    result = agent.run_agent_turn(session_id=None, input_text="book Alinea for 2 at 7pm on the 20th under Ashmita")

    assert result.done is False  # paused, not executed
    assert session_store.get_pending_confirmation(result.session_id) == {
        "tool": "book_reservation", "arguments": booking_args
    }


def test_requires_approval_covers_booking_and_destructive():
    assert agent._requires_approval("book_reservation") is True
    assert agent._requires_approval("delete_reminder") is True
    assert agent._requires_approval("plan_schedule") is False
    assert agent._requires_approval("list_reminders") is False
