import pytest

from app import config, db
from app.schemas import ParsedIntent
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


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lifeops_crud.db"
    monkeypatch.setattr(config.settings, "database_path", str(db_path))
    db.init_db()
    yield


def _make_reminder(title="Call mom", due_date="2026-07-09T17:00:00"):
    return create_reminder(
        ParsedIntent(intent_type="reminder", title=title, due_date=due_date, raw_text=title)
    )


def _make_event(title="Team sync", start_time="2026-07-10T14:00:00"):
    return create_calendar_event(
        ParsedIntent(intent_type="calendar_event", title=title, start_time=start_time, raw_text=title)
    )


def _make_journal_entry(text="Felt productive today"):
    return create_journal_entry(ParsedIntent(intent_type="journal_entry", description=text, raw_text=text))


# --- reminders ---------------------------------------------------------

def test_list_reminders_excludes_completed_by_default():
    r1 = _make_reminder(title="A")
    r2 = _make_reminder(title="B")
    complete_reminder(r2["id"])

    active = list_reminders()
    assert [r["id"] for r in active] == [r1["id"]]

    all_reminders = list_reminders(include_completed=True)
    assert {r["id"] for r in all_reminders} == {r1["id"], r2["id"]}


def test_update_reminder_changes_only_given_fields():
    record = _make_reminder(title="Original", due_date="2026-07-09T17:00:00")

    updated = update_reminder(record["id"], priority="high")

    assert updated["title"] == "Original"
    assert updated["priority"] == "high"


def test_complete_reminder_marks_completed():
    record = _make_reminder()

    completed = complete_reminder(record["id"])

    assert completed["completed"] is True


def test_delete_reminder_removes_row():
    record = _make_reminder()

    assert delete_reminder(record["id"]) is True
    assert delete_reminder(record["id"]) is False
    assert list_reminders(include_completed=True) == []


def test_update_and_delete_reminder_missing_id_are_safe():
    assert update_reminder(9999, title="x") is None
    assert complete_reminder(9999) is None
    assert delete_reminder(9999) is False


# --- calendar events -----------------------------------------------------

def test_list_update_delete_calendar_event():
    record = _make_event()

    events = list_calendar_events()
    assert len(events) == 1

    updated = update_calendar_event(record["id"], start_time="2026-07-11T09:00:00")
    assert updated["start_time"] == "2026-07-11T09:00:00"
    assert updated["title"] == record["title"]

    assert delete_calendar_event(record["id"]) is True
    assert list_calendar_events() == []


# --- journal entries -----------------------------------------------------

def test_list_and_delete_journal_entry():
    record = _make_journal_entry()

    entries = list_journal_entries()
    assert len(entries) == 1

    assert delete_journal_entry(record["id"]) is True
    assert list_journal_entries() == []
