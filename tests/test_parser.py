import pytest

from app import config
from app.parser import parse_user_command


@pytest.fixture(autouse=True)
def force_demo_mode(monkeypatch):
    # These tests exercise the offline regex fallback specifically, regardless
    # of whether a real OPENAI_API_KEY happens to be set in .env.
    monkeypatch.setattr(config.settings, "openai_api_key", None)


def test_reminder_intent_with_clarification_when_no_time():
    intent = parse_user_command("remind me to call mom")
    assert intent.intent_type == "reminder"
    assert intent.needs_clarification is True


def test_reminder_intent_with_time_resolved():
    intent = parse_user_command("remind me to call mom tomorrow at 5pm")
    assert intent.intent_type == "reminder"
    assert intent.needs_clarification is False
    assert intent.due_date is not None


def test_calendar_event_intent():
    intent = parse_user_command("schedule a meeting with Sam Friday at 2pm")
    assert intent.intent_type == "calendar_event"
    assert intent.start_time is not None


def test_journal_entry_intent():
    intent = parse_user_command("journal: I felt really productive today")
    assert intent.intent_type == "journal_entry"
    assert intent.needs_clarification is False


def test_unknown_intent():
    intent = parse_user_command("asdkfjaslkdfj random gibberish")
    assert intent.intent_type == "unknown"


def test_raw_text_preserved():
    text = "remind me to water the plants tomorrow morning"
    intent = parse_user_command(text)
    assert intent.raw_text == text
