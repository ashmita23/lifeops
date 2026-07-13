import pytest

from app import config, db
from app.services import router


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lifeops.db"
    monkeypatch.setattr(config.settings, "database_path", str(db_path))
    # These tests exercise the offline regex fallback specifically, regardless
    # of whether a real OPENAI_API_KEY happens to be set in .env.
    monkeypatch.setattr(config.settings, "openai_api_key", None)
    db.init_db()
    yield


def test_reminder_flow_stores_record():
    response = router.process_command("remind me to call mom tomorrow at 5pm")
    assert response.success is True
    assert response.stored_record is not None
    assert response.stored_record["title"]


def test_reminder_missing_time_asks_for_clarification():
    response = router.process_command("remind me to call mom")
    assert response.success is False
    assert response.intent.needs_clarification is True
    assert response.stored_record is None


def test_calendar_event_flow_stores_record():
    response = router.process_command("schedule a meeting with Sam Friday at 2pm")
    assert response.success is True
    assert response.stored_record["start_time"] is not None


def test_journal_entry_flow_stores_record():
    response = router.process_command("journal: I felt really productive today")
    assert response.success is True
    assert response.stored_record["content"]


def test_unknown_intent_returns_helpful_failure():
    response = router.process_command("asdkfjaslkdfj random gibberish")
    assert response.success is False
    assert response.stored_record is None


def test_daily_summary_returns_structure():
    router.process_command("remind me to call mom tomorrow at 5pm")
    response = router.process_command("what do i have today")
    assert response.success is True
    assert "reminders" in response.stored_record
    assert "calendar_events" in response.stored_record
    assert "journal_entries" in response.stored_record
