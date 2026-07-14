import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import frontend.gradio_app as ga
from app import config, db


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    # The handler now reads per-session usage for the observability footer,
    # which queries the session_usage table - so the DB must exist.
    monkeypatch.setattr(config.settings, "database_path", str(tmp_path / "gradio.db"))
    db.init_db()
    yield


def _fake_result(message="ok", session_id="sess-1", trace_id="trace-1"):
    return SimpleNamespace(message=message, session_id=session_id, trace_id=trace_id, actions=[])


def test_slow_trace_export_does_not_delay_chat_reply(monkeypatch):
    monkeypatch.setattr(ga, "run_agent_turn", lambda **kwargs: _fake_result())

    def slow_export(trace_id):
        time.sleep(2)
        return "logs/traces/trace-1.json"

    monkeypatch.setattr(ga, "export_trace", slow_export)

    test_executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(ga, "_EXPORT_EXECUTOR", test_executor)

    start = time.perf_counter()
    history, session_id = ga.handle_agent_submit({"text": "hello", "files": []}, "", [])
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0  # must not block on the 2s "export"
    assert session_id == "sess-1"
    assert history[-1]["content"].startswith("ok")  # reply + observability footer

    test_executor.shutdown(wait=True)  # let the background task finish before moving on


def test_background_export_exception_is_logged_not_raised(monkeypatch, caplog):
    def failing_export(trace_id):
        raise RuntimeError("export exploded")

    monkeypatch.setattr(ga, "export_trace", failing_export)

    with caplog.at_level("ERROR", logger="frontend.gradio_app"):
        ga._export_trace_background("trace-x")  # must not raise

    assert "Background trace export failed" in caplog.text


def test_handler_logs_latency_breakdown(monkeypatch, caplog):
    monkeypatch.setattr(ga, "run_agent_turn", lambda **kwargs: _fake_result())
    monkeypatch.setattr(ga, "export_trace", lambda trace_id: None)

    test_executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(ga, "_EXPORT_EXECUTOR", test_executor)

    with caplog.at_level("INFO", logger="frontend.gradio_app"):
        ga.handle_agent_submit({"text": "hello", "files": []}, "", [])

    assert "chat turn" in caplog.text
    assert "agent=" in caplog.text
    assert "total=" in caplog.text
    assert "cost=" in caplog.text  # observability surface includes cost now

    test_executor.shutdown(wait=True)
