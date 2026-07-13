import json
from types import SimpleNamespace

import pytest

from app import agent, config, db


@pytest.fixture(autouse=True)
def temp_database_and_key(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lifeops_mcp.db"
    monkeypatch.setattr(config.settings, "database_path", str(db_path))
    monkeypatch.setattr(config.settings, "openai_api_key", "sk-test-key")
    db.init_db()
    yield


def _fake_tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _fake_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def test_build_tools_excludes_local_calendar_when_mcp_active(monkeypatch):
    fake_mcp_tools = [
        {
            "type": "function",
            "function": {"name": "list-events", "description": "List Google Calendar events", "parameters": {}},
        }
    ]
    monkeypatch.setattr(agent.mcp_client, "get_mcp_tools", lambda: fake_mcp_tools)

    tools, mcp_active = agent._build_tools()

    assert mcp_active is True
    tool_names = {t["function"]["name"] for t in tools}
    assert "list-events" in tool_names
    assert agent._LOCAL_CALENDAR_TOOL_NAMES.isdisjoint(tool_names)
    assert "create_reminder" in tool_names  # non-calendar local tools stay


def test_build_tools_keeps_local_calendar_when_mcp_inactive(monkeypatch):
    monkeypatch.setattr(agent.mcp_client, "get_mcp_tools", lambda: [])

    tools, mcp_active = agent._build_tools()

    assert mcp_active is False
    tool_names = {t["function"]["name"] for t in tools}
    assert agent._LOCAL_CALENDAR_TOOL_NAMES.issubset(tool_names)


def test_agent_routes_mcp_tool_call_through_mcp_client(monkeypatch):
    fake_mcp_tools = [
        {
            "type": "function",
            "function": {"name": "create-event", "description": "Create a Google Calendar event", "parameters": {}},
        }
    ]
    monkeypatch.setattr(agent.mcp_client, "get_mcp_tools", lambda: fake_mcp_tools)
    monkeypatch.setattr(agent.mcp_client, "is_mcp_tool", lambda name: name == "create-event")
    monkeypatch.setattr(
        agent.mcp_client,
        "call_mcp_tool",
        lambda name, args: {"result": [f"Created event: {args.get('summary')}"], "is_error": False},
    )

    responses = [
        _fake_message(
            tool_calls=[_fake_tool_call("call_1", "create-event", {"summary": "Dentist"})]
        ),
        _fake_message(content="Added Dentist to your calendar."),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    result = agent.run_agent_turn(session_id=None, input_text="schedule a dentist appointment tomorrow at 2pm")

    assert result.done is True
    assert result.tool_called == "create-event"
    assert result.stored_record["is_error"] is False
    assert "Dentist" in result.stored_record["result"][0]


def test_transcribe_audio_demo_mode_guard(monkeypatch):
    from app.llm_client import LLMUnavailableError
    from app.transcription import transcribe_audio

    monkeypatch.setattr(config.settings, "openai_api_key", None)

    with pytest.raises(LLMUnavailableError):
        transcribe_audio("does-not-matter.wav")
