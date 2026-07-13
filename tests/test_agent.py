import json
from types import SimpleNamespace

import pytest

from app import agent, config, db, session_store


@pytest.fixture(autouse=True)
def temp_database_and_key(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lifeops_agent.db"
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


def test_one_shot_tool_call_stores_reminder(monkeypatch):
    responses = [
        _fake_message(
            tool_calls=[
                _fake_tool_call(
                    "call_1",
                    "create_reminder",
                    {"title": "Call mom", "due_date": "2026-07-09T17:00:00", "priority": "medium"},
                )
            ]
        ),
        _fake_message(content="Got it - I'll remind you to call mom tomorrow at 5pm."),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    result = agent.run_agent_turn(session_id=None, input_text="remind me to call mom tomorrow at 5pm")

    assert result.done is True
    assert result.tool_called == "create_reminder"
    assert result.stored_record["title"] == "Call mom"
    assert "call mom" in result.message.lower()
    assert session_store.get_session_messages(result.session_id) is not None


def test_clarification_then_followup_completes(monkeypatch):
    responses = [
        _fake_message(content="What date/time should I set for this reminder?"),
        _fake_message(
            tool_calls=[
                _fake_tool_call(
                    "call_2",
                    "create_reminder",
                    {"title": "Call mom", "due_date": "2026-07-09T17:00:00"},
                )
            ]
        ),
        _fake_message(content="Done - reminder set for tomorrow at 5pm."),
    ]
    call_log = []

    def fake_call(messages, tools, tool_choice="auto"):
        call_log.append(len(messages))
        return responses.pop(0)

    monkeypatch.setattr(agent, "call_llm_with_tools", fake_call)

    first = agent.run_agent_turn(session_id=None, input_text="remind me to call mom")
    assert first.done is False
    assert first.stored_record is None
    assert "?" in first.message

    second = agent.run_agent_turn(session_id=first.session_id, input_text="tomorrow at 5pm")
    assert second.done is True
    assert second.session_id == first.session_id
    assert second.stored_record["title"] == "Call mom"
    # second turn made 2 calls (decision + synthesis), both saw more history than the first turn
    assert call_log[1] > call_log[0]
    assert call_log[2] > call_log[1]


def test_unknown_tool_name_reported_gracefully(monkeypatch):
    fake_response = _fake_message(
        tool_calls=[_fake_tool_call("call_3", "delete_everything", {})]
    )
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": fake_response
    )

    result = agent.run_agent_turn(session_id=None, input_text="do something weird")

    assert result.done is False
    assert result.stored_record is None
    assert "delete_everything" in result.message


def test_demo_mode_returns_helpful_message(monkeypatch):
    monkeypatch.setattr(config.settings, "openai_api_key", None)

    result = agent.run_agent_turn(session_id=None, input_text="remind me to call mom")

    assert result.done is False
    assert "OPENAI_API_KEY" in result.message


def test_complete_reminder_dispatch_marks_existing_reminder_done(monkeypatch):
    from app.schemas import ParsedIntent
    from app.tools.reminders import create_reminder

    seeded = create_reminder(
        ParsedIntent(intent_type="reminder", title="Submit tax forms", raw_text="tax forms")
    )

    responses = [
        _fake_message(tool_calls=[_fake_tool_call("call_4", "complete_reminder", {"id": seeded["id"]})]),
        _fake_message(content="Marked the tax forms reminder as done."),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    result = agent.run_agent_turn(session_id=None, input_text="I already filed taxes, mark that done")

    assert result.done is True
    assert result.tool_called == "complete_reminder"
    assert result.stored_record["completed"] is True


def test_delete_requires_confirmation_then_executes_on_next_turn(monkeypatch):
    from app.schemas import ParsedIntent
    from app.tools.calendar_mock import create_calendar_event, list_calendar_events

    seeded = create_calendar_event(
        ParsedIntent(intent_type="calendar_event", title="Old meeting", raw_text="old meeting")
    )
    delete_args = {"id": seeded["id"]}

    responses = [
        _fake_message(tool_calls=[_fake_tool_call("call_5a", "delete_calendar_event", delete_args)]),
        _fake_message(content="Are you sure you want to delete 'Old meeting'?"),
        _fake_message(
            tool_calls=[_fake_tool_call("call_5b", agent.CONFIRM_TOOL_NAME, {"confirmed": True})]
        ),
        _fake_message(content="Deleted the old meeting."),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    first = agent.run_agent_turn(session_id=None, input_text="delete the old meeting")
    assert first.done is False
    assert first.stored_record is None
    assert list_calendar_events() != []  # not deleted yet
    assert session_store.get_pending_confirmation(first.session_id) == {
        "tool": "delete_calendar_event",
        "arguments": delete_args,
    }

    second = agent.run_agent_turn(session_id=first.session_id, input_text="yes, I'm sure")
    assert second.done is True
    assert second.tool_called == "delete_calendar_event"
    assert second.stored_record["deleted"] is True
    assert list_calendar_events() == []
    assert session_store.get_pending_confirmation(first.session_id) is None


def test_confirmed_delete_chains_into_a_followup_action_same_turn(monkeypatch):
    """Regression test for a real bug: 'delete my dentist reminder and also
    remind me to call back next week' used to drop the second half of the
    request once the delete paused for confirmation. A confirmed delete
    should now continue the loop instead of force-ending the turn."""
    from app.schemas import ParsedIntent
    from app.tools.reminders import create_reminder, list_reminders

    seeded = create_reminder(
        ParsedIntent(intent_type="reminder", title="Call the dentist", raw_text="call the dentist")
    )
    delete_args = {"id": seeded["id"]}

    responses = [
        _fake_message(tool_calls=[_fake_tool_call("call_10a", "delete_reminder", delete_args)]),
        _fake_message(content="Are you sure you want to delete that reminder?"),
        # Confirmation turn: the confirm tool replays the stored delete, then
        # the model keeps going and creates the follow-up reminder in the
        # SAME turn.
        _fake_message(
            tool_calls=[_fake_tool_call("call_10b", agent.CONFIRM_TOOL_NAME, {"confirmed": True})]
        ),
        _fake_message(
            tool_calls=[
                _fake_tool_call(
                    "call_10c",
                    "create_reminder",
                    {"title": "Call dentist back", "due_date": "2026-07-17T10:00:00"},
                )
            ]
        ),
        _fake_message(content="Deleted the old reminder and set a new one for next week."),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    first = agent.run_agent_turn(
        session_id=None, input_text="delete my dentist reminder and also remind me to call back next week"
    )
    assert first.done is False

    second = agent.run_agent_turn(session_id=first.session_id, input_text="yes")
    assert second.done is True
    assert len(second.actions) == 2
    assert [a.tool for a in second.actions] == ["delete_reminder", "create_reminder"]
    remaining = list_reminders(include_completed=True)
    assert any(r["title"] == "Call dentist back" for r in remaining)


def test_declining_a_pending_delete_clears_it_without_executing(monkeypatch):
    from app.schemas import ParsedIntent
    from app.tools.calendar_mock import create_calendar_event, list_calendar_events

    seeded = create_calendar_event(
        ParsedIntent(intent_type="calendar_event", title="Old meeting", raw_text="old meeting")
    )
    delete_args = {"id": seeded["id"]}

    responses = [
        _fake_message(tool_calls=[_fake_tool_call("call_6a", "delete_calendar_event", delete_args)]),
        _fake_message(content="Are you sure?"),
        _fake_message(
            tool_calls=[_fake_tool_call("call_6b", agent.CONFIRM_TOOL_NAME, {"confirmed": False})]
        ),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    first = agent.run_agent_turn(session_id=None, input_text="delete the old meeting")
    assert first.done is False

    second = agent.run_agent_turn(session_id=first.session_id, input_text="no, never mind")
    assert second.done is False
    assert second.message == agent._DECLINE_MESSAGE
    assert list_calendar_events() != []
    assert session_store.get_pending_confirmation(first.session_id) is None


def test_multi_step_chain_completes_in_one_turn(monkeypatch):
    responses = [
        _fake_message(
            tool_calls=[_fake_tool_call("call_7a", "list_reminders", {"include_completed": False})]
        ),
        _fake_message(
            tool_calls=[
                _fake_tool_call(
                    "call_7b",
                    "create_reminder",
                    {"title": "Buy milk", "due_date": "2026-07-10T09:00:00"},
                )
            ]
        ),
        _fake_message(
            tool_calls=[
                _fake_tool_call(
                    "call_7c",
                    "create_calendar_event",
                    {"title": "Celebrate", "start_time": "2026-07-10T18:00:00"},
                )
            ]
        ),
        _fake_message(content="Done - checked your reminders, added 'Buy milk', and scheduled 'Celebrate'."),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    result = agent.run_agent_turn(
        session_id=None, input_text="check my reminders, add buy milk, and schedule a celebration"
    )

    assert result.done is True
    assert len(result.actions) == 3
    assert [a.tool for a in result.actions] == ["list_reminders", "create_reminder", "create_calendar_event"]
    assert all(a.status == "success" for a in result.actions)


def test_cap_hit_forces_summary_and_stops(monkeypatch):
    # 5 distinct non-destructive tool calls, then the loop should force a
    # summary instead of asking the model a 6th time.
    responses = [
        _fake_message(
            tool_calls=[
                _fake_tool_call(f"call_8{i}", "create_journal_entry", {"content": f"entry {i}", "title": None})
            ]
        )
        for i in range(agent.MAX_TOOL_ITERATIONS)
    ] + [_fake_message(content="I made 5 journal entries and stopped at the limit.")]

    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    result = agent.run_agent_turn(session_id=None, input_text="log 5 different journal entries")

    assert result.done is True
    assert len(result.actions) == agent.MAX_TOOL_ITERATIONS
    assert "5" in result.message or "limit" in result.message.lower()


def test_repeated_identical_call_stops_early(monkeypatch):
    same_call = _fake_message(
        tool_calls=[_fake_tool_call("call_9", "list_reminders", {"include_completed": False})]
    )
    # Model keeps requesting the exact same call - should stop after the 2nd
    # attempt (the repeat) instead of consuming the full cap.
    responses = [same_call, same_call, same_call, same_call, same_call]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses[0]
    )

    result = agent.run_agent_turn(session_id=None, input_text="list my reminders")

    assert len(result.actions) == 1  # only the first attempt actually executed


def test_list_reminders_dispatch_returns_wrapped_dict(monkeypatch):
    fake_response = _fake_message(
        tool_calls=[_fake_tool_call("call_6", "list_reminders", {"include_completed": False})]
    )
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": fake_response
    )

    result = agent.run_agent_turn(session_id=None, input_text="what reminders do I have?")

    assert result.tool_called == "list_reminders"
    assert result.stored_record == {"reminders": []}


def test_batch_response_executes_all_calls_in_one_round_trip(monkeypatch):
    """The model can legitimately return multiple tool calls in a single
    response (confirmed live via real trace data) - both should execute
    without needing a separate LLM round-trip for each."""
    call_log = []
    responses = [
        _fake_message(
            tool_calls=[
                _fake_tool_call(
                    "call_11a", "create_reminder", {"title": "Buy milk", "due_date": "2026-07-10T09:00:00"}
                ),
                _fake_tool_call(
                    "call_11b", "create_calendar_event", {"title": "Celebrate", "start_time": "2026-07-10T18:00:00"}
                ),
            ]
        ),
        _fake_message(content="Added the milk reminder and scheduled the celebration."),
    ]

    def fake_call(messages, tools, tool_choice="auto"):
        call_log.append(1)
        return responses.pop(0)

    monkeypatch.setattr(agent, "call_llm_with_tools", fake_call)

    result = agent.run_agent_turn(session_id=None, input_text="add a reminder and schedule an event")

    assert result.done is True
    assert len(result.actions) == 2
    assert [a.tool for a in result.actions] == ["create_reminder", "create_calendar_event"]
    assert all(a.status == "success" for a in result.actions)
    # Only 2 LLM round-trips total (decision batch + final answer), not 3 -
    # this is the actual latency win: both actions came from one response.
    assert len(call_log) == 2


def test_cap_reached_mid_batch_stops_remaining_calls_safely(monkeypatch):
    """A single response with more tool calls than the remaining cap budget
    should execute up to the cap and cleanly stop - without leaving the
    message history API-invalid (every tool_call_id must get a result)."""
    six_calls = [
        _fake_tool_call(f"call_12{i}", "create_journal_entry", {"content": f"entry {i}", "title": None})
        for i in range(6)
    ]
    responses = [
        _fake_message(tool_calls=six_calls),
        _fake_message(content="I made 5 journal entries and stopped at the limit."),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    result = agent.run_agent_turn(session_id=None, input_text="log 6 different journal entries at once")

    assert result.done is True
    assert len(result.actions) == agent.MAX_TOOL_ITERATIONS

    # Every one of the 6 original tool_call_ids must have a matching "tool"
    # result message (5 real + 1 synthetic "not_executed"), or the next
    # turn in this session would send an API-invalid message history.
    messages = session_store.get_session_messages(result.session_id)
    tool_result_ids = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
    assert tool_result_ids == {c.id for c in six_calls}


def test_destructive_call_mid_batch_pauses_and_stops_rest_of_batch(monkeypatch):
    from app.schemas import ParsedIntent
    from app.tools.reminders import create_reminder as create_reminder_tool

    seeded = create_reminder_tool(
        ParsedIntent(intent_type="reminder", title="Old reminder", raw_text="old reminder")
    )

    responses = [
        _fake_message(
            tool_calls=[
                _fake_tool_call(
                    "call_13a", "create_reminder", {"title": "New reminder", "due_date": "2026-07-10T09:00:00"}
                ),
                _fake_tool_call("call_13b", "delete_reminder", {"id": seeded["id"]}),
                _fake_tool_call("call_13c", "list_reminders", {"include_completed": False}),
            ]
        ),
        _fake_message(content="I added the new reminder. Should I go ahead and delete the old one?"),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    result = agent.run_agent_turn(session_id=None, input_text="add a reminder and delete the old one")

    assert result.done is False
    # Only the safe call before the destructive one executed - list_reminders
    # (queued after the delete in the batch) never ran.
    assert len(result.actions) == 1
    assert result.actions[0].tool == "create_reminder"
    assert session_store.get_pending_confirmation(result.session_id) == {
        "tool": "delete_reminder",
        "arguments": {"id": seeded["id"]},
    }


def test_repeat_detected_within_a_single_batch_stops_early(monkeypatch):
    same_call_twice = [
        _fake_tool_call("call_14a", "list_reminders", {"include_completed": False}),
        _fake_tool_call("call_14b", "list_reminders", {"include_completed": False}),
    ]
    responses = [
        _fake_message(tool_calls=same_call_twice),
        _fake_message(content="Here are your reminders."),
    ]
    monkeypatch.setattr(
        agent, "call_llm_with_tools", lambda messages, tools, tool_choice="auto": responses.pop(0)
    )

    result = agent.run_agent_turn(session_id=None, input_text="list my reminders")

    assert len(result.actions) == 1  # the duplicate within the same batch never executed


def test_trim_history_keeps_system_message_and_bounds_length():
    system = {"role": "system", "content": "sys"}
    # 50 user/assistant turns (100 messages) - well over the 40-message cap.
    rest = []
    for i in range(50):
        rest.append({"role": "user", "content": f"msg {i}"})
        rest.append({"role": "assistant", "content": f"reply {i}"})
    messages = [system] + rest

    trimmed = agent._trim_history(messages)

    assert trimmed[0] == system
    assert len(trimmed) <= agent.MAX_HISTORY_MESSAGES + 1
    assert trimmed[1]["role"] == "user"  # always starts at a user-message boundary


def test_trim_history_never_orphans_a_tool_result_message():
    system = {"role": "system", "content": "sys"}
    rest = []
    for i in range(50):
        rest.append({"role": "user", "content": f"msg {i}"})
        rest.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        })
        rest.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "{}"})
    messages = [system] + rest

    trimmed = agent._trim_history(messages)

    # Every "tool" message's referenced call_id must have its assistant
    # tool_calls message present in the trimmed history too.
    declared_ids = set()
    for m in trimmed:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared_ids.update(tc["id"] for tc in m["tool_calls"])
    referenced_ids = {m["tool_call_id"] for m in trimmed if m.get("role") == "tool"}
    assert referenced_ids.issubset(declared_ids)


def test_trim_history_leaves_short_conversations_untouched():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert agent._trim_history(messages) == messages
