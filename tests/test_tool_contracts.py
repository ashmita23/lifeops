"""Contract tests for the seam between a local tool's advertised JSON schema
and the handler that actually runs it.

Unit tests elsewhere mock the LLM, so nothing there catches a schema that
disagrees with its handler - e.g. a handler doing args["id"] while the schema
forgets to mark "id" required, which passes CI and then KeyErrors at runtime
the first time the model calls it. These tests check that seam directly.
"""

import pytest

from app import agent, config, db


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_tool_contracts.db"
    monkeypatch.setattr(config.settings, "database_path", str(db_path))
    db.init_db()
    yield


def _local_tool_schemas() -> dict:
    return {t["function"]["name"]: t["function"] for t in agent._LOCAL_TOOLS}


def test_every_declared_tool_has_a_handler_and_vice_versa():
    declared = set(_local_tool_schemas())
    dispatched = set(agent.TOOL_DISPATCH)
    assert declared == dispatched, (
        f"schema/handler mismatch - only declared: {declared - dispatched}, "
        f"only dispatched: {dispatched - declared}"
    )


def test_schemas_satisfy_strict_mode_invariants():
    """OpenAI strict structured-outputs mode requires every property to be
    listed in `required` and additionalProperties=False. A violation is
    rejected by the API at call time, so catch it here instead."""
    for name, fn in _local_tool_schemas().items():
        params = fn["parameters"]
        properties = set(params.get("properties", {}))
        required = set(params.get("required", []))
        assert properties == required, f"{name}: every property must be required in strict mode; missing {properties - required}"
        assert params.get("additionalProperties") is False, f"{name}: additionalProperties must be False in strict mode"


def _dummy_for(name: str, spec: dict):
    """A minimal valid value for a schema property, by type (and name, so
    date/time fields get a parseable ISO string rather than junk)."""
    types = spec.get("type", "string")
    types = types if isinstance(types, list) else [types]
    primary = next((t for t in types if t != "null"), "string")

    if "enum" in spec:
        return next((v for v in spec["enum"] if v is not None), None)
    if primary == "integer":
        return 1
    if primary == "number":
        return 1.0
    if primary == "boolean":
        return True
    if primary == "object":
        return {}
    if "date" in name or "time" in name:
        return "2026-07-14T12:00:00"
    return "test value"


def test_handlers_accept_exactly_their_required_args():
    """Build the minimal argument set the schema PROMISES (its required
    fields) and run the handler. A KeyError/TypeError here means the handler
    reads something the schema doesn't guarantee - i.e. the contract is
    broken. Data-level errors (e.g. a missing row) are fine; we only care
    that the argument shape is honored."""
    schemas = _local_tool_schemas()
    for name, handler in agent.TOOL_DISPATCH.items():
        params = schemas[name]["parameters"]
        props = params.get("properties", {})
        args = {field: _dummy_for(field, props[field]) for field in params.get("required", [])}

        try:
            result = handler(args, "test raw text")
        except (KeyError, TypeError) as exc:
            pytest.fail(f"{name}: handler reads an arg its schema doesn't require -> {exc!r}")
        except Exception:
            # Data-level failures (bad id, parse issues on dummy data) are not
            # contract violations - the argument shape was still honored.
            continue

        assert isinstance(result, dict), f"{name}: handler must return a dict, got {type(result)}"
