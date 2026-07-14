"""Unit tests for the LLM gateway (routing, caching, fallback) and the
per-session budget/rate guardrails. LiteLLM is mocked - no network."""

from types import SimpleNamespace

import pytest

from app import budget, config, db, llm_gateway


@pytest.fixture(autouse=True)
def temp_db_and_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "database_path", str(tmp_path / "gb.db"))
    db.init_db()
    llm_gateway._CACHE.clear()
    budget._call_times.clear()
    yield


def _fake_response(content="hi", pt=10, ct=5):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )


# ---- routing ----

def test_routing_sends_synthesis_and_toolless_calls_local(monkeypatch):
    monkeypatch.setattr(config.settings, "model_routing_enabled", True)
    monkeypatch.setattr(config.settings, "local_model", "ollama/llama3.1:8b")
    monkeypatch.setattr(config.settings, "cloud_model", "gpt-4o-mini")
    # tool_choice="none" -> local
    assert llm_gateway.choose_model([], [{"x": 1}], "none") == "ollama/llama3.1:8b"
    # no tools -> local
    assert llm_gateway.choose_model([], [], "auto") == "ollama/llama3.1:8b"
    # real tool-calling -> cloud
    assert llm_gateway.choose_model([], [{"x": 1}], "auto") == "gpt-4o-mini"


def test_routing_disabled_always_cloud(monkeypatch):
    monkeypatch.setattr(config.settings, "model_routing_enabled", False)
    monkeypatch.setattr(config.settings, "cloud_model", "gpt-4o-mini")
    assert llm_gateway.choose_model([], [], "none") == "gpt-4o-mini"


# ---- caching ----

def test_second_identical_call_is_a_cache_hit(monkeypatch):
    monkeypatch.setattr(config.settings, "llm_cache_enabled", True)
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        return _fake_response()

    monkeypatch.setattr(llm_gateway.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm_gateway, "_cost_of", lambda r: 0.001)

    msgs = [{"role": "user", "content": "same"}]
    first = llm_gateway.complete(msgs, [], "none")
    second = llm_gateway.complete(msgs, [], "none")

    assert calls["n"] == 1  # provider hit only once
    assert first.cache_hit is False
    assert second.cache_hit is True


# ---- fallback ----

def test_local_failure_falls_back_to_cloud(monkeypatch):
    monkeypatch.setattr(config.settings, "model_routing_enabled", True)
    monkeypatch.setattr(config.settings, "llm_cache_enabled", False)
    monkeypatch.setattr(config.settings, "local_model", "ollama/llama3.1:8b")
    monkeypatch.setattr(config.settings, "cloud_model", "gpt-4o-mini")
    monkeypatch.setattr(llm_gateway, "_cost_of", lambda r: 0.0)

    seen_models = []

    def fake_completion(**kwargs):
        seen_models.append(kwargs["model"])
        if kwargs["model"].startswith("ollama/"):
            raise RuntimeError("ollama down")
        return _fake_response()

    monkeypatch.setattr(llm_gateway.litellm, "completion", fake_completion)

    # tool_choice="none" routes local first, then must fall back to cloud
    result = llm_gateway.complete([{"role": "user", "content": "x"}], [], "none")
    assert seen_models == ["ollama/llama3.1:8b", "gpt-4o-mini"]
    assert result.fell_back is True
    assert result.model == "gpt-4o-mini"


def test_cloud_failure_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(config.settings, "model_routing_enabled", False)  # -> cloud
    monkeypatch.setattr(config.settings, "llm_cache_enabled", False)

    def fake_completion(**kwargs):
        raise RuntimeError("cloud down")

    monkeypatch.setattr(llm_gateway.litellm, "completion", fake_completion)
    with pytest.raises(RuntimeError):
        llm_gateway.complete([{"role": "user", "content": "x"}], [], "auto")


# ---- budget + rate ----

def test_budget_cap_raises_once_exceeded(monkeypatch):
    monkeypatch.setattr(config.settings, "session_cost_cap_usd", 0.10)
    budget.add_usage("s", prompt_tokens=100, completion_tokens=50, cost_usd=0.05)
    budget.check_budget("s")  # under cap, fine
    budget.add_usage("s", prompt_tokens=100, completion_tokens=50, cost_usd=0.06)  # now 0.11
    with pytest.raises(budget.BudgetExceededError):
        budget.check_budget("s")


def test_rate_limit_raises_after_threshold(monkeypatch):
    monkeypatch.setattr(config.settings, "session_rate_limit_per_min", 3)
    for _ in range(3):
        budget.check_rate("s")
    with pytest.raises(budget.BudgetExceededError):
        budget.check_rate("s")


def test_usage_accumulates(monkeypatch):
    budget.add_usage("s", prompt_tokens=10, completion_tokens=5, cost_usd=0.01)
    budget.add_usage("s", prompt_tokens=20, completion_tokens=7, cost_usd=0.02)
    usage = budget.get_usage("s")
    assert usage["prompt_tokens"] == 30
    assert usage["completion_tokens"] == 12
    assert round(usage["cost_usd"], 4) == 0.03
    assert usage["call_count"] == 2
