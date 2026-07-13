import json
from types import SimpleNamespace

import pytest

from app import config, trace_export


class FakeResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def stub_langfuse_and_traces_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(config.settings, "langfuse_secret_key", "sk-test")
    monkeypatch.setattr(config.settings, "langfuse_host", "https://fake.langfuse.test")
    monkeypatch.setattr(trace_export, "get_client", lambda: SimpleNamespace(flush=lambda: None))
    # No real waiting in tests - the retry loop's sleep calls become no-ops.
    monkeypatch.setattr(trace_export.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(trace_export, "_TRACES_DIR", tmp_path / "traces")
    yield


def _queue(monkeypatch, responses):
    monkeypatch.setattr(trace_export.httpx, "get", lambda *a, **k: responses.pop(0))


def _root(obs_id="root", end_time="2026-01-01T00:00:05.000Z", marker="complete"):
    return {
        "id": obs_id,
        "type": "AGENT",
        "parentObservationId": None,
        "startTime": "2026-01-01T00:00:00.000Z",
        "endTime": end_time,
        "metadata": ({"lifeops_export_marker": marker} if marker else {}),
    }


def _generation(obs_id, end_time="2026-01-01T00:00:02.000Z", start_time="2026-01-01T00:00:01.000Z"):
    return {
        "id": obs_id,
        "type": "GENERATION",
        "parentObservationId": "root",
        "startTime": start_time,
        "endTime": end_time,
    }


def _trace(observations, latency=5.0):
    return FakeResponse(200, {"observations": observations, "latency": latency})


# A single "complete" snapshot, repeated 3x, is the minimum to satisfy the
# stability check (EXPORT_STABILITY_CHECKS = 3).
def _stable(observations, latency=5.0, times=3):
    return [_trace(observations, latency) for _ in range(times)]


def test_accepts_a_genuinely_complete_stable_trace(monkeypatch):
    complete = [_root(), _generation("g1")]
    _queue(monkeypatch, _stable(complete))

    path = trace_export.export_trace("trace-ok")

    assert path is not None
    saved = json.loads(open(path).read())
    assert len(saved["observations"]) == 2


def test_missing_root_agent_span_is_incomplete(monkeypatch):
    # Only a generation, no root AGENT observation at all.
    monkeypatch.setattr(
        trace_export.httpx, "get", lambda *a, **k: _trace([_generation("g1")], latency=2.0)
    )

    path = trace_export.export_trace("trace-no-root")

    assert path is None


def test_root_without_end_time_is_incomplete(monkeypatch):
    unfinished_root = _root(end_time=None)
    monkeypatch.setattr(
        trace_export.httpx,
        "get",
        lambda *a, **k: _trace([unfinished_root, _generation("g1")], latency=2.0),
    )

    path = trace_export.export_trace("trace-no-end")

    assert path is None


def test_missing_completion_marker_is_incomplete(monkeypatch):
    root_no_marker = _root(marker=None)
    monkeypatch.setattr(
        trace_export.httpx,
        "get",
        lambda *a, **k: _trace([root_no_marker, _generation("g1")], latency=2.0),
    )

    path = trace_export.export_trace("trace-no-marker")

    assert path is None


def test_no_generation_observations_is_incomplete(monkeypatch):
    # Root span present and "complete"-looking, but no LLM activity at all.
    monkeypatch.setattr(trace_export.httpx, "get", lambda *a, **k: _trace([_root()], latency=2.0))

    path = trace_export.export_trace("trace-no-generation")

    assert path is None


def test_observation_missing_end_time_is_incomplete(monkeypatch):
    unfinished_generation = _generation("g1", end_time=None)
    monkeypatch.setattr(
        trace_export.httpx,
        "get",
        lambda *a, **k: _trace([_root(), unfinished_generation], latency=2.0),
    )

    path = trace_export.export_trace("trace-unfinished-child")

    assert path is None


def test_waits_for_more_observations_before_accepting(monkeypatch):
    """One observation appears first; a second arrives on the next poll -
    the fingerprint must restabilize before export accepts it."""
    partial = [_root(), _generation("g1")]
    fuller = [_root(), _generation("g1"), _generation("g2")]
    _queue(monkeypatch, [_trace(partial, 3.0)] + _stable(fuller, 6.0))

    path = trace_export.export_trace("trace-grows")

    assert path is not None
    saved = json.loads(open(path).read())
    assert len(saved["observations"]) == 3


def test_zero_latency_is_incomplete_even_with_full_structure(monkeypatch):
    complete_but_zero_latency = _trace([_root(), _generation("g1")], latency=0)
    real = [_root(), _generation("g1")]
    _queue(monkeypatch, [complete_but_zero_latency, complete_but_zero_latency] + _stable(real, 4.0))

    path = trace_export.export_trace("trace-zero-latency")

    assert path is not None
    saved = json.loads(open(path).read())
    assert saved["latency"] == 4.0


def test_latency_far_from_expected_is_rejected(monkeypatch):
    # Trace reports 2s but the caller measured ~20s wall-clock for this turn -
    # a suspiciously small/partial export, per check 7.
    monkeypatch.setattr(
        trace_export.httpx, "get", lambda *a, **k: _trace([_root(), _generation("g1")], latency=2.0)
    )

    path = trace_export.export_trace("trace-latency-mismatch", expected_latency_seconds=20.0)

    assert path is None


def test_latency_within_tolerance_of_expected_is_accepted(monkeypatch):
    _queue(monkeypatch, _stable([_root(), _generation("g1")], latency=5.2))

    path = trace_export.export_trace("trace-latency-ok", expected_latency_seconds=5.0)

    assert path is not None


def test_timeout_never_stabilizing_does_not_save_a_file(monkeypatch):
    # Always missing the completion marker - never becomes "complete".
    monkeypatch.setattr(
        trace_export.httpx,
        "get",
        lambda *a, **k: _trace([_root(marker=None), _generation("g1")], latency=2.0),
    )

    path = trace_export.export_trace("trace-timeout")

    assert path is None
    assert not (trace_export._TRACES_DIR / "trace-timeout.json").exists()


def test_timeout_does_not_overwrite_a_previously_saved_valid_file(monkeypatch):
    trace_export._TRACES_DIR.mkdir(parents=True, exist_ok=True)
    existing_path = trace_export._TRACES_DIR / "trace-existing.json"
    existing_path.write_text(json.dumps({"observations": [{"id": "old"}], "latency": 5.0}))

    monkeypatch.setattr(
        trace_export.httpx,
        "get",
        lambda *a, **k: _trace([_root(marker=None), _generation("g1")], latency=2.0),
    )

    path = trace_export.export_trace("trace-existing")

    assert path is None
    saved = json.loads(existing_path.read_text())
    assert saved["observations"][0]["id"] == "old"  # untouched


def test_no_trace_id_returns_none_without_calling_api(monkeypatch):
    calls = []
    monkeypatch.setattr(trace_export.httpx, "get", lambda *a, **k: calls.append(1) or FakeResponse(200))

    assert trace_export.export_trace(None) is None
    assert calls == []


def test_tracing_disabled_returns_none_without_calling_api(monkeypatch):
    monkeypatch.setattr(config.settings, "langfuse_public_key", None)
    calls = []
    monkeypatch.setattr(trace_export.httpx, "get", lambda *a, **k: calls.append(1) or FakeResponse(200))

    assert trace_export.export_trace("trace-disabled") is None
    assert calls == []


def test_http_error_is_treated_as_not_yet_available(monkeypatch):
    import httpx as real_httpx

    responses = [real_httpx.HTTPError("boom")] + _stable([_root(), _generation("g1")], 3.0)

    def fake_get(*args, **kwargs):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(trace_export.httpx, "get", fake_get)

    path = trace_export.export_trace("trace-http-error")

    assert path is not None
