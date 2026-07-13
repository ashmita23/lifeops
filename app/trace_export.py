"""Exports a Langfuse trace (plus its nested observations) to a flat local
JSON file. The Langfuse dashboard is a trace tree you have to click through;
this gives you one file per turn you can just open or hand to me directly.

Uses the same Langfuse project credentials already configured in .env - no
new account or service needed.

Completeness is checked against all of the following before a trace is
accepted and written (confirmed live: a 200 response with a plausible-
looking but partial observation list is a real failure mode, not just a
theoretical one - simple "has some observations" checks passed on genuinely
truncated exports):

1. A root AGENT span (run_agent_turn) exists in the observation list.
2. That root span has an end time (the run actually finished).
3. That root span carries our own "lifeops_export_marker" metadata - an
   explicit application-level completion signal, not inferred from
   Langfuse's ingestion state alone (see app.agent.run_agent_turn).
4. At least one GENERATION observation exists (real LLM activity happened).
5. Every observation has both a start and end time (no still-in-flight
   children).
6. The full observation set (by id + endTime) is unchanged across
   EXPORT_STABILITY_CHECKS consecutive polls - late-arriving observations
   reset this counter.
7. The trace's own computed latency is within tolerance of the
   independently-measured agent latency passed in by the caller, when
   given - catches partial exports whose total latency is suspiciously
   smaller than what actually happened.
"""

import json
import logging
import time
from pathlib import Path

import httpx
from langfuse import get_client

from app.config import settings

logger = logging.getLogger(__name__)

_TRACES_DIR = Path("logs/traces")

# Polling configuration - all part of a "give up and return None" budget,
# never a hang: worst case is EXPORT_POLL_MAX_ATTEMPTS * EXPORT_POLL_INTERVAL_SECONDS.
EXPORT_POLL_MAX_ATTEMPTS = 20
EXPORT_POLL_INTERVAL_SECONDS = 1.5
# Consecutive polls with an unchanged, structurally-complete observation set
# before we consider the trace "done indexing," not just "started."
EXPORT_STABILITY_CHECKS = 3
# How far the trace's own computed latency may drift from the
# independently-measured agent latency and still be accepted.
LATENCY_TOLERANCE_SECONDS = 1.5
LATENCY_TOLERANCE_RATIO = 0.3


def _fetch_trace(trace_id: str, base_url: str) -> dict | None:
    try:
        response = httpx.get(
            f"{base_url}/api/public/traces/{trace_id}",
            params={"fields": "core,io,observations,metrics"},
            auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
            timeout=10,
        )
    except httpx.HTTPError as exc:
        logger.warning("Trace fetch request failed for trace_id=%s: %s", trace_id, exc)
        return None

    if response.status_code != 200:
        return None
    return response.json()


def _root_span(data: dict) -> dict | None:
    for obs in data.get("observations") or []:
        if obs.get("type") == "AGENT" and obs.get("parentObservationId") is None:
            return obs
    return None


def _is_complete(data: dict, expected_latency_seconds: float | None) -> bool:
    observations = data.get("observations") or []

    root = _root_span(data)
    if root is None or not root.get("endTime"):  # checks 1-2
        return False
    if (root.get("metadata") or {}).get("lifeops_export_marker") != "complete":  # check 3
        return False
    if not any(obs.get("type") == "GENERATION" for obs in observations):  # check 4
        return False
    if any(not obs.get("startTime") or not obs.get("endTime") for obs in observations):  # check 5
        return False
    if not data.get("latency"):
        return False

    if expected_latency_seconds is not None:  # check 7
        tolerance = max(LATENCY_TOLERANCE_SECONDS, expected_latency_seconds * LATENCY_TOLERANCE_RATIO)
        if abs(data["latency"] - expected_latency_seconds) > tolerance:
            return False

    return True


def _fingerprint(data: dict) -> tuple:
    return tuple(
        sorted((obs.get("id"), obs.get("endTime")) for obs in data.get("observations") or [])
    )


def _write_atomic(trace_id: str, data: dict) -> str:
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    final_path = _TRACES_DIR / f"{trace_id}.json"
    tmp_path = _TRACES_DIR / f".{trace_id}.json.tmp"
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(final_path)  # atomic on the same filesystem
    return str(final_path)


def export_trace(trace_id: str | None, expected_latency_seconds: float | None = None) -> str | None:
    """Polls the Langfuse API until the trace is complete (see module
    docstring's 7 checks) and writes it to logs/traces/<trace_id>.json.
    Returns the file path, or None if tracing isn't configured, trace_id is
    missing (e.g. demo mode), or the trace never stabilizes within the poll
    budget - in which case nothing is written (an existing valid file, if
    any, is left untouched).

    expected_latency_seconds: the independently-measured wall-clock time
    the caller's own code observed for this turn (e.g. gradio_app.py's
    agent_latency_ms / 1000). Used for check 7; omit to skip that check."""
    if not trace_id or not settings.tracing_enabled:
        return None

    get_client().flush()

    base_url = settings.langfuse_host or "https://cloud.langfuse.com"
    last_data: dict | None = None
    last_fingerprint: tuple | None = None
    stable_count = 0

    for attempt in range(EXPORT_POLL_MAX_ATTEMPTS):
        data = _fetch_trace(trace_id, base_url)
        if data is not None and _is_complete(data, expected_latency_seconds):
            fingerprint = _fingerprint(data)
            stable_count = stable_count + 1 if fingerprint == last_fingerprint else 1
            last_data, last_fingerprint = data, fingerprint
            if stable_count >= EXPORT_STABILITY_CHECKS:
                return _write_atomic(trace_id, last_data)
        if attempt < EXPORT_POLL_MAX_ATTEMPTS - 1:
            time.sleep(EXPORT_POLL_INTERVAL_SECONDS)

    logger.warning(
        "Trace %s did not reach a stable, complete state after %d attempts (%.1fs) - "
        "not saving an incomplete export.",
        trace_id,
        EXPORT_POLL_MAX_ATTEMPTS,
        EXPORT_POLL_MAX_ATTEMPTS * EXPORT_POLL_INTERVAL_SECONDS,
    )
    return None
