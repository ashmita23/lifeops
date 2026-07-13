"""Golden dataset eval runner.

Runs every case in evals/golden_cases.py against the REAL agent (real
OpenAI/MCP calls - this costs a small amount of real money and time, run
it on demand, not as part of pytest/CI). Prints a pass/fail summary table
and writes a flat comparison report to evals/results/<timestamp>.json -
compare two runs by just diffing the two JSON files, or handing both to
Claude directly.

Usage:
    python evals/run_evals.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

# Isolated from the real lifeops.db - reminders/journal/local-calendar state
# from one eval run shouldn't leak into the next and skew reproducibility.
# Reset fresh every run. NOTE: this does NOT isolate the real Google
# Calendar - cases that create an MCP calendar event without a matching
# delete step leave a real, un-cleaned-up event on your actual calendar.
_EVAL_DB_PATH = Path(__file__).parent / "eval_lifeops.db"
settings.database_path = str(_EVAL_DB_PATH)

from app.db import init_db
from app.mcp_client import start as mcp_start
from app.tracing import init_tracing
from app.agent import run_agent_turn
from app.trace_export import export_trace
from evals.golden_cases import GOLDEN_CASES

_RESULTS_DIR = Path(__file__).parent / "results"


def _check_tools(actual_tools: list[str], expected_tools: list[str] | None) -> bool:
    """Subset match, not exact - the model legitimately calls auxiliary read
    tools (list-calendars, search-events, an availability check before
    creating) that aren't the specific action being tested for. We only
    care that every expected tool was actually called at least once, not
    that nothing else was."""
    if expected_tools is None:
        return True
    return set(expected_tools).issubset(set(actual_tools))


def _check_min_actions(actual_count: int, expected_min: int | None) -> bool:
    if expected_min is None:
        return True
    return actual_count >= expected_min


def _check_keywords(message: str, expected_keywords: list[str] | None) -> bool:
    if not expected_keywords:
        return True
    lowered = message.lower()
    return all(keyword.lower() in lowered for keyword in expected_keywords)


def run_case(case: dict) -> dict:
    session_id = None
    result = None
    start = time.perf_counter()

    for turn_text in case["turns"]:
        result = run_agent_turn(session_id=session_id, input_text=turn_text)
        session_id = result.session_id

    latency_seconds = time.perf_counter() - start

    actual_tools = [action.tool for action in result.actions]
    tools_ok = _check_tools(actual_tools, case.get("expected_tools"))
    count_ok = _check_min_actions(len(actual_tools), case.get("expected_min_actions"))
    keywords_ok = _check_keywords(result.message, case.get("expected_keywords"))
    passed = tools_ok and count_ok and keywords_ok

    trace_path = export_trace(result.trace_id, expected_latency_seconds=latency_seconds)

    return {
        "case_id": case["id"],
        "turns": case["turns"],
        "expected_tools": case.get("expected_tools"),
        "expected_min_actions": case.get("expected_min_actions"),
        "expected_keywords": case.get("expected_keywords"),
        "actual_tools": actual_tools,
        "message": result.message,
        "passed": passed,
        "latency_seconds": round(latency_seconds, 2),
        "trace_path": trace_path,
    }


def main() -> None:
    _EVAL_DB_PATH.unlink(missing_ok=True)  # fresh local state every run
    init_tracing()
    init_db()
    mcp_start()

    results = []
    for case in GOLDEN_CASES:
        print(f"Running {case['id']}...", flush=True)
        try:
            outcome = run_case(case)
        except Exception as exc:
            outcome = {
                "case_id": case["id"],
                "turns": case["turns"],
                "passed": False,
                "error": str(exc),
            }
        results.append(outcome)
        status = "PASS" if outcome.get("passed") else "FAIL"
        latency = outcome.get("latency_seconds", "?")
        print(f"  [{status}] {case['id']} ({latency}s)")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = _RESULTS_DIR / f"{timestamp}.json"
    report_path.write_text(json.dumps(results, indent=2))

    passed_count = sum(1 for r in results if r.get("passed"))
    print(f"\n{passed_count}/{len(results)} passed. Report: {report_path}")


if __name__ == "__main__":
    main()
