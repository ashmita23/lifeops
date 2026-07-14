"""Per-session cost/token budget, rate limiting, and usage accounting.

Two guardrails and one accounting surface, all keyed by session:
- check_budget(): refuse a call once a session has spent more than
  SESSION_COST_CAP_USD. Nothing else caps dollars - MAX_TOOL_ITERATIONS caps
  *actions*, not spend, so a pathological loop could still run up a real bill.
- check_rate(): refuse a call once a session exceeds SESSION_RATE_LIMIT_PER_MIN
  calls in the trailing 60s (in-memory sliding window).
- add_usage()/get_usage(): durable token+cost totals that back both the budget
  check and the observability surface (footer/logs).
"""

import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from app.config import settings
from app.db import connection_scope


class BudgetExceededError(RuntimeError):
    """Raised when a session's spend or call rate exceeds its cap. The agent
    catches this like LLMUnavailableError and ends the turn cleanly."""


# Sliding-window call timestamps per session, in memory (the rate limit is a
# fast-path guard, not something that needs to survive a restart). Guarded by a
# lock because Gradio handlers run in a threadpool.
_call_times: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def check_rate(session_id: str) -> None:
    limit = settings.session_rate_limit_per_min
    if limit <= 0:
        return
    now = time.monotonic()
    with _lock:
        window = _call_times[session_id]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            raise BudgetExceededError(
                f"Rate limit reached ({limit} calls/min). Please slow down."
            )
        window.append(now)


def get_usage(session_id: str) -> dict:
    with connection_scope() as conn:
        row = conn.execute(
            "SELECT prompt_tokens, completion_tokens, cost_usd, call_count, error_count "
            "FROM session_usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "call_count": 0, "error_count": 0}
    return dict(row)


def check_budget(session_id: str) -> None:
    cap = settings.session_cost_cap_usd
    if cap <= 0:
        return
    if get_usage(session_id)["cost_usd"] >= cap:
        raise BudgetExceededError(
            f"Per-session budget cap of ${cap:.2f} reached; stopping to avoid further spend."
        )


def add_usage(session_id: str, *, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection_scope() as conn:
        conn.execute(
            """
            INSERT INTO session_usage
                (session_id, prompt_tokens, completion_tokens, cost_usd, call_count, error_count, updated_at)
            VALUES (?, ?, ?, ?, 1, 0, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                completion_tokens = completion_tokens + excluded.completion_tokens,
                cost_usd = cost_usd + excluded.cost_usd,
                call_count = call_count + 1,
                updated_at = excluded.updated_at
            """,
            (session_id, prompt_tokens, completion_tokens, cost_usd, now),
        )


def record_error(session_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection_scope() as conn:
        conn.execute(
            """
            INSERT INTO session_usage (session_id, error_count, updated_at)
            VALUES (?, 1, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                error_count = error_count + 1, updated_at = excluded.updated_at
            """,
            (session_id, now),
        )
