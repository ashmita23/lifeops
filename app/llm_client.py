"""LLM entry point for the agent.

call_llm_with_tools() is the single choke point every LLM call flows through.
It runs the per-session guardrails (rate limit, budget cap), delegates the
actual provider call to the gateway (which handles model routing, caching, and
local->cloud fallback), records token/cost usage, and returns the raw response
message so callers keep inspecting .tool_calls / .content directly.

The session id is carried on a contextvar (session_scope) rather than a
function argument, so the public signature stays exactly
(messages, tools, tool_choice) - the agent's tests monkeypatch this function
with that signature, and threading a new parameter through would break them.
"""

import contextlib
import contextvars

from app.config import settings
from app import budget, llm_gateway


class LLMUnavailableError(RuntimeError):
    """Raised when no LLM backend is configured."""


# Request-scoped session id, set by run_agent_turn via session_scope(). The
# real call reads it for per-session budget/usage; mocked calls in tests never
# touch it, so the function signature stays unchanged.
_session_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("lifeops_session_id", default=None)


@contextlib.contextmanager
def session_scope(session_id: str | None):
    token = _session_ctx.set(session_id)
    try:
        yield
    finally:
        _session_ctx.reset(token)


def call_llm_with_tools(messages: list[dict], tools: list[dict], tool_choice: str = "auto"):
    """Sends a chat history plus tool schemas and returns the raw response
    message object, so callers can inspect .tool_calls or .content directly.

    tool_choice="none" forces a plain-text response with no further tool calls
    (the synthesis/confirmation/summary steps); the tool schemas are omitted
    from the request entirely rather than sent-but-unusable."""
    if settings.demo_mode:
        raise LLMUnavailableError("No OPENAI_API_KEY configured; running in demo mode")

    session_id = _session_ctx.get()

    # Per-session guardrails run BEFORE the call so an over-budget or
    # over-rate session never spends more. Both raise BudgetExceededError,
    # which the agent catches and turns into a clean end-of-turn message.
    if session_id is not None:
        budget.check_rate(session_id)
        budget.check_budget(session_id)

    try:
        result = llm_gateway.complete(messages, tools, tool_choice)
    except Exception:
        if session_id is not None:
            budget.record_error(session_id)
        raise

    # A cache hit consumed no new tokens and cost nothing, so it is not billed.
    if session_id is not None and result.usage is not None and not result.cache_hit:
        budget.add_usage(
            session_id,
            prompt_tokens=getattr(result.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(result.usage, "completion_tokens", 0) or 0,
            cost_usd=result.cost_usd,
        )

    return result.message
