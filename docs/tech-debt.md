# Tech debt

## Resolved

- **Dead pre-agent command paths** — `app/parser.py`, `app/services/router.py`,
  and the never-deployed FastAPI backend `app/main.py` have been deleted, along
  with their tests and now-orphaned schemas. `get_daily_summary` (the agent's
  only dependency on the old router) moved to `app/services/summary.py`. The
  tool-calling agent is now the single command path.
- **`app/agent.py` god function** — the ~200-line tool loop was refactored into
  an explicit `_TurnState` object plus `_process_one_call` / `_process_confirmation`
  / `_finish_stopped`. Behavior-preserving (test suite green before and after);
  largest logic function dropped from ~200 to ~58 lines.
- **Cost/budget cap** — per-session cost cap + rate limit now enforced
  (`app/budget.py`), raising `BudgetExceededError` which the agent ends the turn
  on. Tokens/cost captured from `response.usage` via the gateway.
- **Prompt-injection hardening** — untrusted tool data is now fenced
  (`app/guardrails.py`, wired into `_TurnState.append_tool_result`); destructive
  AND booking actions still require human approval regardless.
- **Evals** — LLM-as-judge + trajectory checks added (`evals/run_evals.py`),
  plus injection/planner/booking golden cases.

## Still open (lower priority)

- **PII in traces** — API keys are redacted before reaching Langfuse, but actual
  calendar/reminder content (personal data) is not.
- **Long-term memory** — history is trimmed (dropped) for the model's context
  window, never summarized. Fine for a demo; a real limit for long sessions.
- **Cache/rate-limit durability** — the response cache and rate-limit window are
  in-process; multi-replica deployment would want Redis-backed versions.

## Known latent behavior (documented, not yet fixed)

- If the model batches other tool calls *after* a `respond_to_pending_confirmation`
  with `confirmed=false`, those trailing calls don't get synthetic tool-result
  messages before the immediate decline return, which could make the next turn's
  history API-invalid. Extremely unlikely in practice (the confirm tool is only
  offered in response to a yes/no), and untested. Modeling decline as a normal
  stop_reason (so trailing calls get `not_executed` results) would close it.
