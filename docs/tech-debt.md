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

## Still open (lower priority — the original AI-eng gaps)

- **Cost/budget cap** — actions per turn are capped (`MAX_TOOL_ITERATIONS`);
  dollars spent are not. A confused loop could still run up a real bill.
- **Prompt-injection hardening** — text pulled back into context from calendar
  events / reminders is treated the same as trusted instructions. Should be
  fenced as data.
- **Evals** — mostly keyword-match; upgrading some cases to LLM-judged grading
  would catch semantic regressions string-matching misses.
- **PII in traces** — API keys are redacted before reaching Langfuse, but actual
  calendar/reminder content (personal data) is not.
- **Long-term memory** — history is trimmed (dropped) for the model's context
  window, never summarized. Fine for a demo; a real limit for long sessions.

## Known latent behavior (documented, not yet fixed)

- If the model batches other tool calls *after* a `respond_to_pending_confirmation`
  with `confirmed=false`, those trailing calls don't get synthetic tool-result
  messages before the immediate decline return, which could make the next turn's
  history API-invalid. Extremely unlikely in practice (the confirm tool is only
  offered in response to a yes/no), and untested. Modeling decline as a normal
  stop_reason (so trailing calls get `not_executed` results) would close it.
