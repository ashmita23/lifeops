# Deferred tech debt — needs your call

Two items from the code review were **intentionally not done autonomously**
because they're either risky to do without you watching, or involve deleting
code you wrote. They're written up here so the decision is yours.

Everything else from the review is done and on `main` (race-condition fix,
fail-loud MCP, real timezone, tighter confirmation guard, contract tests,
doc-drift fixes). Test suite: 66 passing.

---

## 1. Dead / duplicated command paths — DELETE decision needed

There are three ways a user command can be handled, and only one is live:

| Path | File | Status |
|---|---|---|
| Regex parser | `app/parser.py` | Only reachable via the router below |
| Classify-then-route (single shot) | `app/services/router.py` | `get_daily_summary` is used by the agent; `process_command` only by `app/main.py` |
| Tool-calling agent | `app/agent.py` | **This is what's deployed** (Dockerfile runs the Gradio app, which uses the agent) |

`app/main.py` (the FastAPI backend) is **not deployed at all** — the
Dockerfile runs `space_app.py` (Gradio) only.

**Why I didn't touch it:** deleting files you authored is a one-way,
judgment-dependent call. You may want to keep the FastAPI backend as a
portfolio talking point ("same core, two frontends"), or you may want it
gone as clutter. That's yours to decide.

**Recommended options (pick one):**
- **Keep + justify:** add a one-paragraph note to the README explaining that
  `app/main.py` is an alternate REST frontend and `parser.py`/`router.py` are
  the pre-agent baseline kept for comparison. Turns "dead code" into "shows
  iteration." Lowest effort.
- **Delete:** remove `app/parser.py`, `app/services/router.py` (keep
  `get_daily_summary` — move it into the agent or a small `summary.py`),
  `app/main.py`, and their tests. Cleaner repo, but loses the REST story.

An interviewer *will* ask "what's `parser.py`, is it used?" — the goal is
that you have a crisp answer either way, not that it's silent surface area.

---

## 2. `app/agent.py` god module — REFACTOR needed (do it together)

`agent.py` is ~950 lines, and `_run_agent_turn_body` is a single ~200-line
function containing the whole tool loop as a `stop_reason` state machine.
It works and it's well-commented, but it's more than one person can hold in
their head, and "one giant function" is a fair review criticism.

**Why I didn't do it solo:** this is the one change with real regression
risk, and the payoff of doing it *with* you is that you learn the refactor —
extracting a state machine, using the test suite as a safety net. It's a
perfect pairing exercise, not a background task.

**Recommended approach when we pair:**
1. Extract the per-call body of the `for call in tool_calls` loop into a
   `_process_one_call(call, ...) -> Outcome` helper that returns a small
   result object (execute / pause-for-confirm / stop-repeat / stop-unknown),
   instead of mutating `stop_reason` inline.
2. The loop becomes: build outcome, act on it. Each concern (confirm tool,
   destructive gate, unknown tool, normal execute) becomes its own readable
   branch/function.
3. Run the 66 tests after *each* extraction step — never two changes deep
   without a green bar. That discipline is the whole lesson.

Target: no single function over ~60 lines, same behavior, same tests green.

---

## Also worth doing later (lower priority, from the original 7 AI-eng gaps)
- Per-session token/cost budget cap (nothing caps dollars today).
- Prompt-injection hardening: treat calendar/reminder text pulled back into
  context as data, never instructions.
- Upgrade some evals from keyword-match to LLM-judged grading.
- PII: calendar/reminder content flows into Langfuse traces unredacted.
