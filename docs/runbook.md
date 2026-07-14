# LifeOps Agent — Incident Runbook

What each alarm/symptom means and the first action for it. Kept short on
purpose: the point is a fast first move, not exhaustive diagnosis.

## Signals to watch (in logs / Langfuse)
- `llm_call model=... route=fallback_cloud` — the local model failed and the
  gateway fell back to cloud. One-off = fine. Sustained = local tier is down.
- `Per-session budget cap ... reached` / `Rate limit reached` — a session hit
  `SESSION_COST_CAP_USD` / `SESSION_RATE_LIMIT_PER_MIN`.
- `guardrail: user input matched injection patterns` — a message looked like a
  prompt-injection attempt (logged, not blocked).
- `Google Calendar: NOT connected` at startup — running on the mock calendar.
- `chat turn ... cost=$... errors=N` — per-turn cost/error surface.

## Incidents

### Local model down (fallback firing constantly)
- **Means:** Ollama is unreachable or the model isn't pulled; every cheap-tier
  call is paying the fallback penalty (a failed local attempt + a cloud call =
  slower and more expensive than just going cloud).
- **First action:** flip the feature flag off — set `MODEL_ROUTING_ENABLED=false`.
  All traffic goes straight to cloud; no fallback penalty. This is the rollback
  lever; no code change or redeploy logic needed.
- **Then:** check `ollama list` / `OLLAMA_BASE_URL`; `ollama pull <LOCAL_MODEL>`;
  re-enable the flag once healthy.

### Budget cap hit for a session
- **Means:** that session spent over `SESSION_COST_CAP_USD` (working as
  designed — the guard stopped further spend).
- **First action:** confirm it's a real user, not a loop. Check the session's
  traces for repeated identical tool calls (repeat-detection should catch true
  loops; a cap hit without repeats is just heavy usage).
- **Then:** raise `SESSION_COST_CAP_USD` if legitimate, or investigate the loop.

### Costs spiking overall
- **First action:** confirm routing is on (`MODEL_ROUTING_ENABLED=true`) so
  cheap turns use local; confirm `LLM_CACHE_ENABLED=true`.
- **Then:** check Langfuse for which model/turns dominate cost; consider routing
  more call types to local.

### Calendar silently using the mock
- **Means:** the MCP calendar server didn't connect (bad/missing OAuth creds or,
  on a headless host, no token).
- **First action:** check the startup `status=failed: ...` line for the reason.
- **Then:** verify `GOOGLE_OAUTH_CREDENTIALS_PATH` and, on Railway, that a token
  is present at `GOOGLE_CALENDAR_MCP_TOKEN_PATH` (see README "Google Calendar on
  Railway").

### Suspected prompt injection
- **Means:** untrusted data (an event/reminder title) tried to steer the agent.
- **Design:** injected tool data is fenced (treated as data), and destructive/
  booking actions still require human approval regardless — so an injection
  alone cannot delete or book anything.
- **First action:** review the flagged content; no emergency action needed
  unless an approval was granted. Tighten `app/guardrails.py` patterns if a new
  phrasing slipped through.

### Langfuse export failing (401)
- **Means:** trace export can't authenticate (known: keys rotated/expired).
- **Impact:** observability only — the agent itself is unaffected.
- **First action:** refresh `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` /
  `LANGFUSE_HOST` (check the region host).

## Rollback levers (no redeploy)
- `MODEL_ROUTING_ENABLED=false` — everything to cloud.
- `LLM_CACHE_ENABLED=false` — disable response cache.
- `SESSION_COST_CAP_USD` / `SESSION_RATE_LIMIT_PER_MIN` — tighten/loosen guards.
