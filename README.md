---
title: LifeOps Agent
emoji: 🗓️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: space_app.py
pinned: false
---

# LifeOps Agent

A voice/text personal productivity agent. Type or speak a command like
*"remind me to call mom tomorrow at 5pm"* or *"schedule lunch with Sam
Thursday at noon"*, and a real tool-calling LLM agent decides what to do -
create/update/complete/delete reminders, journal entries, and (when
connected) real Google Calendar events - chaining up to 5 actions per
message and always pausing for confirmation before deleting anything.

## Features

- **Real tool-calling agent** (`app/agent.py`) - not regex matching. Chains
  multiple actions per turn, self-corrects on bad tool-call arguments
  (OpenAI strict structured outputs), and asks before any destructive action.
- **Real Google Calendar** via an MCP server (`app/mcp_client.py`) - falls
  back to a local mock calendar automatically if not configured.
- **Voice input** via Whisper (`app/transcription.py`).
- **Langfuse tracing** on every LLM/tool call, with an automatic local JSON
  export of each turn's full trace once Langfuse finishes indexing it
  (`app/trace_export.py`) - a completeness check (root span, generations,
  full timing, latency-consistency) prevents saving a partial trace.
- **Persistent conversations** - sessions and pending confirmations live in
  SQLite (`app/session_store.py`), so they survive a process restart.
- **A separate, simpler regex/single-shot path** (`app/parser.py`,
  `app/services/router.py`) still exists and is tested, but isn't wired
  into the UI - the real agent is the one actually used.

## Project layout

```
lifeops-agent/
  space_app.py               Hugging Face Spaces entry point
  app/
    main.py                  FastAPI app (classic + agent endpoints)
    agent.py                 The real tool-calling agent loop
    config.py                Environment-based settings
    db.py                    SQLite connection + schema
    session_store.py         Persisted conversations/pending confirmations
    schemas.py                Pydantic models
    llm_client.py             OpenAI SDK wrapper (Langfuse-instrumented)
    mcp_client.py              Google Calendar MCP bridge
    transcription.py           Whisper voice transcription
    tracing.py                  Langfuse client init + secret masking
    trace_export.py             Per-turn trace JSON export
    parser.py / services/router.py   Classic regex path (not agent-facing)
    tools/                       reminders.py, journal.py, calendar_mock.py
  frontend/
    gradio_app.py             Gradio chat UI
  tests/                       53 tests, all mocked (no live API calls)
```

## Setup

```bash
cd lifeops-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `OPENAI_API_KEY` - required for the real agent (tool-calling needs a real
  model); leave unset to fall back to the offline demo mode on `/command`.
- `GOOGLE_OAUTH_CREDENTIALS_PATH` - optional, enables real Google Calendar.
- `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` - optional,
  enables tracing + trace export.
- `GRADIO_AUTH_USER`/`GRADIO_AUTH_PASS` - **strongly recommended for any
  deployment reachable outside your own machine.** If either is unset, the
  Gradio UI has no login at all and anyone with the URL can use it (and
  spend your OpenAI credits) - a warning is logged at startup as a reminder.

## Run the Gradio agent (primary UI)

```bash
python frontend/gradio_app.py
# or, identically:
python space_app.py
```

## Architecture (v2)

- **LLM gateway** (`app/llm_gateway.py`, via LiteLLM) — one choke point that
  routes cheap/synthesis calls to a local quantized model (Ollama) and real
  tool-calling to a cloud model, with an in-process response cache and automatic
  local→cloud fallback. Feature-flagged by `MODEL_ROUTING_ENABLED` (off ⇒ all
  cloud, the instant rollback lever).
- **Per-session guardrails** (`app/budget.py`) — token/cost accounting from
  `response.usage`, a cost cap and a rate limit (both raise `BudgetExceededError`,
  ending the turn cleanly). Per-turn cost/tokens/latency show up as a footer in
  the chat and in structured logs.
- **Supervisor + specialists** — the agent loop is the supervisor; it delegates
  to `plan_schedule` (finds free slots around your **real Google Calendar** via
  the MCP get-freebusy tool, falling back to the local calendar if Calendar
  isn't connected) and `book_reservation` (`app/specialists/`). Booking is a
  human-in-the-loop action: it pauses for explicit approval + ID (guest name)
  before executing, reusing the same approval gate as destructive deletes.
- **Injection guardrails** (`app/guardrails.py`) — untrusted tool data (e.g. a
  calendar/reminder title) is fenced as data before re-entering context;
  destructive/booking actions require human approval regardless.
- **Journal RAG** (`app/journal_index.py`, `app/embeddings.py`) — journal
  entries are embedded (OpenAI) and stored in **Pinecone** (managed serverless
  vector DB); the `search_journal` tool retrieves relevant past entries so the
  agent answers reflective/recall questions grounded in them. Evaluated with
  Langfuse managed LLM-as-judge evaluators (faithfulness / context-relevance /
  answer-relevance) — see `docs/rag-evals.md`. No-ops without `PINECONE_API_KEY`.
- **Ops docs** — `docs/runbook.md` (incident first-actions + rollback levers),
  `docs/tech-debt.md` (what's done vs open).

To try local model routing: install [Ollama](https://ollama.com), run
`ollama pull llama3.1:8b`, then set `MODEL_ROUTING_ENABLED=true`.

## Run tests

```bash
pytest
```

The mocked unit/integration suite (`tests/`) runs on every push/PR via
GitHub Actions (`.github/workflows/ci.yml`). The golden-dataset eval
harness (`evals/run_evals.py`) makes real OpenAI/MCP calls and costs money
per run, so it's intentionally *not* part of CI - run it manually when you
want to check real-model behavior.

## Deployment

**Primary target: Railway**, using the included `Dockerfile` and
`railway.toml`. Push to a Railway-linked repo; it builds the Docker image
and runs `python space_app.py`, reading the dynamic `$PORT` Railway
assigns.

This repo can also still run as a Hugging Face Space (the YAML frontmatter
at the top of this file configures it, with `app_file: space_app.py` as
the entry point) if you prefer that instead - note HF Spaces' Gradio/Docker
SDK now requires a paid tier.

### Persistence on Railway

The SQLite database (`DATABASE_PATH`, default `lifeops.db`) lives on local
disk inside the container - **without a persistent volume, every redeploy
wipes it.** One-time setup:

1. In the Railway dashboard, add a **Volume** to this service and mount it
   at, e.g., `/data`.
2. Set `DATABASE_PATH=/data/lifeops.db` in the service's environment
   variables so the DB file lives on that volume instead of the ephemeral
   container filesystem.
3. Optionally, back up periodically: `python scripts/backup_db.py` dumps
   the DB to a timestamped `.sql` file under `backups/` (gitignored) -
   run it manually, or as a scheduled Railway cron service pointed at the
   same volume.

### Google Calendar on Railway

Real Google Calendar access (as opposed to the local mock calendar) needs
**two** things to survive on a headless host, and it's easy to get the
first working locally but miss the second:

1. **The OAuth client credentials JSON** (`GOOGLE_OAUTH_CREDENTIALS_PATH`) -
   the file you download from Google Cloud Console.
2. **A cached OAuth *token*** - generated the first time you authorize the
   app, via an interactive browser consent flow. Locally this gets cached
   automatically (by default under `~/.config/google-calendar-mcp/tokens.json`)
   and everything "just works" after that first login. **Railway has no
   browser and no way to complete that interactive flow**, so if all you do
   is set `GOOGLE_OAUTH_CREDENTIALS_PATH` there, calendar access will
   silently fall back to the local mock tool every time - there's no token
   for it to use.

First, generate the token once locally:

- Run the app locally with `GOOGLE_OAUTH_CREDENTIALS_PATH` set and complete
  the browser consent flow once (you'll see "Tokens updated and saved" in
  the logs). Find the resulting token file - by default
  `~/.config/google-calendar-mcp/tokens.json` on macOS/Linux.
- Make sure the token is long-lived: if your Google OAuth app is still in
  **Testing** mode, its refresh token expires after **7 days**. Publish the
  app (Google Auth Platform -> Audience -> "Publish app" / In production),
  then delete `tokens.json` and re-run the consent flow so a fresh,
  non-expiring token is issued (the `refresh_token_expires_in` field
  disappears once it's long-lived).

Then carry both files over to Railway, either way:

**Option A - environment variables (no volume shell needed).** Set two env
vars on the service to the *contents* of the two files, and the app writes
them back to real files on startup (see
`app/mcp_client.py::_materialize_google_secrets_from_env`):

- `GOOGLE_OAUTH_CREDENTIALS_JSON` = the full contents of your
  `gcp-oauth-keys.json`
- `GOOGLE_CALENDAR_MCP_TOKEN_JSON` = the full contents of your `tokens.json`

The files are written next to `DATABASE_PATH`, so pointing `DATABASE_PATH`
at a persistent volume (e.g. `/data/lifeops.db`) lets the refreshed token
survive redeploys. No `GOOGLE_OAUTH_CREDENTIALS_PATH` /
`GOOGLE_CALENDAR_MCP_TOKEN_PATH` needed - the app sets those itself.

**Option B - upload the files to the volume directly.** Put both files on
the same persistent volume as the database (e.g. `/data/gcp-oauth.keys.json`
and `/data/gcp-tokens.json` - Railway's dashboard lets you open a shell, or
`scp`/`railway run` a copy), then set
`GOOGLE_OAUTH_CREDENTIALS_PATH=/data/gcp-oauth.keys.json` and
`GOOGLE_CALENDAR_MCP_TOKEN_PATH=/data/gcp-tokens.json`.

**Never commit either file to git** - `tokens.json` contains a live refresh
token for your real calendar. With Option A the secrets live only in
Railway's env-var store; with Option B, only on the volume. Either way they
stay out of the repo entirely.

### Multi-user: "Sign in with Google"

By default the app is **single-user**: one shared password (or none), one
calendar. Set the four `GOOGLE_OAUTH_CLIENT_ID/SECRET`, `SESSION_SECRET`,
`TOKEN_ENCRYPTION_KEY` variables (see `.env.example`) to switch it into
**multi-user** mode instead:

- Every visitor signs in with their own Google account (app/web.py wraps the
  Gradio UI in a FastAPI OAuth login; unauthenticated requests redirect to
  `/login`).
- Reminders, journal entries, and calendar all become per-user - each person
  acts on their **own** Google Calendar (app/google_calendar.py), not a shared
  one. The single global MCP calendar is not started in this mode.

Setup:

1. **Create a Web application OAuth client** in Google Cloud Console
   (Credentials -> Create client -> Web application). This is separate from the
   Desktop client the single-user MCP calendar uses.
2. **Register redirect URIs** on it: `http://localhost:7860/oauth2callback`
   for local dev and `https://<your-railway-domain>/oauth2callback` for
   production. `OAUTH_REDIRECT_URI` must match the one in use.
3. **Set the variables** on the Railway service:
   `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`,
   `OAUTH_REDIRECT_URI=https://<your-railway-domain>/oauth2callback`,
   `SESSION_SECRET`, `TOKEN_ENCRYPTION_KEY` (generate the last two per the
   commands in `.env.example`). Point `DATABASE_PATH` at a persistent volume so
   users and their encrypted tokens survive redeploys.
4. **Authorize your users.** The Google *sensitive* calendar scope means the
   app must be verified before the general public can use it. Until then, add
   each person under Google Auth Platform -> Audience -> **Test users** (up to
   100). They'll see a one-time "Google hasn't verified this app" screen
   (Advanced -> continue) - expected for an unverified app.

Refresh tokens are stored **encrypted** (Fernet) in the `google_credentials`
table; never commit real secrets. A user can sign out (`/logout`) or, if their
token is revoked, the app asks them to reconnect by signing in again.
