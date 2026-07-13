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

## Run the FastAPI backend

```bash
uvicorn app.main:app --reload
```

- `GET /` - health check
- `POST /command` - classic regex/single-shot path
- `POST /agent/command` - the real tool-calling agent
- `POST /agent/voice-command` - agent + Whisper transcription
- `GET /summary/today` - today's reminders, events, journal entries

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

To fix this, generate the token once locally, then carry it over:

1. Run the app locally with `GOOGLE_OAUTH_CREDENTIALS_PATH` set and
   complete the browser consent flow once (you'll see "Tokens updated and
   saved" in the logs). Find the resulting token file - by default
   `~/.config/google-calendar-mcp/tokens.json` on macOS/Linux.
2. Upload both the credentials JSON and that `tokens.json` onto the same
   persistent Railway volume you set up for the database (e.g. as
   `/data/gcp-oauth.keys.json` and `/data/gcp-tokens.json` - Railway's
   dashboard lets you open a shell on the volume, or you can `scp`/`railway
   run` a copy).
3. Set `GOOGLE_OAUTH_CREDENTIALS_PATH=/data/gcp-oauth.keys.json` and
   `GOOGLE_CALENDAR_MCP_TOKEN_PATH=/data/gcp-tokens.json` in the service's
   environment variables.

**Never commit either of these files to git** - `tokens.json` contains a
live refresh token for your real calendar. Both stay off the volume's
public surface and out of the repo entirely; they only need to exist on
the Railway volume itself.
