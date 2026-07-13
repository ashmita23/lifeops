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

## Deployment

This repo is set up to run as a Hugging Face Space directly (the YAML
frontmatter at the top of this file configures it) - push to a Space repo
with secrets set in the Spaces UI, and `app.py` is the entry point.
