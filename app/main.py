"""FastAPI entrypoint for the LifeOps agent."""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from app import mcp_client
from app.agent import run_agent_turn
from app.db import init_db
from app.schemas import AgentResponse, AgentTurnRequest, AgentTurnResult, UserCommand
from app.services.router import get_daily_summary, process_command
from app.tracing import init_tracing
from app.transcription import transcribe_audio

app = FastAPI(title="LifeOps Agent")


@app.on_event("startup")
def on_startup() -> None:
    init_tracing()
    init_db()
    mcp_client.start()


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "lifeops-agent"}


@app.post("/command", response_model=AgentResponse)
def handle_command(command: UserCommand) -> AgentResponse:
    return process_command(
        input_text=command.input_text,
        input_type=command.input_type,
        timezone=command.timezone,
    )


@app.get("/summary/today")
def summary_today() -> dict:
    return get_daily_summary()


@app.post("/init-db")
def init_database() -> dict:
    init_db()
    return {"status": "initialized"}


@app.post("/agent/command", response_model=AgentTurnResult)
def handle_agent_command(turn: AgentTurnRequest) -> AgentTurnResult:
    return run_agent_turn(
        session_id=turn.session_id,
        input_text=turn.input_text,
        timezone=turn.timezone,
    )


@app.post("/agent/voice-command", response_model=AgentTurnResult)
async def handle_agent_voice_command(
    file: UploadFile, session_id: str | None = None, timezone: str = "America/Chicago"
) -> AgentTurnResult:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        transcribed_text = transcribe_audio(tmp.name)

    return run_agent_turn(session_id=session_id, input_text=transcribed_text, timezone=timezone)
