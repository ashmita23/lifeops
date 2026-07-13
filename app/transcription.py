"""Voice transcription via OpenAI's hosted Whisper API."""

from app.config import settings
from app.llm_client import LLMUnavailableError, _get_client


def transcribe_audio(file_path: str) -> str:
    if settings.demo_mode:
        raise LLMUnavailableError("No OPENAI_API_KEY configured; running in demo mode")

    client = _get_client()
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text
