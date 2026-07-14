"""Voice transcription via OpenAI's hosted Whisper API.

Uses a direct OpenAI client rather than the LLM gateway: transcription is the
audio API, not chat completions, so it doesn't share the gateway's routing/
budget pipeline. It keeps its own small client factory here.
"""

from openai import OpenAI

from app.config import settings
from app.llm_client import LLMUnavailableError

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _audio_client() -> OpenAI:
    # Pass an explicit base_url: python-dotenv may load an empty OPENAI_BASE_URL
    # into the env, which the SDK would otherwise pick up and break requests.
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or _DEFAULT_OPENAI_BASE_URL,
    )


def transcribe_audio(file_path: str) -> str:
    if settings.demo_mode:
        raise LLMUnavailableError("No OPENAI_API_KEY configured; running in demo mode")

    client = _audio_client()
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text
