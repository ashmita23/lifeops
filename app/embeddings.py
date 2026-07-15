"""Text embeddings via OpenAI, for journal RAG retrieval.

Uses a direct OpenAI client (like app/transcription.py), NOT the chat gateway
in app/llm_client.py - embeddings are a different API surface (the embeddings
endpoint, not chat completions) and shouldn't run through the tool-calling
budget/routing pipeline.
"""

from openai import OpenAI

from app.config import settings
from app.llm_client import LLMUnavailableError

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _client() -> OpenAI:
    # Explicit base_url: python-dotenv may load an empty OPENAI_BASE_URL into the
    # env, which the SDK would otherwise pick up and break requests.
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or _DEFAULT_OPENAI_BASE_URL,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Raises LLMUnavailableError in demo mode."""
    if settings.demo_mode:
        raise LLMUnavailableError("No OPENAI_API_KEY configured; cannot embed")
    if not texts:
        return []
    response = _client().embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
