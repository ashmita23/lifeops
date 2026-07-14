"""Langfuse tracing setup.

Initializes the Langfuse singleton client (used implicitly by every
@observe-decorated function) with basic secret masking, before any traced
function runs. Safe to call even when no LANGFUSE_* keys are configured -
the client just becomes a disabled no-op, per Langfuse's own behavior.
"""

import logging
import re

from langfuse import Langfuse

from app.config import settings

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),  # OpenAI-style API keys
    re.compile(r"GOCSPX-[A-Za-z0-9_-]+"),  # Google OAuth client secrets
]


def _mask(data):
    if isinstance(data, str):
        for pattern in _SECRET_PATTERNS:
            data = pattern.sub("[REDACTED]", data)
    return data


def init_tracing() -> None:
    Langfuse(mask=_mask)

    # LLM calls now go through LiteLLM (app/llm_gateway.py), not the old
    # langfuse.openai drop-in. LiteLLM's Langfuse callback emits a generation
    # (with tokens/cost) per call that nests under the active @observe trace,
    # preserving auto-capture. Only wire it when tracing is actually
    # configured, so we don't attach a callback that errors on every call.
    if settings.tracing_enabled:
        try:
            import litellm

            for cb, hook in (("langfuse", litellm.success_callback), ("langfuse", litellm.failure_callback)):
                if cb not in hook:
                    hook.append(cb)
        except Exception:
            logger.warning("Could not attach LiteLLM Langfuse callback; continuing without it.", exc_info=True)
