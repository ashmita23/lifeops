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

    # We deliberately do NOT attach LiteLLM's Langfuse callback. Both of
    # LiteLLM's callbacks are incompatible with the Langfuse 4.x SDK this app
    # requires (@observe/get_client): the classic "langfuse" callback reads
    # langfuse.version.__version__ (renamed to langfuse._version in 4.x ->
    # AttributeError on every call) and also passes a removed sdk_integration
    # kwarg; the "langfuse_otel" callback overwrites our @observe agent span,
    # mislabeling the trace root. LiteLLM 1.92.0 is the newest release, so
    # there is no upgrade that fixes this.
    #
    # Instead, app/llm_gateway.py wraps each real litellm.completion call in a
    # Langfuse generation observation using the native 4.x SDK - it already
    # has the tokens (response.usage) and cost, so the generation nests
    # cleanly under the active @observe trace with no version conflict.
