"""Langfuse tracing setup.

Initializes the Langfuse singleton client (used implicitly by every
@observe-decorated function) with basic secret masking, before any traced
function runs. Safe to call even when no LANGFUSE_* keys are configured -
the client just becomes a disabled no-op, per Langfuse's own behavior.
"""

import re

from langfuse import Langfuse

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
