"""OpenAI-compatible LLM client abstraction.

If OPENAI_API_KEY is configured, call_llm() sends the prompt to a real
OpenAI-compatible chat completions endpoint. Otherwise it raises
LLMUnavailableError so callers (namely app.parser) can fall back to the
deterministic local parser used for offline demo mode.
"""

# app.config must be imported (and its load_dotenv() run) before langfuse,
# or Langfuse may initialize its singleton client with missing credentials.
from app.config import settings

from langfuse.openai import OpenAI  # drop-in: auto-captures model/tokens/cost
# as a proper "generation" observation, nested under whatever @observe span
# is active (e.g. app.agent.run_agent_turn) - no manual @observe needed here.


class LLMUnavailableError(RuntimeError):
    """Raised when no LLM backend is configured."""


_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _get_client():
    # Always pass an explicit, non-empty base_url. python-dotenv loads
    # OPENAI_BASE_URL= (empty) into the process environment, and the openai
    # SDK falls back to reading that env var itself if we don't pass one -
    # an empty-but-present value wins over the SDK's own default and breaks
    # every request. Passing a real string here always short-circuits that.
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or _DEFAULT_OPENAI_BASE_URL,
    )


def call_llm(prompt: str) -> str:
    if settings.demo_mode:
        raise LLMUnavailableError("No OPENAI_API_KEY configured; running in demo mode")

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def call_llm_with_tools(messages: list[dict], tools: list[dict], tool_choice: str = "auto"):
    """Sends a chat history plus tool schemas and returns the raw response
    message object, so callers can inspect .tool_calls or .content directly.

    tool_choice="none" forces a plain-text response with no further tool
    calls - used for the synthesis step after a tool has already run."""
    if settings.demo_mode:
        raise LLMUnavailableError("No OPENAI_API_KEY configured; running in demo mode")

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0,
    )
    return response.choices[0].message
