"""LLM gateway: model routing, response caching, and local->cloud fallback.

All LLM traffic funnels through app.llm_client.call_llm_with_tools, which
delegates the actual provider call to complete() here. Keeping this logic in
one module (rather than scattered around the agent) is what lets "route",
"cache", and "fall back" each be one small, testable function instead of
conditionals sprinkled through the call sites.

Provider calls go through LiteLLM, a thin gateway over many providers
(OpenAI, Anthropic, Ollama, ...) that all return the same OpenAI-shaped
response object - so callers keep using .choices[0].message.tool_calls/
.content/.usage unchanged. Langfuse cost/token capture is wired via LiteLLM's
callback in app.tracing, not the old langfuse.openai drop-in.
"""

import hashlib
import json
import logging

import litellm

from app.config import settings

logger = logging.getLogger(__name__)

# In-process response cache. Keyed on (model, messages, tools, tool_choice),
# so an identical request within a run returns instantly and for free. Fine as
# a plain dict for a single-process app; a STRETCH item swaps in Redis / a
# semantic cache. Bounded loosely to avoid unbounded growth.
_CACHE: dict[str, object] = {}
_CACHE_MAX_ENTRIES = 512


def choose_model(messages: list[dict], tools: list[dict], tool_choice: str) -> str:
    """Pick the cheap local tier vs the strong cloud tier for this call.

    Heuristic: a call that can't or won't invoke tools (the tool_choice="none"
    synthesis/summary calls, or a turn offering no tools) is low-stakes phrasing
    work a small local model handles well - route it local. A real tool-calling
    turn needs the stronger model's function-calling reliability - route cloud.
    When routing is flag-disabled, everything goes cloud (safe default)."""
    if not settings.model_routing_enabled:
        return settings.cloud_model
    if tool_choice == "none" or not tools:
        return settings.local_model
    return settings.cloud_model


def _is_local(model: str) -> bool:
    return model.startswith("ollama/")


def cache_key(model: str, messages: list[dict], tools: list[dict], tool_choice: str) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _completion_kwargs(model: str, messages: list[dict], tools: list[dict], tool_choice: str) -> dict:
    kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
    if tool_choice != "none":
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if _is_local(model):
        kwargs["api_base"] = settings.ollama_base_url
        kwargs["timeout"] = settings.local_timeout_seconds
    return kwargs


class GatewayResult:
    """What complete() returns: the raw provider message plus metadata the
    observability surface reports (which model actually served it, whether it
    was a cache hit, whether a local->cloud fallback fired, tokens, and cost)."""

    def __init__(self, message, *, model: str, cache_hit: bool, fell_back: bool, usage=None, cost_usd: float = 0.0):
        self.message = message
        self.model = model
        self.cache_hit = cache_hit
        self.fell_back = fell_back
        self.usage = usage
        self.cost_usd = cost_usd


def _cost_of(response) -> float:
    """USD cost of one response via LiteLLM's pricing tables. Returns 0.0 for
    models it can't price (e.g. a local Ollama model, which is genuinely free)."""
    try:
        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0


def complete(messages: list[dict], tools: list[dict], tool_choice: str) -> GatewayResult:
    """Route -> (cache) -> call, falling back local->cloud on any local error.

    The fallback is the whole point of a gateway: a flaky/slow/absent local
    model must never take the feature down - it silently retries on cloud."""
    model = choose_model(messages, tools, tool_choice)
    key = cache_key(model, messages, tools, tool_choice)

    if settings.llm_cache_enabled and key in _CACHE:
        cached = _CACHE[key]
        logger.info("llm_call model=%s route=cache_hit", model)
        # A cache hit spends nothing new - cost_usd is 0 by design.
        return GatewayResult(cached.choices[0].message, model=model, cache_hit=True, fell_back=False,
                             usage=getattr(cached, "usage", None), cost_usd=0.0)

    fell_back = False
    try:
        response = litellm.completion(**_completion_kwargs(model, messages, tools, tool_choice))
        served_model = model
    except Exception:
        if not _is_local(model):
            raise  # cloud failure has nowhere to fall back to - let it surface
        logger.warning("Local model %s failed; falling back to cloud %s.", model, settings.cloud_model,
                       exc_info=True)
        served_model = settings.cloud_model
        fell_back = True
        response = litellm.completion(
            **_completion_kwargs(served_model, messages, tools, tool_choice)
        )

    if settings.llm_cache_enabled:
        if len(_CACHE) >= _CACHE_MAX_ENTRIES:
            _CACHE.clear()
        _CACHE[key] = response

    logger.info("llm_call model=%s route=%s", served_model, "fallback_cloud" if fell_back else "direct")
    return GatewayResult(
        response.choices[0].message,
        model=served_model,
        cache_hit=False,
        fell_back=fell_back,
        usage=getattr(response, "usage", None),
        cost_usd=_cost_of(response),
    )
