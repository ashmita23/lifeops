"""Bridges the synchronous rest of this app to an async MCP stdio server.

Currently used for the Google Calendar MCP server
(@cocal/google-calendar-mcp, run via npx). MCP client sessions are
asyncio-based, but app.agent.run_agent_turn and the rest of this codebase
are synchronous (called from FastAPI's threadpool or plain Gradio
callbacks). Rather than rewrite that call chain to async, we run a single
background thread with its own event loop for the process lifetime and
bridge into it with asyncio.run_coroutine_threadsafe(...).result().

If the server isn't configured or fails to start (e.g. Google OAuth setup
hasn't been completed yet), every function here degrades to "no MCP tools
available" instead of raising, so the rest of the agent keeps working with
just its local tools.
"""

import asyncio
import logging
import threading

# app.config must be imported (and its load_dotenv() run) before langfuse,
# or Langfuse may initialize its singleton client with missing credentials.
from app.config import settings

from langfuse import observe
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_session: ClientSession | None = None
_session_cm = None
_stdio_cm = None
_mcp_tools_cache: list[dict] | None = None


def _server_params() -> StdioServerParameters | None:
    if not settings.google_oauth_credentials_path:
        return None
    return StdioServerParameters(
        command="npx",
        args=["-y", "@cocal/google-calendar-mcp"],
        env={"GOOGLE_OAUTH_CREDENTIALS": settings.google_oauth_credentials_path},
    )


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _connect() -> None:
    global _session, _session_cm, _stdio_cm

    params = _server_params()
    if params is None:
        logger.info("GOOGLE_OAUTH_CREDENTIALS_PATH not set - skipping Google Calendar MCP server.")
        return

    try:
        _stdio_cm = stdio_client(params)
        read, write = await _stdio_cm.__aenter__()
        _session_cm = ClientSession(read, write)
        _session = await _session_cm.__aenter__()
        await _session.initialize()
        logger.info("Connected to Google Calendar MCP server.")
    except Exception:
        logger.warning("Could not start Google Calendar MCP server; continuing without it.", exc_info=True)
        _session = None


def start() -> None:
    """Starts the background event loop + MCP connection. Safe to call even
    when MCP isn't configured - it just becomes a no-op in that case."""
    global _loop, _thread

    if _loop is not None:
        return

    _loop = asyncio.new_event_loop()
    _thread = threading.Thread(target=_run_loop, args=(_loop,), daemon=True)
    _thread.start()

    future = asyncio.run_coroutine_threadsafe(_connect(), _loop)
    future.result(timeout=30)


def _mcp_tool_to_openai_schema(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


async def _list_tools_async() -> list[dict]:
    if _session is None:
        return []
    result = await _session.list_tools()
    return [_mcp_tool_to_openai_schema(tool) for tool in result.tools]


def get_mcp_tools() -> list[dict]:
    """Returns cached OpenAI-function-calling-shaped tool schemas for
    whatever the MCP server exposes. Empty list if MCP isn't connected."""
    global _mcp_tools_cache

    if _loop is None:
        start()

    if _session is None:
        return []

    if _mcp_tools_cache is None:
        future = asyncio.run_coroutine_threadsafe(_list_tools_async(), _loop)
        _mcp_tools_cache = future.result(timeout=15)

    return _mcp_tools_cache


def is_mcp_tool(name: str) -> bool:
    return any(tool["function"]["name"] == name for tool in get_mcp_tools())


async def _call_tool_async(name: str, args: dict) -> dict:
    result = await _session.call_tool(name, args)
    content_items = []
    for item in result.content:
        text = getattr(item, "text", None)
        content_items.append(text if text is not None else str(item))
    return {"result": content_items, "is_error": bool(getattr(result, "isError", False))}


@observe(name="call_mcp_tool", as_type="tool")
def call_mcp_tool(name: str, args: dict) -> dict:
    if _session is None or _loop is None:
        return {"error": "Google Calendar MCP server is not connected."}

    future = asyncio.run_coroutine_threadsafe(_call_tool_async(name, args), _loop)
    return future.result(timeout=30)
