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
import os
import threading
from pathlib import Path

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

# Human-readable calendar-connection status. The failure mode this exists for:
# a misconfigured calendar silently degrades to the local mock with no visible
# signal (exactly what happened on the first Railway deploy). Surfacing a
# status here lets startup, a health check, or the UI say WHY calendar is
# unavailable instead of leaving the user to guess. One of:
#   "not_configured" | "connected" | "failed: <reason>"
_status: str = "not_configured"


def get_status() -> str:
    """Current Google Calendar MCP connection status (see _status)."""
    return _status


def _materialize_google_secrets_from_env() -> None:
    """Recreate the Google Calendar credential + token files from env vars.

    A headless host like Railway has no browser to run Google's interactive
    OAuth consent flow and no easy way to hand-place files on its disk. So we
    let the two files ride in as environment variables and write them back to
    real files here at startup, pointing the path settings at them:

      GOOGLE_OAUTH_CREDENTIALS_JSON   -> the OAuth client credentials JSON
      GOOGLE_CALENDAR_MCP_TOKEN_JSON  -> the token minted by the local `auth` run

    Files are written next to DATABASE_PATH, so when DATABASE_PATH is on a
    persistent Railway volume the token the MCP server refreshes at runtime
    survives redeploys. The token is only written when absent, so a live token
    already maintained on the volume is never clobbered by the (older) seed;
    the credentials file is static, so it's always refreshed. A no-op when
    neither env var is set - local dev keeps using the files it already has."""
    creds_json = os.environ.get("GOOGLE_OAUTH_CREDENTIALS_JSON")
    token_json = os.environ.get("GOOGLE_CALENDAR_MCP_TOKEN_JSON")
    if not creds_json and not token_json:
        return

    db_path = Path(settings.database_path)
    base = db_path.parent if db_path.is_absolute() else Path.cwd()
    base.mkdir(parents=True, exist_ok=True)

    if creds_json:
        creds_path = base / "gcp-oauth.keys.json"
        creds_path.write_text(creds_json)
        creds_path.chmod(0o600)
        settings.google_oauth_credentials_path = str(creds_path)
        logger.info("Wrote Google OAuth credentials from env to %s.", creds_path)

    if token_json:
        token_path = base / "gcp-tokens.json"
        if token_path.exists():
            logger.info("Google Calendar token already present at %s; keeping it.", token_path)
        else:
            token_path.write_text(token_json)
            token_path.chmod(0o600)
            logger.info("Wrote Google Calendar token from env to %s.", token_path)
        settings.google_calendar_mcp_token_path = str(token_path)


def _server_params() -> StdioServerParameters | None:
    if not settings.google_oauth_credentials_path:
        return None
    env = {"GOOGLE_OAUTH_CREDENTIALS": settings.google_oauth_credentials_path}
    # Without this, the MCP server caches OAuth tokens under its own default
    # config dir (e.g. ~/.config/google-calendar-mcp on Linux) - fine
    # locally, but on an ephemeral host that directory doesn't persist
    # across redeploys and there's no browser to redo the interactive OAuth
    # consent flow headlessly. Pointing this at the same persistent volume
    # used for DATABASE_PATH lets a token generated once (locally) keep
    # working after every redeploy.
    if settings.google_calendar_mcp_token_path:
        env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = settings.google_calendar_mcp_token_path
    return StdioServerParameters(
        command="npx",
        args=["-y", "@cocal/google-calendar-mcp"],
        env=env,
    )


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _connect() -> None:
    global _session, _session_cm, _stdio_cm, _status

    params = _server_params()
    if params is None:
        _status = "not_configured"
        logger.info(
            "GOOGLE_OAUTH_CREDENTIALS_PATH not set - skipping Google Calendar MCP server "
            "(calendar features will use the local mock)."
        )
        return

    try:
        _stdio_cm = stdio_client(params)
        read, write = await _stdio_cm.__aenter__()
        _session_cm = ClientSession(read, write)
        _session = await _session_cm.__aenter__()
        await _session.initialize()
        _status = "connected"
        logger.info("Connected to Google Calendar MCP server.")
    except Exception as exc:
        _session = None
        _status = f"failed: {type(exc).__name__}: {exc}"
        # WARNING (not info) and with the reason spelled out, because a silent
        # fallback to the mock is the exact bug this surfacing prevents.
        logger.warning(
            "Could not start Google Calendar MCP server (%s) - calendar features will "
            "fall back to the local mock. Check GOOGLE_OAUTH_CREDENTIALS_PATH and, on a "
            "headless host, GOOGLE_CALENDAR_MCP_TOKEN_PATH.",
            _status,
            exc_info=True,
        )


def start() -> None:
    """Starts the background event loop + MCP connection. Safe to call even
    when MCP isn't configured - it just becomes a no-op in that case."""
    global _loop, _thread

    if _loop is not None:
        return

    _materialize_google_secrets_from_env()

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
