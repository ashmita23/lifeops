"""Gradio demo UI for the LifeOps tool-calling agent."""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import auth, budget, mcp_client
from app.agent import run_agent_turn
from app.config import settings
from app.db import init_db
from app.llm_client import LLMUnavailableError
from app.tracing import init_tracing
from app.trace_export import export_trace
from app.transcription import transcribe_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# One reused executor for the process lifetime - trace export runs here so
# it never blocks the chat reply. Each export can hold a worker for up to
# ~30s (the completeness-polling budget in app/trace_export.py), so a small
# pool means rapid-fire messages queue up behind earlier exports - 4 gives
# real headroom without being wasteful for a background side-task.
_EXPORT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trace-export")


def _export_trace_background(trace_id: str | None, expected_latency_seconds: float | None = None) -> None:
    start = time.perf_counter()
    try:
        path = export_trace(trace_id, expected_latency_seconds=expected_latency_seconds)
    except Exception:
        logger.exception("Background trace export failed for trace_id=%s", trace_id)
        return
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("trace_export_latency_ms=%.1f trace_id=%s path=%s", elapsed_ms, trace_id, path)

# Applied at launch()/mount time (Gradio 6.0 no longer takes these on Blocks).
UI_THEME = gr.themes.Ocean()

_CUSTOM_CSS = """
:root { color-scheme: dark; }

/* Full-viewport Lovable-style gradient behind everything (navy -> blue ->
   magenta -> warm), pinned so it doesn't scroll. */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background: linear-gradient(180deg,
        #0a0a12 0%, #0f1a4d 20%, #3a2a8f 42%,
        #9c3a86 62%, #e0568f 80%, #f2743f 100%);
}
body { background: transparent !important; }

.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    background: transparent !important;
}
/* Let the gradient show through Gradio's inner panels. */
.gradio-container .main, .gradio-container .wrap, .gradio-container .panel,
.gradio-container .form, .gradio-container .block, .gradio-container .gap,
.gradio-container .container, gradio-app { background: transparent !important; }

#chat-title {
    text-align: center;
    margin-bottom: 0 !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.02em;
    color: #ffffff !important;
    text-shadow: 0 2px 30px rgba(0, 0, 0, 0.35);
}
#chat-subtitle {
    text-align: center;
    color: rgba(255, 255, 255, 0.75) !important;
    margin-top: 0.4rem !important;
    margin-bottom: 1rem !important;
}

/* "Signed in as … · Sign out" strip */
#lifeops-account { margin: 0 0 1rem !important; }
#lifeops-account .account-inner {
    text-align: right;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.7);
}
#lifeops-account b { color: #ffffff; font-weight: 600; }
#lifeops-account a { color: #c9b6ff; text-decoration: none; }
#lifeops-account a:hover { text-decoration: underline; }

/* Chat surface: dark frosted glass */
#lifeops-chatbot {
    border-radius: 22px !important;
    background: rgba(15, 16, 28, 0.55) !important;
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45) !important;
}
#lifeops-chatbot, #lifeops-chatbot * { color: #f3f4f8 !important; }
#lifeops-chatbot sub { color: rgba(255, 255, 255, 0.5) !important; }

#lifeops-chatbot .message, #lifeops-chatbot .bubble {
    animation: lifeops-fade-in 0.25s ease-out;
    border-radius: 16px !important;
}
/* User bubble: accent gradient. Assistant bubble: subtle glass. */
#lifeops-chatbot .user, #lifeops-chatbot [data-testid="user"] {
    background: linear-gradient(135deg, #6d5cff, #c14ba0) !important;
    border: none !important;
}
#lifeops-chatbot .bot, #lifeops-chatbot [data-testid="bot"] {
    background: rgba(255, 255, 255, 0.06) !important;
}
@keyframes lifeops-fade-in {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Input: glowing dark pill */
#lifeops-input {
    border-radius: 999px !important;
    background: rgba(15, 16, 28, 0.6) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
#lifeops-input:focus-within {
    border-color: rgba(155, 120, 255, 0.7) !important;
    box-shadow: 0 0 0 3px rgba(120, 90, 255, 0.25), 0 10px 40px rgba(0, 0, 0, 0.5);
}
#lifeops-input textarea {
    background: transparent !important;
    color: #f3f4f8 !important;
    border-radius: 999px !important;
}
#lifeops-input textarea::placeholder { color: rgba(255, 255, 255, 0.5) !important; }

/* Reset button: subtle glass pill */
#lifeops-reset {
    border-radius: 999px !important;
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    color: #ffffff !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
#lifeops-reset:hover {
    transform: translateY(-1px);
    background: rgba(255, 255, 255, 0.14) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}
"""


def handle_agent_submit(
    multimodal_value: dict,
    session_id: str,
    history: list,
    timezone: str = "",
    request: gr.Request | None = None,
):
    handler_start = time.perf_counter()

    # In multi-user mode app/web.py's auth_dependency puts the signed-in Google
    # user id on gr.Request.username (None in single-user local dev). Threaded
    # into run_agent_turn below so per-user data (reminders, journal, RAG) is
    # scoped to this user.
    user_id = getattr(request, "username", None) if request is not None else None

    # The browser reports its own IANA timezone (see the demo.load JS hook);
    # fall back to a sane default if it's missing so relative dates like
    # "tomorrow at 5pm" resolve in the user's actual timezone, not the
    # server's. Empty rather than None-safe because Gradio passes "".
    timezone = timezone or "America/Chicago"

    text = (multimodal_value or {}).get("text", "") or ""
    files = (multimodal_value or {}).get("files") or []

    if files:
        try:
            text = transcribe_audio(files[0])
        except LLMUnavailableError:
            history = history + [
                {"role": "assistant", "content": "Voice input requires OPENAI_API_KEY to be set."}
            ]
            return history, session_id

    if not text or not text.strip():
        return history, session_id

    # Snapshot cumulative usage before the turn so we can report THIS turn's
    # delta (a turn may make several LLM calls - loop + synthesis).
    usage_before = budget.get_usage(session_id) if session_id else {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    agent_start = time.perf_counter()
    result = run_agent_turn(
        session_id=session_id or None, input_text=text, timezone=timezone, user_id=user_id
    )
    agent_latency_ms = (time.perf_counter() - agent_start) * 1000

    usage_after = budget.get_usage(result.session_id)
    turn_tokens = (
        (usage_after["prompt_tokens"] + usage_after["completion_tokens"])
        - (usage_before["prompt_tokens"] + usage_before["completion_tokens"])
    )
    turn_cost = usage_after["cost_usd"] - usage_before["cost_usd"]
    footer = f"\n\n<sub>⏱ {agent_latency_ms / 1000:.1f}s · {turn_tokens} tok · ${turn_cost:.4f}</sub>"

    postprocess_start = time.perf_counter()
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": result.message + footer},
    ]
    postprocess_latency_ms = (time.perf_counter() - postprocess_start) * 1000

    # Trace export runs in the background executor, not inline - it can take
    # several seconds waiting on Langfuse to finish indexing, and the chat
    # reply should never be held up by that. Failures are logged, not raised.
    _EXPORT_EXECUTOR.submit(_export_trace_background, result.trace_id, agent_latency_ms / 1000.0)

    total_handler_latency_ms = (time.perf_counter() - handler_start) * 1000
    logger.info(
        "chat turn: agent=%.1fms postprocess=%.1fms total=%.1fms tokens=%d cost=$%.4f errors=%d",
        agent_latency_ms,
        postprocess_latency_ms,
        total_handler_latency_ms,
        turn_tokens,
        turn_cost,
        usage_after.get("error_count", 0),
    )

    return history, result.session_id


def handle_agent_reset():
    return [], ""


def _account_bar(request: gr.Request | None = None) -> str:
    """The 'Signed in as … · Sign out' strip. Populated per page load from the
    logged-in user (multi-user mode); empty in single-user local dev, where
    there's no account and /logout doesn't exist."""
    user_id = getattr(request, "username", None) if request is not None else None
    if not user_id:
        return ""
    user = auth.get_user(user_id) or {}
    label = user.get("email") or user.get("name") or "your account"
    return (
        f'<div class="account-inner">Signed in as <b>{label}</b>'
        f' &nbsp;·&nbsp; <a href="/logout">Sign out</a></div>'
    )


def build_demo() -> gr.Blocks:
    """Construct the Gradio UI (no side effects, no server started).

    Kept separate from main() so app/web.py can mount the same UI inside its
    FastAPI "Sign in with Google" wrapper, while main() still launches it
    standalone for local single-user dev. Theme/CSS live on the Blocks so both
    entrypoints render identically.
    """
    # NOTE: theme/css are NOT passed here - Gradio 6.0 moved them off the Blocks
    # constructor to launch()/mount_gradio_app(). They're applied by main() and
    # by app/web.py's mount instead (see UI_THEME / _CUSTOM_CSS).
    with gr.Blocks(title="LifeOps Agent") as demo:
        gr.Markdown("# ✨ LifeOps Agent", elem_id="chat-title")
        gr.Markdown(
            "Type or record a message - e.g. *\"remind me to call mom tomorrow at 5pm\"*.",
            elem_id="chat-subtitle",
        )

        # "Signed in as … · Sign out" strip, only in multi-user mode. Filled on
        # load (below) from the logged-in user.
        account_bar = None
        if settings.google_login_enabled:
            account_bar = gr.HTML("", elem_id="lifeops-account")

        session_state = gr.State("")
        # Hidden field the browser fills with its own IANA timezone on load,
        # so relative dates ("tomorrow at 5pm") resolve in the user's zone
        # instead of the server's. Falls back to a default if JS is blocked.
        tz_state = gr.Textbox(visible=False)
        chat = gr.Chatbot(label=None, show_label=False, elem_id="lifeops-chatbot", height=520)
        agent_input = gr.MultimodalTextbox(
            show_label=False,
            placeholder="Message LifeOps Agent...",
            sources=["microphone", "upload"],
            file_types=["audio"],
            elem_id="lifeops-input",
        )
        agent_reset_btn = gr.Button("New conversation", size="sm", elem_id="lifeops-reset")

        demo.load(
            fn=None,
            inputs=None,
            outputs=tz_state,
            js="() => Intl.DateTimeFormat().resolvedOptions().timeZone",
        )

        if account_bar is not None:
            demo.load(fn=_account_bar, inputs=None, outputs=account_bar)

        agent_input.submit(
            fn=handle_agent_submit,
            inputs=[agent_input, session_state, chat, tz_state],
            outputs=[chat, session_state],
        ).then(lambda: {"text": "", "files": []}, outputs=agent_input)

        agent_reset_btn.click(fn=handle_agent_reset, outputs=[chat, session_state])

    return demo


def main() -> None:
    # Standalone single-user launch for local dev. The multi-user "Sign in with
    # Google" entrypoint is app/web.py (served via space_app.py). Side-effecting
    # startup lives here, not at module level, so importing this module (e.g.
    # from tests) never triggers it.
    init_tracing()
    init_db()
    mcp_client.start()

    mcp_status = mcp_client.get_status()
    if mcp_status == "connected":
        logger.info("Google Calendar: connected via MCP.")
    else:
        logger.warning(
            "Google Calendar: NOT connected (status=%s) - the app is using the local mock "
            "calendar. This is fine locally; in production it means real calendar actions "
            "won't happen. See README 'Google Calendar on Railway'.",
            mcp_status,
        )

    if settings.gradio_auth is None:
        logger.warning(
            "GRADIO_AUTH_USER/GRADIO_AUTH_PASS not set - the app is running with NO login "
            "gate and is fully open to anyone who can reach the URL. Set both env vars to "
            "require a shared password."
        )

    # Railway (and most container hosts) assign a dynamic port via $PORT and
    # expect the app to bind 0.0.0.0, not localhost. Defaults match local dev.
    build_demo().launch(
        theme=UI_THEME,
        css=_CUSTOM_CSS,
        server_name=os.environ.get("SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", 7860)),
        auth=settings.gradio_auth,
    )


if __name__ == "__main__":
    main()
