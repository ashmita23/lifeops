"""Gradio demo UI for the LifeOps tool-calling agent."""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import mcp_client
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

_CUSTOM_CSS = """
.gradio-container {
    max-width: 760px !important;
    margin: 0 auto !important;
}
#chat-title {
    text-align: center;
    margin-bottom: 0 !important;
}
#chat-subtitle {
    text-align: center;
    opacity: 0.7;
    margin-top: 0 !important;
    margin-bottom: 1.25rem !important;
}
#lifeops-chatbot {
    border-radius: 18px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
}
#lifeops-chatbot .message {
    animation: lifeops-fade-in 0.25s ease-out;
}
@keyframes lifeops-fade-in {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
#lifeops-input textarea {
    border-radius: 999px !important;
}
#lifeops-reset {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#lifeops-reset:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
}
"""


def handle_agent_submit(multimodal_value: dict, session_id: str, history: list, timezone: str = ""):
    handler_start = time.perf_counter()

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

    agent_start = time.perf_counter()
    result = run_agent_turn(session_id=session_id or None, input_text=text, timezone=timezone)
    agent_latency_ms = (time.perf_counter() - agent_start) * 1000

    postprocess_start = time.perf_counter()
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": result.message},
    ]
    postprocess_latency_ms = (time.perf_counter() - postprocess_start) * 1000

    # Trace export runs in the background executor, not inline - it can take
    # several seconds waiting on Langfuse to finish indexing, and the chat
    # reply should never be held up by that. Failures are logged, not raised.
    _EXPORT_EXECUTOR.submit(_export_trace_background, result.trace_id, agent_latency_ms / 1000.0)

    total_handler_latency_ms = (time.perf_counter() - handler_start) * 1000
    logger.info(
        "chat turn latency: agent=%.1fms postprocess=%.1fms total=%.1fms",
        agent_latency_ms,
        postprocess_latency_ms,
        total_handler_latency_ms,
    )

    return history, result.session_id


def handle_agent_reset():
    return [], ""


def main() -> None:
    # Side-effecting startup (DB init, MCP subprocess, real UI construction)
    # lives here, not at module level, so importing this module (e.g. from
    # tests) never triggers it - only actually running it as a script does.
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

    with gr.Blocks(title="LifeOps Agent") as demo:
        gr.Markdown("# ✨ LifeOps Agent", elem_id="chat-title")
        gr.Markdown(
            "Type or record a message - e.g. *\"remind me to call mom tomorrow at 5pm\"*.",
            elem_id="chat-subtitle",
        )

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

        agent_input.submit(
            fn=handle_agent_submit,
            inputs=[agent_input, session_state, chat, tz_state],
            outputs=[chat, session_state],
        ).then(lambda: {"text": "", "files": []}, outputs=agent_input)

        agent_reset_btn.click(fn=handle_agent_reset, outputs=[chat, session_state])

    # Railway (and most container hosts) assign a dynamic port via $PORT and
    # expect the app to bind 0.0.0.0, not localhost. Defaults match local dev.
    demo.launch(
        theme=gr.themes.Ocean(),
        css=_CUSTOM_CSS,
        server_name=os.environ.get("SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", 7860)),
        auth=settings.gradio_auth,
    )


if __name__ == "__main__":
    main()
