"""FastAPI wrapper that adds per-user "Sign in with Google" around the Gradio UI.

Pure Gradio only supports a single shared username/password; it can't run an
OAuth redirect flow. So we mount the existing Gradio Blocks inside a small
FastAPI app that owns the login routes and a signed session cookie, then gate
access to the UI behind a Google login.

Two modes, chosen by settings.google_login_enabled:
  - configured  -> multi-user: every request must carry a logged-in session;
                   the logged-in user id reaches Gradio handlers via
                   gr.Request.username (populated by the auth_dependency).
  - unconfigured -> single-user: the UI is mounted with no login at all, so
                    local dev and tests behave exactly as before.

The entrypoint (space_app.py) serves the object returned by create_app().
"""

import logging
import secrets

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app import auth, mcp_client
from app.config import settings
from app.db import init_db
from app.tracing import init_tracing
from frontend.gradio_app import build_demo

logger = logging.getLogger(__name__)

# Paths that must stay reachable without a login, or the login flow can't run.
_PUBLIC_PATHS = {"/login", "/oauth2callback", "/logout", "/health"}


def _session_user_id(request: Request) -> str | None:
    """auth_dependency for the mounted Gradio app: the logged-in user's id, or
    None. Gradio copies the returned value onto gr.Request.username (which,
    unlike the raw request, survives the event queue), so handlers read the
    current user from there."""
    user = request.session.get("user") if "session" in request.scope else None
    return user.get("id") if user else None


async def _require_login(request: Request, call_next):
    """Gate every non-public path behind a login. Browser navigations get a
    friendly redirect to /login; programmatic calls get a 401. Runs only in
    multi-user mode (see create_app)."""
    path = request.url.path
    if path in _PUBLIC_PATHS or not request.session.get("user"):
        if path in _PUBLIC_PATHS:
            return await call_next(request)
        accept = request.headers.get("accept", "")
        if request.method == "GET" and "text/html" in accept:
            return RedirectResponse("/login")
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    return await call_next(request)


def create_app() -> FastAPI:
    init_tracing()
    init_db()
    # Global single-user MCP calendar still runs (phases 1-3 keep using it);
    # per-user real calendars replace it in Phase 4. Harmless no-op when the
    # calendar isn't configured.
    mcp_client.start()

    app = FastAPI()
    demo = build_demo()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    if not settings.google_login_enabled:
        logger.warning(
            "GOOGLE_OAUTH_CLIENT_ID/SECRET not set - running single-user with NO login. "
            "Set them (a Web OAuth client) to enable per-user 'Sign in with Google'."
        )
        gr.mount_gradio_app(app, demo, path="/")
        return app

    @app.get("/login")
    def login(request: Request):
        # Opaque anti-CSRF token echoed back by Google and re-checked on callback.
        state = secrets.token_urlsafe(24)
        request.session["oauth_state"] = state
        return RedirectResponse(auth.build_authorization_url(state))

    @app.get("/oauth2callback")
    async def oauth2callback(request: Request):
        if request.query_params.get("state") != request.session.get("oauth_state"):
            return JSONResponse({"error": "state mismatch"}, status_code=400)
        request.session.pop("oauth_state", None)
        code = request.query_params.get("code")
        if not code:
            return JSONResponse({"error": "missing code"}, status_code=400)

        tokens = await auth.exchange_code(code)
        info = await auth.fetch_userinfo(tokens["access_token"])
        user_id = info["sub"]
        auth.upsert_user(user_id, info.get("email"), info.get("name"))
        request.session["user"] = {
            "id": user_id,
            "email": info.get("email"),
            "name": info.get("name"),
        }
        # Phase 2 stores tokens["refresh_token"] (encrypted) here.
        logger.info("User signed in: %s", info.get("email"))
        return RedirectResponse("/")

    @app.get("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login")

    app.add_middleware(BaseHTTPMiddleware, dispatch=_require_login)
    # SessionMiddleware is added last so it is the OUTERMOST layer: it decodes
    # request.session before _require_login (and the routes) read it.
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret, same_site="lax")

    gr.mount_gradio_app(app, demo, path="/", auth_dependency=_session_user_id)
    return app
