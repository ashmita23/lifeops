"""Google OAuth 2.0 helpers for per-user "Sign in with Google".

The web layer (app/web.py) drives the Authorization Code flow:
  1. build_authorization_url() -> send the user to Google's consent screen.
  2. Google redirects back to OAUTH_REDIRECT_URI with a `code`.
  3. exchange_code() -> trade the code for tokens (incl. a refresh token,
     thanks to access_type=offline + prompt=consent).
  4. fetch_userinfo() -> the signed-in user's stable id / email / name.

One consent grants BOTH identity (openid/email/profile) and calendar access,
so the same refresh token later drives the user's own calendar (app/tokens.py,
app/google_calendar.py). Uses httpx (already a dependency); no Google SDK.
"""

import logging
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.db import connection_scope

logger = logging.getLogger(__name__)

_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"

# openid/email/profile identify the user; calendar is requested up front so the
# single consent also covers the calendar features (used from Phase 4 on).
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar",
]


def build_authorization_url(state: str) -> str:
    """URL of Google's consent screen. `state` is an opaque anti-CSRF token the
    caller stores in the session and re-checks on callback."""
    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": settings.oauth_redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        # offline + consent are what make Google return a *refresh* token (not
        # just a one-hour access token) so we can act on the user's calendar
        # later without them re-approving each time.
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return str(httpx.URL(_AUTH_ENDPOINT, params=params))


async def exchange_code(code: str) -> dict:
    """Trade an authorization `code` for tokens. Returns Google's token JSON
    (access_token, refresh_token, expires_in, scope, id_token, ...)."""
    data = {
        "code": code,
        "client_id": settings.google_oauth_client_id,
        "client_secret": settings.google_oauth_client_secret,
        "redirect_uri": settings.oauth_redirect_uri,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(_TOKEN_ENDPOINT, data=data)
        resp.raise_for_status()
        return resp.json()


async def fetch_userinfo(access_token: str) -> dict:
    """The signed-in user's profile: {sub, email, name, ...}."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            _USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()


def upsert_user(user_id: str, email: str | None, name: str | None) -> None:
    """Create the user row on first login; refresh profile + last_login_at on
    subsequent logins. user_id is the Google `sub`."""
    now = datetime.now(timezone.utc).isoformat()
    with connection_scope() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, email, name, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                email = excluded.email,
                name = excluded.name,
                last_login_at = excluded.last_login_at
            """,
            (user_id, email, name, now, now),
        )


def get_user(user_id: str) -> dict | None:
    with connection_scope() as conn:
        row = conn.execute(
            "SELECT user_id, email, name, created_at, last_login_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None
