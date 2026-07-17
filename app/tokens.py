"""Per-user Google credential storage + access-token minting.

A user's Google **refresh token** (obtained once at consent, app/auth.py) is a
long-lived secret that lets us act on their calendar without re-prompting. We
store it ENCRYPTED at rest (Fernet) in the google_credentials table, and mint
short-lived access tokens from it on demand (used by app/google_calendar.py in
Phase 4).

Encryption key: TOKEN_ENCRYPTION_KEY if set (a Fernet key from
Fernet.generate_key()); otherwise a stable key derived from SESSION_SECRET, so
tokens are never written in plaintext even if the explicit key is missing.
Setting TOKEN_ENCRYPTION_KEY explicitly is strongly recommended in production.
"""

import base64
import hashlib
import logging
from datetime import datetime, timezone

import httpx
from cryptography.fernet import Fernet

from app.config import settings
from app.db import connection_scope

logger = logging.getLogger(__name__)

_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


def _fernet() -> Fernet:
    key = settings.token_encryption_key
    if not key:
        # Derive a valid 32-byte urlsafe-base64 Fernet key from the session
        # secret so refresh tokens are still encrypted when TOKEN_ENCRYPTION_KEY
        # isn't configured. Deterministic, so it survives restarts.
        key = base64.urlsafe_b64encode(hashlib.sha256(settings.session_secret.encode()).digest())
    return Fernet(key if isinstance(key, bytes) else key.encode())


def store_credentials(user_id: str, refresh_token: str | None, scope: str | None = None) -> None:
    """Persist (encrypted) a user's Google refresh token. No-op if Google didn't
    return one (it omits it on some re-consents) so we never clobber a good
    stored token with nothing."""
    if not refresh_token:
        logger.info("No refresh_token returned for user %s; keeping any existing one.", user_id)
        return
    token = _fernet().encrypt(refresh_token.encode()).decode()
    now = datetime.now(timezone.utc).isoformat()
    with connection_scope() as conn:
        conn.execute(
            """
            INSERT INTO google_credentials (user_id, refresh_token, scope, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                refresh_token = excluded.refresh_token,
                scope = excluded.scope,
                updated_at = excluded.updated_at
            """,
            (user_id, token, scope, now),
        )


def has_calendar_credentials(user_id: str) -> bool:
    with connection_scope() as conn:
        row = conn.execute(
            "SELECT 1 FROM google_credentials WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row is not None


def get_refresh_token(user_id: str) -> str | None:
    with connection_scope() as conn:
        row = conn.execute(
            "SELECT refresh_token FROM google_credentials WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    return _fernet().decrypt(row["refresh_token"].encode()).decode()


def delete_credentials(user_id: str) -> None:
    """Forget a user's calendar grant (used by the 'disconnect' affordance)."""
    with connection_scope() as conn:
        conn.execute("DELETE FROM google_credentials WHERE user_id = ?", (user_id,))


class CalendarAuthError(Exception):
    """Raised when a user has no stored token or Google rejects the refresh
    (revoked / expired). Callers should prompt the user to reconnect."""


def get_access_token(user_id: str) -> str:
    """Mint a fresh short-lived access token from the user's stored refresh
    token. Synchronous: called from the agent's (sync) tool dispatch. Raises
    CalendarAuthError if there's nothing stored or Google refuses (e.g. the
    user revoked access)."""
    refresh_token = get_refresh_token(user_id)
    if not refresh_token:
        raise CalendarAuthError(f"No calendar credentials stored for user {user_id}.")
    data = {
        "client_id": settings.google_oauth_client_id,
        "client_secret": settings.google_oauth_client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(_TOKEN_ENDPOINT, data=data)
    if resp.status_code != 200:
        raise CalendarAuthError(
            f"Google refused to refresh the token for user {user_id}: "
            f"{resp.status_code} {resp.text[:200]}"
        )
    return resp.json()["access_token"]
