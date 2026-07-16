"""SQLite storage layer. Plain sqlite3, no ORM, to keep the MVP simple."""

import contextvars
import sqlite3
from contextlib import contextmanager

from app.config import settings

# Bucket for data created when no one is signed in (single-user local dev / the
# pre-multi-user rows). In multi-user mode the value is the Google `sub`.
DEFAULT_USER_ID = "local"

# Request-scoped current user, set by run_agent_turn via user_scope(). The
# per-user data tools (reminders, journal, journal RAG) read current_user_id()
# to scope their queries, so their call signatures - and the agent dispatchers
# that call them - don't all have to thread user_id through by hand.
_user_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("lifeops_user_id", default=None)


@contextmanager
def user_scope(user_id: str | None):
    token = _user_ctx.set(user_id)
    try:
        yield
    finally:
        _user_ctx.reset(token)


def current_user_id() -> str:
    """The signed-in user's id for the current turn, or DEFAULT_USER_ID in
    single-user mode. Every per-user row is created and read under this key."""
    return _user_ctx.get() or DEFAULT_USER_ID

_SCHEMA = """
-- One row per person who has signed in with Google. user_id is the Google
-- account's stable `sub` claim (never the email, which can change). All
-- per-user data (reminders, journal, calendar credentials) keys off this.
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT,
    name TEXT,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);

-- A user's Google Calendar grant. refresh_token is a long-lived secret and is
-- stored ENCRYPTED (Fernet) - see app/tokens.py. One row per user; replaced on
-- re-consent. Kept separate from `users` so a user can exist (signed in) with
-- no calendar connected, and so the secret table can be locked down on its own.
CREATE TABLE IF NOT EXISTS google_credentials (
    user_id TEXT PRIMARY KEY,
    refresh_token TEXT NOT NULL,
    scope TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'local',
    title TEXT NOT NULL,
    description TEXT,
    due_date TEXT,
    priority TEXT,
    completed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    raw_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders (user_id);

CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'local',
    title TEXT,
    content TEXT NOT NULL,
    mood TEXT,
    tags TEXT,
    created_at TEXT NOT NULL,
    raw_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_journal_entries_user ON journal_entries (user_id);

CREATE TABLE IF NOT EXISTS calendar_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    start_time TEXT,
    end_time TEXT,
    duration_minutes INTEGER,
    created_at TEXT NOT NULL,
    raw_text TEXT NOT NULL
);

-- Append-only conversation log: ONE row per message, never rewritten.
-- Replaces the old single-blob-per-session `conversations` table. Because a
-- new message is an INSERT (atomic) rather than a read-modify-write of the
-- whole history, two concurrent turns on the same session can't clobber each
-- other's messages (a "lost update"). Ordering within a session is by the
-- autoincrement id = insertion order. See app/session_store.py.
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
    ON conversation_messages (session_id, id);

-- Legacy blob table, kept only so _migrate() can copy old rows into the new
-- append-only table on first run. Nothing writes to it anymore.
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT PRIMARY KEY,
    messages TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_confirmations (
    session_id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Per-session token/cost/error accounting (see app/budget.py). Backs the
-- budget cap and the observability surface. Written once per LLM call.
CREATE TABLE IF NOT EXISTS session_usage (
    session_id TEXT PRIMARY KEY,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    call_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    # WAL (Write-Ahead Logging): in SQLite's default rollback-journal mode a
    # writer blocks all readers and vice versa. WAL lets readers keep reading
    # while one writer writes - a big concurrency win for one PRAGMA. The
    # setting is stored in the database file itself and persists across
    # connections, but setting it here is idempotent and cheap.
    conn.execute("PRAGMA journal_mode=WAL")
    # WAL allows many readers + ONE writer, but two concurrent writers still
    # contend for the write lock. Without this, the loser gets an immediate
    # "database is locked" error; busy_timeout makes it wait up to 5s for the
    # lock instead - so concurrent writers queue rather than fail.
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def connection_scope():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection_scope() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    # CREATE TABLE IF NOT EXISTS doesn't alter tables created before this
    # column existed, so backfill it for any pre-existing lifeops.db file.
    try:
        conn.execute("ALTER TABLE reminders ADD COLUMN completed INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Multi-user scoping: add user_id to the two previously-global tables. The
    # DEFAULT 'local' also backfills existing rows in one step, so a single-user
    # DB's data lands in the DEFAULT_USER_ID bucket and keeps working.
    for table in ("reminders", "journal_entries"):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'")
        except sqlite3.OperationalError:
            pass  # column already exists

    _migrate_conversations_to_rows(conn)


def _migrate_conversations_to_rows(conn: sqlite3.Connection) -> None:
    """One-time copy of the old single-blob-per-session `conversations` table
    into the new append-only `conversation_messages` table. Best-effort and
    idempotent: only runs when the new table is empty and the old one has
    data, and never raises (a migration failure must not block startup)."""
    import json
    from datetime import datetime, timezone

    try:
        already = conn.execute("SELECT COUNT(*) AS n FROM conversation_messages").fetchone()["n"]
        if already:
            return  # new table already populated - nothing to migrate

        old_rows = conn.execute("SELECT session_id, messages, updated_at FROM conversations").fetchall()
        for row in old_rows:
            try:
                messages = json.loads(row["messages"])
            except (json.JSONDecodeError, TypeError):
                continue
            created = row["updated_at"] or datetime.now(timezone.utc).isoformat()
            for message in messages:
                conn.execute(
                    "INSERT INTO conversation_messages (session_id, message, created_at) VALUES (?, ?, ?)",
                    (row["session_id"], json.dumps(message), created),
                )
    except sqlite3.OperationalError:
        pass  # old table doesn't exist / schema mismatch - safe to skip
