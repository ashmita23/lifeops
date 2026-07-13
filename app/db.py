"""SQLite storage layer. Plain sqlite3, no ORM, to keep the MVP simple."""

import sqlite3
from contextlib import contextmanager

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    due_date TEXT,
    priority TEXT,
    completed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    raw_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT NOT NULL,
    mood TEXT,
    tags TEXT,
    created_at TEXT NOT NULL,
    raw_text TEXT NOT NULL
);

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
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
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
