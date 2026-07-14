"""Concurrency guarantees for the append-only conversation log.

The point of storing one row per message (instead of one rewritten blob per
session) is that concurrent turns can't lose each other's messages. These
tests exercise that directly.
"""

import threading

import pytest

from app import config, db, session_store


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_session_store.db"
    monkeypatch.setattr(config.settings, "database_path", str(db_path))
    db.init_db()
    yield


def test_append_then_read_preserves_order():
    session_store.append_messages("s1", [{"role": "system", "content": "sys"}])
    session_store.append_messages("s1", [{"role": "user", "content": "hi"}])
    session_store.append_messages("s1", [{"role": "assistant", "content": "hello"}])

    messages = session_store.get_session_messages("s1")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_empty_session_reads_as_none():
    assert session_store.get_session_messages("does-not-exist") is None


def test_concurrent_appends_lose_nothing():
    """The regression guard for the lost-update race. Many threads append to
    the SAME session at once; with the old read-modify-write blob some writes
    would clobber others. With append-only INSERTs, every message must land."""
    session_id = "hot-session"
    writers = 20
    per_writer = 5

    def worker(writer_id: int) -> None:
        for i in range(per_writer):
            session_store.append_messages(
                session_id, [{"role": "user", "content": f"w{writer_id}-m{i}"}]
            )

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    messages = session_store.get_session_messages(session_id)
    contents = {m["content"] for m in messages}

    # Every single message from every writer is present - nothing lost.
    assert len(messages) == writers * per_writer
    expected = {f"w{w}-m{i}" for w in range(writers) for i in range(per_writer)}
    assert contents == expected
