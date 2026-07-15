"""Tests for journal RAG retrieval. Embeddings and Pinecone are both mocked -
no network, no API key needed."""

import numpy as np
import pytest

from app import config, db, journal_index


class FakeIndex:
    """In-memory stand-in for a Pinecone Index: stores vectors and ranks a
    query by cosine similarity, mimicking .upsert/.query."""

    def __init__(self):
        self.vectors = {}

    def upsert(self, items):
        for id_, vec, meta in items:
            self.vectors[id_] = (list(vec), meta)

    def query(self, vector, top_k, include_metadata=True):
        q = np.asarray(vector, dtype=float)
        scored = []
        for id_, (vec, meta) in self.vectors.items():
            v = np.asarray(vec, dtype=float)
            sim = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            scored.append((sim, id_, meta))
        scored.sort(reverse=True)
        return {"matches": [{"id": i, "score": s, "metadata": m} for s, i, m in scored[:top_k]]}


# Fixed fake embeddings keyed by substring, so ranking is deterministic.
_FAKE = {"cat": [1.0, 0.0, 0.0], "car": [0.9, 0.1, 0.0], "dog": [0.0, 1.0, 0.0]}


def _fake_embed_query(text: str):
    for key, vec in _FAKE.items():
        if key in text.lower():
            return vec
    return [0.0, 0.0, 1.0]


def _fake_embed_texts(texts):
    return [_fake_embed_query(t) for t in texts]


@pytest.fixture(autouse=True)
def reset_and_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "database_path", str(tmp_path / "ji.db"))
    monkeypatch.setattr(config.settings, "openai_api_key", "sk-test")
    db.init_db()
    journal_index._index = None  # clear cached handle between tests
    monkeypatch.setattr(journal_index, "embed_query", _fake_embed_query)
    monkeypatch.setattr(journal_index, "embed_texts", _fake_embed_texts)
    yield
    journal_index._index = None


def _enable_rag_with_fake_index(monkeypatch) -> FakeIndex:
    monkeypatch.setattr(config.settings, "pinecone_api_key", "pc-test")  # makes rag_enabled True
    fake = FakeIndex()
    journal_index._index = fake  # inject; _get_index returns it, bypassing real Pinecone
    return fake


def test_retrieve_ranks_by_similarity_and_respects_k(monkeypatch):
    fake = _enable_rag_with_fake_index(monkeypatch)
    journal_index.index_entry(1, "a cat sat on a mat")
    journal_index.index_entry(2, "my car is fast")
    journal_index.index_entry(3, "a dog barked")

    results = journal_index.retrieve("cat", k=2)
    assert [r["id"] for r in results] == [1, 2]  # cat, then car (closest), dog excluded by k
    assert results[0]["content"] == "a cat sat on a mat"


def test_retrieve_returns_empty_without_calling_embeddings_when_disabled(monkeypatch):
    # RAG off (no pinecone key). retrieve must short-circuit BEFORE embedding.
    monkeypatch.setattr(config.settings, "pinecone_api_key", None)
    journal_index._index = None

    def boom(_):
        raise AssertionError("embed_query must not be called when RAG is disabled")

    monkeypatch.setattr(journal_index, "embed_query", boom)
    assert journal_index.retrieve("anything", k=5) == []


def test_backfill_indexes_all_entries(monkeypatch):
    fake = _enable_rag_with_fake_index(monkeypatch)
    n = journal_index.backfill([{"id": 1, "content": "cat"}, {"id": 2, "content": "dog"}, {"id": 3, "content": ""}])
    assert n == 2  # empty-content entry skipped
    assert set(fake.vectors) == {"1", "2"}


def test_dispatch_create_journal_indexes_it(monkeypatch):
    from app import agent

    fake = _enable_rag_with_fake_index(monkeypatch)
    record = agent._dispatch_create_journal_entry({"content": "felt like a cat today", "title": None}, "raw")
    assert str(record["id"]) in fake.vectors


def test_indexing_failure_does_not_fail_create(monkeypatch):
    from app import agent

    _enable_rag_with_fake_index(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("pinecone down")

    monkeypatch.setattr(journal_index, "index_entry", boom)
    # The write must still succeed even though indexing raises.
    record = agent._dispatch_create_journal_entry({"content": "hello", "title": None}, "raw")
    assert record["id"] is not None
