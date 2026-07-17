"""Vector index over journal entries, backed by Pinecone (managed serverless).

This is the "retrieval" half of RAG: journal entries are embedded and upserted
here on creation; retrieve() embeds a query and returns the most similar past
entries, which the agent then grounds its answer in.

Everything degrades to a no-op when Pinecone isn't configured (settings.
rag_enabled is False): retrieve() returns [], index_entry() silently skips. So
the app and the test suite run fine without a Pinecone key - same graceful
degradation as the MCP calendar. The `search_journal` tool built on top reports
"unavailable" in that case.
"""

import logging

from app.config import settings
from app.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 1536  # text-embedding-3-small
_index = None  # cached Pinecone Index handle


def _get_index():
    """Lazily create/connect the Pinecone index, cached for the process.
    Returns None if Pinecone isn't configured or setup fails (RAG then no-ops)."""
    global _index
    if _index is not None:
        return _index
    if not settings.rag_enabled:
        return None
    try:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = {i["name"] for i in pc.list_indexes()}
        if settings.pinecone_index_name not in existing:
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=_EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        _index = pc.Index(settings.pinecone_index_name)
        return _index
    except Exception:
        logger.warning("Could not initialize Pinecone; journal RAG disabled.", exc_info=True)
        return None


def index_entry(entry_id: int, text: str) -> None:
    """Embed and upsert one journal entry. No-op if RAG is unavailable. Upsert
    keyed by id, so re-indexing the same entry is idempotent. Tags the vector
    with the current user so retrieve() can filter to that user's entries."""
    from app.db import current_user_id

    index = _get_index()
    if index is None or not text:
        return
    vector = embed_query(text)
    index.upsert([(str(entry_id), vector, {"content": text, "user_id": current_user_id()})])


def backfill(entries: list[dict]) -> int:
    """Index a list of journal rows (dicts with id + content). One-off/idempotent.
    Returns how many were indexed. No-op (returns 0) if RAG is unavailable."""
    index = _get_index()
    if index is None:
        return 0
    rows = [(e["id"], e.get("content") or "") for e in entries if (e.get("content") or "")]
    if not rows:
        return 0
    vectors = embed_texts([content for _, content in rows])
    index.upsert([(str(rid), vec, {"content": content}) for (rid, content), vec in zip(rows, vectors)])
    return len(rows)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return up to k journal entries most similar to `query`, each as
    {id, content, score}. Returns [] immediately when RAG is unavailable -
    importantly BEFORE any embedding call, so offline tests never hit the
    network."""
    from app.db import current_user_id

    index = _get_index()
    if index is None or not query:
        return []
    vector = embed_query(query)
    # Filter to the current user's own entries so RAG never surfaces another
    # user's journal.
    result = index.query(
        vector=vector, top_k=k, include_metadata=True,
        filter={"user_id": {"$eq": current_user_id()}},
    )
    matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
    out = []
    for m in matches:
        mid = m["id"] if isinstance(m, dict) else m.id
        score = m["score"] if isinstance(m, dict) else m.score
        meta = (m.get("metadata") if isinstance(m, dict) else m.metadata) or {}
        out.append({"id": int(mid), "content": meta.get("content", ""), "score": score})
    return out
