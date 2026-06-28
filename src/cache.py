"""
Semantic Cache with user-feedback loop.

Storage: SQLite for metadata + answer; ChromaDB (separate collection) for query embeddings.

Flow:
  1. On query: embed query → search cache collection for similar past queries
  2. If similarity > threshold AND feedback != negative → return cached answer
  3. After answer generated: store in cache (pending feedback)
  4. On user feedback: update feedback field
"""
import sqlite3, json, os, time, hashlib
from typing import Optional, Dict, Any, Tuple

import chromadb

from src.config import CACHE_DB_PATH, VECTOR_DB_DIR
from src.indexer import SiliconFlowEmbeddingFunction
from src.logger import get_logger

log = get_logger(__name__)

CACHE_COLLECTION   = "query_cache"
CACHE_SIM_THRESHOLD = 0.92   # cosine similarity threshold for cache hit
_embed_fn = None
_cache_col = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = SiliconFlowEmbeddingFunction()
    return _embed_fn


def _get_cache_col():
    global _cache_col
    if _cache_col is None:
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        _cache_col = client.get_or_create_collection(
            CACHE_COLLECTION,
            embedding_function=_get_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _cache_col


def _get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id          TEXT PRIMARY KEY,
            query       TEXT NOT NULL,
            answer      TEXT NOT NULL,
            sources     TEXT,
            feedback    INTEGER DEFAULT 0,  -- 0=none, 1=positive, -1=negative
            created_at  REAL,
            hit_count   INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def cache_lookup(query: str) -> Optional[Dict]:
    """
    Look up query in semantic cache.
    Returns cached entry dict (with 'answer', 'sources') or None.
    """
    try:
        col = _get_cache_col()
    except Exception:
        log.exception("cache_lookup | failed to get collection")
        return None
    try:
        count = col.count()
        if count == 0:
            log.debug("cache_lookup | cache empty")
            return None
    except Exception:
        log.exception("cache_lookup | col.count() failed")
        return None

    try:
        results = col.query(query_texts=[query], n_results=1, include=["distances", "metadatas"])
    except Exception:
        log.exception("cache_lookup | col.query() failed")
        return None

    if not results["ids"][0]:
        log.debug("cache_lookup | no results")
        return None

    dist  = results["distances"][0][0]
    sim   = 1.0 - dist
    if sim < CACHE_SIM_THRESHOLD:
        log.debug("cache_lookup | miss (sim=%.3f < threshold=%.2f)", sim, CACHE_SIM_THRESHOLD)
        return None

    cache_id = results["ids"][0][0]
    meta     = results["metadatas"][0][0]

    conn = _get_db()
    row = conn.execute(
        "SELECT query, answer, sources, feedback, hit_count FROM cache WHERE id=?",
        (cache_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None
    query_orig, answer, sources_json, feedback, hit_count = row

    if feedback == -1:
        log.debug("cache_lookup | skipped (negative feedback) id=%s", cache_id)
        return None

    conn = _get_db()
    conn.execute("UPDATE cache SET hit_count=hit_count+1 WHERE id=?", (cache_id,))
    conn.commit()
    conn.close()

    log.info("cache_lookup | HIT id=%s sim=%.3f hits=%d", cache_id, sim, hit_count + 1)
    return {
        "cache_id":   cache_id,
        "similarity": sim,
        "query_orig": query_orig,
        "answer":     answer,
        "sources":    json.loads(sources_json) if sources_json else [],
        "hit_count":  hit_count + 1,
    }


def cache_store(query: str, answer: str, sources: list = None) -> str:
    """Store a new query-answer pair. Returns cache_id (or empty string on error)."""
    cache_id = hashlib.md5(query.encode()).hexdigest()
    try:
        col = _get_cache_col()
        col.upsert(
            ids=[cache_id],
            documents=[query],
            metadatas=[{"query": query[:200]}],
        )
        conn = _get_db()
        conn.execute("""
            INSERT OR REPLACE INTO cache (id, query, answer, sources, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (cache_id, query, answer, json.dumps(sources or []), time.time()))
        conn.commit()
        conn.close()
        log.info("cache_store | stored id=%s answer_len=%d", cache_id, len(answer))
    except Exception:
        log.exception("cache_store | failed (non-fatal) id=%s", cache_id)
    return cache_id


def cache_feedback(cache_id: str, positive: bool):
    """Record user feedback for a cached answer."""
    val = 1 if positive else -1
    conn = _get_db()
    conn.execute("UPDATE cache SET feedback=? WHERE id=?", (val, cache_id))
    conn.commit()
    conn.close()


def cache_stats() -> Dict:
    """Return cache statistics."""
    conn = _get_db()
    row = conn.execute("""
        SELECT COUNT(*), SUM(hit_count), SUM(feedback=1), SUM(feedback=-1)
        FROM cache
    """).fetchone()
    conn.close()
    total, hits, pos, neg = row
    return {
        "total_entries": total or 0,
        "total_hits":    hits or 0,
        "positive_feedback": pos or 0,
        "negative_feedback": neg or 0,
    }
