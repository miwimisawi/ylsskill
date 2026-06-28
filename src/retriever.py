"""
Hybrid Retrieval: Dense (BGE-M3) + Sparse (BM25) fused with RRF.
"""
import re
from typing import List, Dict, Any, Tuple

from src.config import TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FUSED, RRF_K
from src.indexer import _tokenize


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)


def dense_search(col, query: str, top_k: int = TOP_K_DENSE) -> List[Dict]:
    """Query ChromaDB with dense embedding."""
    results = col.query(
        query_texts=[query],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":        doc,
            "parent_text": meta.get("parent_text", doc),
            "source":      meta.get("source", ""),
            "db":          meta.get("db", ""),
            "score":       1.0 - dist,   # cosine similarity
            "method":      "dense",
        })
    return hits


def sparse_search(bm25, bm25_docs: List[Dict], query: str, top_k: int = TOP_K_SPARSE) -> List[Dict]:
    """BM25 keyword search."""
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    hits = []
    for idx, score in ranked:
        if score <= 0:
            continue
        doc = bm25_docs[idx]
        hits.append({
            "text":        doc["text"],
            "parent_text": doc["parent_text"],
            "source":      doc["source"],
            "db":          doc["db"],
            "score":       float(score),
            "method":      "sparse",
        })
    return hits


def rrf_fusion(
    dense_hits: List[Dict],
    sparse_hits: List[Dict],
    top_k: int = TOP_K_FUSED,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion of two ranked lists.
    Deduplicates by (text snippet), keeps highest-score metadata.
    """
    scores: Dict[str, float] = {}
    data:   Dict[str, Dict]  = {}

    for rank, hit in enumerate(dense_hits):
        key = hit["text"][:120]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        if key not in data:
            data[key] = {**hit, "methods": ["dense"]}
        else:
            data[key]["methods"].append("dense")

    for rank, hit in enumerate(sparse_hits):
        key = hit["text"][:120]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        if key not in data:
            data[key] = {**hit, "methods": ["sparse"]}
        else:
            data[key]["methods"] = list(set(data[key].get("methods", []) + ["sparse"]))

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for key, rrf_score in fused:
        entry = data[key].copy()
        entry["rrf_score"] = rrf_score
        results.append(entry)
    return results


def hybrid_search(
    col, bm25, bm25_docs: List[Dict],
    query: str,
    top_k_final: int = TOP_K_FUSED,
) -> List[Dict]:
    """Full hybrid search pipeline: dense + sparse → RRF fusion."""
    dense_hits  = dense_search(col, query, top_k=TOP_K_DENSE)
    sparse_hits = sparse_search(bm25, bm25_docs, query, top_k=TOP_K_SPARSE)
    fused       = rrf_fusion(dense_hits, sparse_hits, top_k=top_k_final)
    return fused
