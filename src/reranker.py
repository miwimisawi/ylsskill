"""
Two-stage reranking:
  1. BGE-Reranker via SiliconFlow rerank API (Top-N → Top-K)
  2. Contextual compression (strip irrelevant sentences)
"""
from typing import List, Dict, Any
import re, time
import requests

from src.config import (
    RERANK_MODEL, TOP_K_RERANK,
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL,
)

RERANK_URL = SILICONFLOW_BASE_URL.rstrip("/") + "/rerank"


def rerank(query: str, hits: List[Dict], top_k: int = TOP_K_RERANK) -> List[Dict]:
    """
    API-based reranking with SiliconFlow BAAI/bge-reranker-v2-m3.
    Returns top_k hits sorted by rerank score, with 'rerank_score' field added.
    Falls back to original order on API failure.
    """
    if not hits:
        return []

    documents = [h["text"] for h in hits]
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(RERANK_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 2:
                # Fallback: return hits as-is with dummy scores
                result = []
                for i, h in enumerate(hits[:top_k]):
                    h = h.copy()
                    h["rerank_score"] = 1.0 / (i + 1)
                    result.append(h)
                return result
            time.sleep(2 ** attempt)

    results = data.get("results", [])
    hits_with_scores = []
    for r in results:
        idx = r["index"]
        score = r["relevance_score"]
        h = hits[idx].copy()
        h["rerank_score"] = float(score)
        hits_with_scores.append(h)

    return sorted(hits_with_scores, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def compress_context(query: str, hits: List[Dict]) -> List[Dict]:
    """
    Contextual compression: for each hit's parent_text, keep only
    sentences that are relevant to the query (simple keyword overlap).
    Reduces tokens sent to LLM.
    """
    query_tokens = set(re.findall(r'[一-鿿]|[a-zA-Z]{3,}', query.lower()))

    compressed = []
    for hit in hits:
        parent = hit.get("parent_text", hit["text"])
        sentences = re.split(r'(?<=[。！？.!?])\s*', parent)
        kept = []
        for sent in sentences:
            sent_tokens = set(re.findall(r'[一-鿿]|[a-zA-Z]{3,}', sent.lower()))
            if len(query_tokens & sent_tokens) > 0 or len(sent) < 60:
                kept.append(sent)
        compressed_text = " ".join(kept) if kept else parent[:500]
        h = hit.copy()
        h["context"] = compressed_text
        compressed.append(h)
    return compressed
