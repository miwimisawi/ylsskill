"""
CRAG (Corrective RAG): Evaluate retrieval quality and decide action.

Confidence levels:
  HIGH   (>= threshold) → use retrieved docs as-is
  LOW    (< threshold)  → supplement with web/LLM knowledge
  MEDIUM (borderline)   → merge local + supplemental

Relevance scoring uses the reranker's top score as proxy.
"""
from typing import List, Dict, Tuple
from src.config import CRAG_RELEVANCE_THRESHOLD


CONFIDENCE_HIGH   = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW    = "low"


def evaluate_retrieval(
    query: str,
    hits: List[Dict],
    threshold: float = CRAG_RELEVANCE_THRESHOLD,
) -> Tuple[str, float]:
    """
    Assess retrieval quality.

    Returns (confidence_level, best_score).
    Uses 'rerank_score' if present, else 'rrf_score'.
    """
    if not hits:
        return CONFIDENCE_LOW, 0.0

    # Get best relevance signal
    scores = []
    for h in hits:
        if "rerank_score" in h:
            scores.append(h["rerank_score"])
        elif "rrf_score" in h:
            scores.append(h["rrf_score"])
        elif "score" in h:
            scores.append(h["score"])

    best = max(scores) if scores else 0.0

    # Normalize rerank scores (logistic sigmoid) if they look like raw logits
    if best > 3.0 or best < -3.0:
        import math
        best_norm = 1.0 / (1.0 + math.exp(-best))
    else:
        best_norm = best

    if best_norm >= threshold * 2:
        level = CONFIDENCE_HIGH
    elif best_norm >= threshold:
        level = CONFIDENCE_MEDIUM
    else:
        level = CONFIDENCE_LOW

    return level, best_norm


def build_context(
    hits: List[Dict],
    confidence: str,
    supplemental: str = "",
    max_chars: int = 6000,
) -> str:
    """
    Build the final context string for the LLM.

    - HIGH: top hits' context only
    - MEDIUM: local hits + supplemental note
    - LOW: supplemental only (or warning)
    """
    local_parts = []
    for i, h in enumerate(hits):
        text = h.get("context", h.get("parent_text", h["text"]))
        src = f"[{h.get('db','?')}] {h.get('source','')}"
        local_parts.append(f"【参考{i+1} | {src}】\n{text}")

    local_ctx = "\n\n".join(local_parts)

    if confidence == CONFIDENCE_HIGH:
        ctx = local_ctx
    elif confidence == CONFIDENCE_MEDIUM:
        ctx = local_ctx
        if supplemental:
            ctx += f"\n\n【补充信息】\n{supplemental}"
    else:  # LOW
        if supplemental:
            ctx = f"【注意：本地知识库相关度较低，以下为补充信息】\n{supplemental}"
        else:
            ctx = (
                local_ctx + "\n\n【警告：检索到的内容相关度较低，请谨慎参考。"
                "建议向上级医生确认。】"
            )

    return ctx[:max_chars]
