"""
Full RAG Pipeline: Orchestrates all components.

Flow:
  1. Cache lookup → hit: return cached answer
  2. Intent detection → template request: exact match from db6
  3. Query enhancement: HyDE / Multi-Query / Step-back
  4. Hybrid retrieval (dense + BM25 + RRF)
  5. Reranking + contextual compression
  6. CRAG quality evaluation
  7. LLM generation
  8. Store in cache
"""
import time
from typing import Dict, Any, Optional, List

from src.cache     import cache_lookup, cache_store, cache_feedback
from src.retriever import hybrid_search
from src.enhancer  import hyde, multi_query, step_back
from src.reranker  import rerank, compress_context
from src.crag      import evaluate_retrieval, build_context, CONFIDENCE_LOW
from src.generator import generate
from src.config    import TOP_K_RERANK


def _is_template_request(query: str) -> bool:
    """Heuristic: detect document template requests."""
    keywords = ["模板", "文书", "病程", "知情同意", "手术记录", "首程", "术前", "出院小结"]
    return any(k in query for k in keywords)


def run(
    query: str,
    col,
    bm25,
    bm25_docs: List[Dict],
    provider: str = "siliconflow",
    model: Optional[str] = None,
    use_hyde: bool = True,
    use_multi_query: bool = False,  # expensive: 3x LLM calls for retrieval
    use_step_back: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the full pipeline.

    Returns dict with:
      answer, sources, cache_id, confidence, debug_info (if debug=True)
    """
    t0 = time.time()
    debug_info: Dict[str, Any] = {"timings": {}}

    # ── 1. Cache lookup ────────────────────────────────────────────────────
    t1 = time.time()
    cached = cache_lookup(query)
    debug_info["timings"]["cache_lookup"] = round(time.time() - t1, 3)

    if cached:
        if debug:
            print(f"[CACHE HIT] similarity={cached['similarity']:.3f}, hits={cached['hit_count']}")
        return {
            "answer":     cached["answer"],
            "sources":    cached["sources"],
            "cache_id":   cached["cache_id"],
            "confidence": "cached",
            "from_cache": True,
            "debug_info": debug_info if debug else {},
        }

    if debug:
        print("[CACHE MISS] Proceeding to retrieval...")

    # ── 2. Template fast path ──────────────────────────────────────────────
    if _is_template_request(query):
        # Restrict search to db6
        pass  # handled by metadata filtering below (future enhancement)

    # ── 3. Query enhancement ───────────────────────────────────────────────
    t1 = time.time()
    search_queries = [query]

    if use_hyde:
        hyde_passage = hyde(query)
        search_queries.append(hyde_passage)
        if debug:
            print(f"[HyDE] Generated: {hyde_passage[:120]}...")

    if use_step_back:
        sb_query = step_back(query)
        search_queries.append(sb_query)
        if debug:
            print(f"[Step-back] {sb_query}")

    if use_multi_query:
        extra = multi_query(query, n=2)
        search_queries.extend(extra[1:])  # skip original
        if debug:
            print(f"[Multi-Query] extras: {extra[1:]}")

    debug_info["timings"]["enhancement"] = round(time.time() - t1, 3)
    debug_info["search_queries"] = search_queries

    # ── 4. Hybrid retrieval (for each search query, fuse results) ──────────
    t1 = time.time()
    all_hits: Dict[str, Dict] = {}
    for q in search_queries:
        hits = hybrid_search(col, bm25, bm25_docs, q)
        for h in hits:
            key = h["text"][:120]
            if key not in all_hits or h.get("rrf_score", 0) > all_hits[key].get("rrf_score", 0):
                all_hits[key] = h

    merged_hits = sorted(all_hits.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)[:40]
    debug_info["timings"]["retrieval"] = round(time.time() - t1, 3)
    debug_info["retrieval_count"] = len(merged_hits)

    if debug:
        print(f"[Retrieval] {len(merged_hits)} unique hits from {len(search_queries)} queries")

    # ── 5. Reranking + compression ─────────────────────────────────────────
    t1 = time.time()
    reranked = rerank(query, merged_hits, top_k=TOP_K_RERANK)
    compressed = compress_context(query, reranked)
    debug_info["timings"]["reranking"] = round(time.time() - t1, 3)

    if debug:
        print(f"[Rerank] Top scores: {[round(h.get('rerank_score',0),3) for h in reranked[:5]]}")

    # ── 6. CRAG quality check ──────────────────────────────────────────────
    confidence, best_score = evaluate_retrieval(query, reranked)
    debug_info["crag_confidence"] = confidence
    debug_info["crag_best_score"] = round(best_score, 4)

    if debug:
        print(f"[CRAG] confidence={confidence}, best_score={best_score:.4f}")

    context = build_context(compressed, confidence)

    # ── 7. LLM generation ──────────────────────────────────────────────────
    t1 = time.time()
    answer = generate(query, context, provider=provider, model=model, stream=debug)
    debug_info["timings"]["generation"] = round(time.time() - t1, 3)

    # ── 8. Cache store ─────────────────────────────────────────────────────
    sources = [{"db": h.get("db",""), "source": h.get("source",""),
                "score": round(h.get("rerank_score",0), 3)} for h in reranked[:5]]
    cache_id = cache_store(query, answer, sources)

    debug_info["timings"]["total"] = round(time.time() - t0, 3)

    return {
        "answer":     answer,
        "sources":    sources,
        "cache_id":   cache_id,
        "confidence": confidence,
        "from_cache": False,
        "debug_info": debug_info if debug else {},
    }
