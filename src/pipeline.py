"""
Full RAG Pipeline: Orchestrates all components.

Flow:
  1. Cache lookup → hit: return cached answer
  2. Query enhancement: HyDE / Multi-Query / Step-back
  3. Hybrid retrieval (dense + BM25 + RRF)
  4. Reranking + contextual compression
  5. CRAG quality evaluation
  6. LLM generation
  7. Store in cache
"""
import time
from typing import Dict, Any, Optional, List, Callable

from src.cache     import cache_lookup, cache_store
from src.retriever import hybrid_search
from src.enhancer  import hyde, multi_query, step_back
from src.reranker  import rerank, compress_context
from src.crag      import evaluate_retrieval, build_context, CONFIDENCE_LOW
from src.generator import generate
from src.config    import TOP_K_RERANK
from src.logger    import get_logger

log = get_logger(__name__)


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
    use_multi_query: bool = False,
    use_step_back: bool = False,
    debug: bool = False,
    emit: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Run the full pipeline.

    emit: optional callable(dict) called with progress/token events:
      {"type": "step",  "step": "...", "progress": 0-100}
      {"type": "token", "text": "..."}

    Returns dict with:
      answer, sources, cache_id, confidence, from_cache, debug_info
    """
    t0 = time.time()
    debug_info: Dict[str, Any] = {"timings": {}}
    log.info("pipeline.run | START query=%r hyde=%s multi=%s stepback=%s model=%s",
             query[:80], use_hyde, use_multi_query, use_step_back, model)

    def _emit(evt: dict):
        if emit:
            try:
                emit(evt)
            except Exception:
                pass

    # ── 1. Cache lookup ────────────────────────────────────────────────────
    _emit({"type": "step", "step": "正在查询缓存…", "progress": 5})
    t1 = time.time()
    cached = cache_lookup(query)
    debug_info["timings"]["cache_lookup"] = round(time.time() - t1, 3)

    if cached:
        log.info("pipeline.run | CACHE HIT sim=%.3f", cached.get("similarity", 0))
        if debug:
            print(f"[CACHE HIT] similarity={cached['similarity']:.3f}, hits={cached['hit_count']}")
        _emit({"type": "step", "step": "命中缓存，正在加载…", "progress": 80})
        # Stream cached answer as tokens so the frontend path is uniform
        answer = cached["answer"]
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            _emit({"type": "token", "text": answer[i:i + chunk_size]})
        return {
            "answer":     answer,
            "sources":    cached["sources"],
            "cache_id":   cached["cache_id"],
            "confidence": "cached",
            "from_cache": True,
            "debug_info": debug_info if debug else {},
        }

    log.info("pipeline.run | CACHE MISS")
    if debug:
        print("[CACHE MISS] Proceeding to retrieval...")

    # ── 2. Query enhancement ───────────────────────────────────────────────
    t1 = time.time()
    search_queries = [query]
    enhancement_progress = 15

    if use_hyde:
        _emit({"type": "step", "step": "正在生成假设文档（HyDE）…", "progress": enhancement_progress})
        hyde_passage = hyde(query)
        search_queries.append(hyde_passage)
        enhancement_progress += 8
        if debug:
            print(f"[HyDE] Generated: {hyde_passage[:120]}...")

    if use_step_back:
        _emit({"type": "step", "step": "正在提炼背景问题…", "progress": enhancement_progress})
        sb_query = step_back(query)
        search_queries.append(sb_query)
        enhancement_progress += 8
        if debug:
            print(f"[Step-back] {sb_query}")

    if use_multi_query:
        _emit({"type": "step", "step": "正在扩展检索视角…", "progress": enhancement_progress})
        extra = multi_query(query, n=2)
        search_queries.extend(extra[1:])
        if debug:
            print(f"[Multi-Query] extras: {extra[1:]}")

    debug_info["timings"]["enhancement"] = round(time.time() - t1, 3)
    debug_info["search_queries"] = search_queries
    log.info("pipeline.run | enhancement done: %d queries in %.2fs",
             len(search_queries), debug_info["timings"]["enhancement"])

    # ── 3. Hybrid retrieval ────────────────────────────────────────────────
    _emit({"type": "step", "step": "正在检索知识库…", "progress": 40})
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
    log.info("pipeline.run | retrieval done: %d merged hits in %.2fs",
             len(merged_hits), debug_info["timings"]["retrieval"])

    if debug:
        print(f"[Retrieval] {len(merged_hits)} unique hits from {len(search_queries)} queries")

    # ── 4. Reranking + compression ─────────────────────────────────────────
    _emit({"type": "step", "step": "正在精排相关文档…", "progress": 62})
    t1 = time.time()
    reranked = rerank(query, merged_hits, top_k=TOP_K_RERANK)
    compressed = compress_context(query, reranked)
    debug_info["timings"]["reranking"] = round(time.time() - t1, 3)
    log.info("pipeline.run | reranking done: %d hits in %.2fs",
             len(reranked), debug_info["timings"]["reranking"])

    if debug:
        print(f"[Rerank] Top scores: {[round(h.get('rerank_score',0),3) for h in reranked[:5]]}")

    # ── 5. CRAG quality check ──────────────────────────────────────────────
    _emit({"type": "step", "step": "正在评估检索质量…", "progress": 75})
    confidence, best_score = evaluate_retrieval(query, reranked)
    debug_info["crag_confidence"] = confidence
    debug_info["crag_best_score"] = round(best_score, 4)

    log.info("pipeline.run | CRAG confidence=%s best_score=%.4f", confidence, best_score)
    if debug:
        print(f"[CRAG] confidence={confidence}, best_score={best_score:.4f}")

    context = build_context(compressed, confidence)

    # ── 6. LLM generation ──────────────────────────────────────────────────
    _emit({"type": "step", "step": "AI 正在生成回答…", "progress": 82})
    t1 = time.time()
    answer = generate(
        query, context,
        provider=provider, model=model,
        stream=debug,
        token_callback=lambda t: _emit({"type": "token", "text": t}),
    )
    debug_info["timings"]["generation"] = round(time.time() - t1, 3)
    log.info("pipeline.run | generation done: answer_len=%d in %.2fs",
             len(answer), debug_info["timings"]["generation"])
    if not answer:
        log.warning("pipeline.run | EMPTY answer returned from generator!")

    # ── 7. Cache store ─────────────────────────────────────────────────────
    sources = [{"db": h.get("db",""), "source": h.get("source",""),
                "score": round(h.get("rerank_score",0), 3)} for h in reranked[:5]]
    cache_id = cache_store(query, answer, sources)

    debug_info["timings"]["total"] = round(time.time() - t0, 3)
    log.info("pipeline.run | DONE total=%.2fs", debug_info["timings"]["total"])

    return {
        "answer":     answer,
        "sources":    sources,
        "cache_id":   cache_id,
        "confidence": confidence,
        "from_cache": False,
        "debug_info": debug_info if debug else {},
    }
