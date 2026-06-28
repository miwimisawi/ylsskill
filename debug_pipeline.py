"""
调试：完整流水线（需要先建好索引）
用法：python debug_pipeline.py -i "泪道阻塞的手术治疗"
     python debug_pipeline.py -i "给我一个VTE病程模板"
     python debug_pipeline.py -i "orbital cellulitis antibiotic"
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="问题")
    ap.add_argument("--hyde",  action="store_true", default=True, help="启用 HyDE")
    ap.add_argument("--no-hyde", dest="hyde", action="store_false")
    ap.add_argument("--multi", action="store_true", default=False, help="启用 Multi-Query")
    ap.add_argument("--stepback", action="store_true", default=False, help="启用 Step-back")
    ap.add_argument("--provider", default="siliconflow")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] 完整 RAG 流水线")
    print(f"问题: {args.input}")
    print(f"HyDE:{args.hyde} | Multi:{args.multi} | StepBack:{args.stepback}")
    print("=" * 60)

    from src.indexer import load_index
    from src.pipeline import run

    print("\n加载索引...")
    col, bm25, bm25_docs = load_index()

    print("运行完整流水线 (debug=True)...\n")
    result = run(
        query=args.input,
        col=col, bm25=bm25, bm25_docs=bm25_docs,
        provider=args.provider,
        use_hyde=args.hyde,
        use_multi_query=args.multi,
        use_step_back=args.stepback,
        debug=True,
    )

    print("\n" + "=" * 60)
    print("[流水线结果]")
    print("=" * 60)
    print(f"来源缓存:  {result['from_cache']}")
    print(f"置信度:    {result['confidence']}")
    print(f"cache_id:  {result['cache_id']}")

    di = result.get("debug_info", {})
    if di:
        print(f"\n[耗时分解]")
        for k, v in di.get("timings", {}).items():
            print(f"  {k}: {v}s")
        print(f"搜索查询数: {len(di.get('search_queries',[]))}")
        print(f"检索命中数: {di.get('retrieval_count', '?')}")
        print(f"CRAG分数:  {di.get('crag_best_score','?')}")

    print(f"\n[来源]")
    for s in result.get("sources", [])[:5]:
        print(f"  [{s['db']}] {s['source']} (score={s['score']})")

    print(f"\n[答案]")
    print("-" * 40)
    print(result["answer"])

    # Simulate feedback
    print(f"\n[模拟用户反馈] 标记为「有用」...")
    from src.cache import cache_feedback
    cache_feedback(result["cache_id"], positive=True)
    print("反馈已记录")

    # Test cache hit on re-run
    print(f"\n[测试缓存] 再次运行相同问题...")
    result2 = run(
        query=args.input,
        col=col, bm25=bm25, bm25_docs=bm25_docs,
        debug=False,
    )
    print(f"来自缓存: {result2['from_cache']} | 置信度: {result2['confidence']}")

    print("\n[OK] 完整流水线正常")

if __name__ == "__main__":
    main()
