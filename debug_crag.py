"""
调试：CRAG 质量评估模块
用法：python debug_crag.py -i "泪囊炎手术时机"
     python debug_crag.py -i "量子计算在眼科的应用"   # 预期低置信度
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="查询")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] CRAG 质量评估模块")
    print(f"查询: {args.input}")
    print("=" * 60)

    from src.indexer import load_index
    from src.retriever import hybrid_search
    from src.reranker import rerank, compress_context
    from src.crag import evaluate_retrieval, build_context

    col, bm25, bm25_docs = load_index()
    hits = hybrid_search(col, bm25, bm25_docs, args.input, top_k_final=20)
    reranked = rerank(args.input, hits, top_k=8)
    compressed = compress_context(args.input, reranked)

    confidence, best_score = evaluate_retrieval(args.input, reranked)

    print(f"\n[CRAG 评估结果]")
    print(f"  置信度:   {confidence.upper()}")
    print(f"  最高分:   {best_score:.4f}")
    print(f"  命中数:   {len(reranked)}")
    print(f"  分数分布: {[round(h.get('rerank_score',0),3) for h in reranked[:5]]}")

    if confidence == "high":
        print("  → 决策: 直接使用本地知识库结果")
    elif confidence == "medium":
        print("  → 决策: 使用本地结果 + 标注置信度")
    else:
        print("  → 决策: ⚠ 触发补充搜索 / 提醒用户核实")

    context = build_context(compressed, confidence, max_chars=800)
    print(f"\n[构建的上下文] ({len(context)} 字符):")
    print("-" * 40)
    print(context[:600])
    if len(context) > 600:
        print(f"... (截断，共 {len(context)} 字符)")

    print("\n[OK] CRAG 模块正常")

if __name__ == "__main__":
    main()
