"""
调试：重排序模块 (BGE-Reranker + 上下文压缩)
用法：python debug_reranker.py -i "泪道阻塞的手术适应证"
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="查询")
    ap.add_argument("--top", type=int, default=5, help="展示前N个重排序结果")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] 重排序模块")
    print(f"查询: {args.input}")
    print("=" * 60)

    from src.indexer import load_index
    from src.retriever import hybrid_search
    from src.reranker import rerank, compress_context

    print("\n加载索引并检索...")
    col, bm25, bm25_docs = load_index()
    hits = hybrid_search(col, bm25, bm25_docs, args.input, top_k_final=20)
    print(f"检索到 {len(hits)} 个候选")

    print(f"\n检索前5 (重排序前):")
    for i, h in enumerate(hits[:5]):
        print(f"  {i+1}. rrf={h.get('rrf_score',0):.4f} [{h['db']}] {h['text'][:80]}...")

    print(f"\n正在运行 BGE-Reranker (首次运行需下载模型~300MB)...")
    reranked = rerank(args.input, hits, top_k=args.top)

    print(f"\n重排序后 Top {args.top}:")
    for i, h in enumerate(reranked):
        rrf = h.get('rrf_score', 0)
        rer = h.get('rerank_score', 0)
        print(f"  {i+1}. rerank={rer:.4f} (rrf was {rrf:.4f}) [{h['db']}]")
        print(f"       {h['text'][:100]}...")

    print(f"\n上下文压缩后:")
    compressed = compress_context(args.input, reranked)
    for i, h in enumerate(compressed[:3]):
        orig_len = len(h.get("parent_text",""))
        comp_len = len(h.get("context",""))
        print(f"  {i+1}. 压缩: {orig_len} → {comp_len} 字符 ({100*comp_len//max(1,orig_len)}%)")
        print(f"       {h.get('context','')[:120]}...")

    print("\n[OK] 重排序模块正常")

if __name__ == "__main__":
    main()
