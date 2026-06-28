"""
调试：混合检索模块（需要先运行 debug_indexer.py 建立索引）
用法：python debug_retriever.py -i "泪道阻塞的治疗方法"
     python debug_retriever.py -i "orbital cellulitis management"
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="检索查询")
    ap.add_argument("--top", type=int, default=10, help="显示前N个结果")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] 混合检索模块")
    print(f"查询: {args.input}")
    print("=" * 60)

    from src.indexer import load_index
    from src.retriever import dense_search, sparse_search, rrf_fusion, hybrid_search

    print("\n加载索引...")
    col, bm25, bm25_docs = load_index()
    print(f"ChromaDB: {col.count()} chunks | BM25: {len(bm25_docs)} docs")

    # Dense retrieval
    print(f"\n[1] 稠密检索 (BGE-M3):")
    dense = dense_search(col, args.input, top_k=10)
    for i, h in enumerate(dense[:5]):
        print(f"  {i+1}. [{h['db']}] sim={h['score']:.4f} | {h['text'][:80]}...")

    # Sparse retrieval
    print(f"\n[2] 稀疏检索 (BM25):")
    sparse = sparse_search(bm25, bm25_docs, args.input, top_k=10)
    if sparse:
        for i, h in enumerate(sparse[:5]):
            print(f"  {i+1}. [{h['db']}] bm25={h['score']:.3f} | {h['text'][:80]}...")
    else:
        print("  (无 BM25 命中，关键词可能不在词库中)")

    # RRF fusion
    print(f"\n[3] RRF 融合后 Top {args.top}:")
    fused = rrf_fusion(dense, sparse, top_k=args.top)
    for i, h in enumerate(fused):
        methods = "+".join(h.get("methods", ["?"]))
        print(f"  {i+1}. [{h['db']}|{methods}] rrf={h['rrf_score']:.4f}")
        print(f"       {h['text'][:100]}...")

    print(f"\n[OK] 混合检索正常 | dense:{len(dense)} sparse:{len(sparse)} fused:{len(fused)}")

if __name__ == "__main__":
    main()
