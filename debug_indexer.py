"""
调试：索引构建模块
用法：python debug_indexer.py -i "build"       # 全量构建（首次，约50min）
     python debug_indexer.py -i "incremental"  # 增量更新（只处理变更文件）
     python debug_indexer.py -i "check"        # 检查现有索引状态
     python debug_indexer.py -i "force"        # 强制全量重建
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(__file__))

from src.config import VECTOR_DB_DIR, BM25_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="build",
                    help="'build' | 'incremental' | 'check' | 'force'")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] 索引模块 | 操作: {args.input}")
    print("=" * 60)

    # ── check ─────────────────────────────────────────────────────────────
    if args.input == "check":
        from src.indexer import COLLECTION_NAME, BM25_PICKLE, MANIFEST_FILE, load_index

        print(f"\nVector DB 路径: {VECTOR_DB_DIR}")
        print(f"BM25 Pickle:   {BM25_PICKLE}")
        print(f"BM25 存在:     {os.path.exists(BM25_PICKLE)}")
        print(f"Manifest 存在: {os.path.exists(MANIFEST_FILE)}")

        if os.path.exists(MANIFEST_FILE):
            with open(MANIFEST_FILE, encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"\n已索引文件 ({len(manifest)} 个):")
            for key, info in manifest.items():
                print(f"  {key}: {info['chunk_count']} chunks | {info['indexed_at']}")

        try:
            col, bm25, bm25_docs = load_index()
            print(f"\nChromaDB '{COLLECTION_NAME}': {col.count()} chunks")
            print(f"BM25 docs: {len(bm25_docs)}")

            results = col.query(query_texts=["泪道阻塞"],  n_results=3,
                                include=["documents", "metadatas", "distances"])
            print("\n示例查询 '泪道阻塞'：")
            for doc, meta, dist in zip(results["documents"][0],
                                       results["metadatas"][0],
                                       results["distances"][0]):
                print(f"  [{meta.get('db')}] sim={1-dist:.3f} | {doc[:80]}...")
            print("\n[OK] 索引就绪")
        except Exception as e:
            print(f"\n索引未就绪或出错: {e}")
        return

    # ── build / incremental / force ───────────────────────────────────────
    from src.indexer import build_index, COLLECTION_NAME, BM25_PICKLE, BM25_DOCS_FILE, load_index

    if args.input == "incremental":
        print("\n增量更新模式：只处理新增或变更的文件...\n")
        build_index(incremental=True)
    elif args.input == "force":
        print("\n强制全量重建...\n")
        build_index(force=True)
    else:
        print("\n全量构建（首次）...\n")
        build_index()

    # Verify
    try:
        col, bm25, bm25_docs = load_index()
        print(f"\n[验证] ChromaDB: {col.count()} chunks")
        print(f"[验证] BM25 docs: {len(bm25_docs)}")

        r = col.query(query_texts=["眼眶蜂窝织炎治疗"], n_results=3,
                      include=["documents", "metadatas", "distances"])
        print("\n[验证] 测试查询 '眼眶蜂窝织炎治疗'：")
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            print(f"  [{meta.get('db')}] sim={1-dist:.3f} | {doc[:80]}...")
        print("\n[OK] 索引构建完成")
    except Exception as e:
        print(f"\n验证失败: {e}")

if __name__ == "__main__":
    main()
