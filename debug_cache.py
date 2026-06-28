"""
调试：语义缓存模块
用法：python debug_cache.py -i "测试问题"
     python debug_cache.py -i "stats"      # 查看缓存统计
     python debug_cache.py -i "clear"      # 清空缓存
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="查询 | 'stats' | 'clear'")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] 语义缓存模块")
    print("=" * 60)

    from src.cache import cache_lookup, cache_store, cache_feedback, cache_stats

    if args.input == "stats":
        stats = cache_stats()
        print(f"\n缓存统计：")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.input == "clear":
        from src.config import CACHE_DB_PATH, VECTOR_DB_DIR
        import sqlite3, chromadb
        conn = sqlite3.connect(CACHE_DB_PATH)
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        try:
            client.delete_collection("query_cache")
        except Exception:
            pass
        print("缓存已清空")
        return

    query = args.input

    # Test store
    print(f"\n[1] 存储测试条目: '{query}'")
    cid = cache_store(
        query=query,
        answer=f"这是针对「{query}」的测试缓存答案。",
        sources=[{"db":"test","source":"debug","score":0.99}]
    )
    print(f"  cache_id: {cid}")

    # Test lookup (exact)
    print(f"\n[2] 精确查找: '{query}'")
    result = cache_lookup(query)
    if result:
        print(f"  命中! similarity={result['similarity']:.4f}")
        print(f"  答案: {result['answer']}")
    else:
        print("  未命中（阈值可能太高）")

    # Test similar query
    similar = query.replace("的", "").replace("怎么", "如何") if "的" in query or "怎么" in query else query + "？"
    print(f"\n[3] 相似查找: '{similar}'")
    result2 = cache_lookup(similar)
    if result2:
        print(f"  命中! similarity={result2['similarity']:.4f}, 原始: '{result2['query_orig']}'")
    else:
        print("  未命中（相似度不足）")

    # Test feedback
    print(f"\n[4] 记录正向反馈...")
    cache_feedback(cid, positive=True)
    print(f"  反馈已记录")

    # Stats
    stats = cache_stats()
    print(f"\n当前缓存状态: {stats}")
    print("\n[OK] 缓存模块正常")

if __name__ == "__main__":
    main()
