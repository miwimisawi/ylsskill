"""
调试：切块模块
用法：python debug_chunker.py -i "database4"
     python debug_chunker.py -i "all"
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

from src.chunker import chunk_document, load_database_chunks
from src.config import DATA_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="database6",
                    help="数据库名 (database1-6) 或 'all'")
    ap.add_argument("--show", type=int, default=3, help="展示前N个chunk")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] 切块模块 | 输入: {args.input}")
    print("=" * 60)

    if args.input == "all":
        print(f"\n正在加载所有数据库 ({DATA_DIR})...\n")
        chunks = load_database_chunks(DATA_DIR)
        print(f"\n总计 child chunks: {len(chunks)}")

        # Stats by DB
        from collections import Counter
        db_counts = Counter(c["db"] for c in chunks)
        print("\n各数据库 chunk 数量：")
        for db, cnt in sorted(db_counts.items()):
            print(f"  {db}: {cnt} chunks")
    else:
        db_path = os.path.join(DATA_DIR, args.input)
        full_md = os.path.join(db_path, "full.md")
        chapters_dir = os.path.join(db_path, "chapters")

        if os.path.exists(full_md) and os.path.getsize(full_md) > 100:
            print(f"使用 full.md")
            with open(full_md, encoding="utf-8") as f:
                text = f.read()[:5000]
        elif os.path.exists(chapters_dir):
            chapter_files = sorted(f for f in os.listdir(chapters_dir) if f.endswith(".md"))
            fname = chapter_files[0]
            fpath = os.path.join(chapters_dir, fname)
            print(f"使用章节文件: {fname}")
            with open(fpath, encoding="utf-8") as f:
                text = f.read()
        else:
            # Single md file (e.g. database6)
            md_files = [f for f in os.listdir(db_path) if f.endswith(".md")]
            fname = md_files[0]
            print(f"使用单文件: {fname}")
            with open(os.path.join(db_path, fname), encoding="utf-8") as f:
                text = f.read()[:5000]

        print(f"输入文本长度: {len(text)} 字符")
        chunks = chunk_document(text, source="sample", db_name=args.input)
        print(f"生成 child chunks: {len(chunks)}")

    print(f"\n前 {args.show} 个 chunks 示例：")
    for i, c in enumerate(chunks[:args.show]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"  ID:          {c['id']}")
        print(f"  DB:          {c['db']}  Source: {c['source']}")
        print(f"  Child text:  {c['text'][:150]}...")
        print(f"  Parent text: {c['parent_text'][:200]}...")

    print("\n[OK] 切块模块正常")

if __name__ == "__main__":
    main()
