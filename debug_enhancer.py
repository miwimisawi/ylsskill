"""
调试：查询增强模块 (HyDE + Multi-Query + Step-back)
用法：python debug_enhancer.py -i "急性泪囊炎怎么处理"
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="原始查询")
    ap.add_argument("--mode", default="all",
                    help="'hyde' | 'multi' | 'stepback' | 'all'")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] 查询增强模块")
    print(f"原始查询: {args.input}")
    print("=" * 60)

    from src.enhancer import hyde, multi_query, step_back

    if args.mode in ("hyde", "all"):
        print("\n[1] HyDE - 假设文档生成：")
        passage = hyde(args.input)
        print(f"  {passage}")

    if args.mode in ("multi", "all"):
        print("\n[2] Multi-Query - 多角度重写：")
        queries = multi_query(args.input, n=3)
        for i, q in enumerate(queries):
            label = "原始" if i == 0 else f"改写{i}"
            print(f"  [{label}] {q}")

    if args.mode in ("stepback", "all"):
        print("\n[3] Step-back - 背景问题提炼：")
        sb = step_back(args.input)
        print(f"  {sb}")

    print("\n[OK] 查询增强模块正常")

if __name__ == "__main__":
    main()
