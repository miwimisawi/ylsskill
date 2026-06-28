"""
调试：LLM 生成模块
用法：python debug_generator.py -i "急性泪囊炎首选抗生素"
     python debug_generator.py -i "给我一个VTE评分病程模板" --stream
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="问题")
    ap.add_argument("--model",  default=None, help="覆盖模型名")
    ap.add_argument("--stream", action="store_true", help="流式输出")
    ap.add_argument("--provider", default="siliconflow",
                    help="'siliconflow' | 'openai' | 'local'")
    args = ap.parse_args()

    print("=" * 60)
    print(f"[调试] LLM 生成模块")
    print(f"问题:    {args.input}")
    print(f"Provider: {args.provider} | Stream: {args.stream}")
    print("=" * 60)

    from src.generator import generate
    from src.config import LLM_MODEL

    # Use a simple test context
    context = """【参考1 | database6】
VTE（静脉血栓栓塞）评分标准：
- 0-1分：无需处理
- 2分：中危，需要开VTE医嘱
- ≥3分：高危，需抗凝处理
- ≥5分：需内科会诊

【参考2 | database6】
VTE病程模板要素：评分日期、分值、危险因素、处理措施。"""

    model_name = args.model or LLM_MODEL
    print(f"\n使用模型: {model_name}")
    print(f"\n{'流式' if args.stream else '同步'}生成中...\n")

    import time
    t0 = time.time()
    answer = generate(
        query=args.input,
        context=context,
        provider=args.provider,
        model=args.model,
        stream=args.stream,
    )
    elapsed = time.time() - t0

    if not args.stream:
        print(f"\n[答案]")
        print("-" * 40)
        print(answer)

    print(f"\n[耗时] {elapsed:.2f}s | 长度: {len(answer)} 字符")
    print("\n[OK] LLM 生成模块正常")

if __name__ == "__main__":
    main()
