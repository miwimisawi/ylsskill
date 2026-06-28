"""
启动眼科助手 Web 服务
用法：python run_server.py [--port 8000]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    import uvicorn
    print(f"\n眼科住院助手启动中...")
    print(f"用户界面:  http://{args.host}:{args.port}/")
    print(f"调试界面:  http://{args.host}:{args.port}/debug.html")
    print(f"API 文档:  http://{args.host}:{args.port}/docs\n")
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)
