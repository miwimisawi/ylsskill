"""
FastAPI backend for the ophthalmology assistant.

Endpoints:
  POST /chat          — main query (streaming SSE)
  POST /feedback      — record thumbs up/down
  GET  /stats         — cache statistics
  GET  /health        — health check
  GET  /models        — list available models
"""
import asyncio, json, time, threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import load_index
from src.pipeline import run
from src.cache import cache_feedback, cache_stats
from src.config import AVAILABLE_MODELS, LLM_MODEL
from src.logger import get_logger

log = get_logger(__name__)

# ── Eager index loading ─────────────────────────────────────────────────────
_index = None
_index_lock = threading.Lock()
_index_ready = threading.Event()


def _load_index_bg():
    global _index
    log.info("startup | loading vector index...")
    t0 = time.time()
    try:
        idx = load_index()
    except Exception:
        log.exception("startup | failed to load index")
        return
    with _index_lock:
        _index = idx
    _index_ready.set()
    log.info("startup | index ready in %.1fs", time.time() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_load_index_bg, daemon=True)
    t.start()
    yield


def get_index():
    """Return index, blocking up to 120s if still loading."""
    if not _index_ready.wait(timeout=120):
        raise RuntimeError("Index not ready after 120s — check logs")
    with _index_lock:
        return _index


app = FastAPI(title="眼科住院助手", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    use_hyde: bool = True
    use_multi_query: bool = False
    use_step_back: bool = False
    provider: str = "siliconflow"
    model: Optional[str] = None
    stream: bool = True

class FeedbackRequest(BaseModel):
    cache_id: str
    positive: bool


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "index_ready": _index_ready.is_set()}


@app.get("/models")
def models():
    return {"models": AVAILABLE_MODELS, "default": LLM_MODEL}


@app.get("/stats")
def stats():
    return cache_stats()


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    cache_feedback(req.cache_id, req.positive)
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Returns Server-Sent Events stream.

    Event types:
      data: {"type": "step",  "step": "...", "progress": 0-100}
      data: {"type": "token", "text": "..."}
      data: {"type": "done",  "cache_id": "...", "sources": [...], ...}
      data: {"type": "error", "message": "..."}
    """
    col, bm25, bm25_docs = get_index()

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    result_holder: dict = {}

    def emit(evt: dict):
        """Called from pipeline thread — thread-safe queue put."""
        loop.call_soon_threadsafe(q.put_nowait, evt)

    log.info("chat | query=%r model=%s hyde=%s multi=%s stepback=%s",
             req.query[:80], req.model or LLM_MODEL,
             req.use_hyde, req.use_multi_query, req.use_step_back)

    def pipeline_thread():
        try:
            result = run(
                query=req.query,
                col=col,
                bm25=bm25,
                bm25_docs=bm25_docs,
                provider=req.provider,
                model=req.model,
                use_hyde=req.use_hyde,
                use_multi_query=req.use_multi_query,
                use_step_back=req.use_step_back,
                debug=False,
                emit=emit,
            )
            result_holder["result"] = result
        except Exception as e:
            log.exception("chat | pipeline_thread error")
            emit({"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=pipeline_thread, daemon=True).start()

    async def event_stream() -> AsyncGenerator[str, None]:
        def sse(obj: dict) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        had_error = False
        while True:
            evt = await q.get()
            if evt is None:
                break
            if evt["type"] == "error":
                had_error = True
                yield sse(evt)
                break
            yield sse(evt)  # step or token events forwarded as-is

        if not had_error:
            result = result_holder.get("result", {})
            di = result.get("debug_info", {})
            yield sse({
                "type":            "done",
                "cache_id":        result.get("cache_id"),
                "sources":         result.get("sources", [])[:5],
                "confidence":      result.get("confidence"),
                "from_cache":      result.get("from_cache"),
                "model":           req.model or LLM_MODEL,
                "timings":         di.get("timings", {}),
                "search_queries":  di.get("search_queries", []),
                "retrieval_count": di.get("retrieval_count", 0),
                "crag_best_score": di.get("crag_best_score"),
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Static files (frontend) ────────────────────────────────────────────────

_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
