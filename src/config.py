"""Central configuration for the ophthalmology AI assistant."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(BASE_DIR, "data_given")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
CACHE_DB_PATH = os.path.join(BASE_DIR, "cache_db", "cache.sqlite")
BM25_DIR      = os.path.join(BASE_DIR, "vector_db", "bm25")

# ── Models (local, downloaded on first use) ────────────────────────────────
EMBED_MODEL   = "BAAI/bge-m3"           # bilingual Chinese+English
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"

# ── API ────────────────────────────────────────────────────────────────────
SILICONFLOW_API_KEY  = "sk-wykbbnrkhudmhaervmhrontzugxfrlmucnxkhtmklcdpbenr"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL            = "Qwen/Qwen2.5-72B-Instruct"   # default; overrideable

# Models available in the UI dropdown (SiliconFlow-hosted)
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-72B-Instruct",  "label": "Qwen2.5-72B（默认，高质量）"},
    {"id": "Qwen/Qwen2.5-32B-Instruct",  "label": "Qwen2.5-32B（较快）"},
    {"id": "Qwen/Qwen2.5-7B-Instruct",   "label": "Qwen2.5-7B（最快）"},
    {"id": "deepseek-ai/DeepSeek-V3",     "label": "DeepSeek-V3（推理强）"},
    {"id": "THUDM/glm-4-9b-chat",         "label": "GLM-4-9B（轻量）"},
]

# OpenAI-compatible (with BASE_URL override)
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# ── Chunking ───────────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE   = 1024   # tokens (approx chars/1.5 for Chinese)
CHILD_CHUNK_SIZE    = 256
CHUNK_OVERLAP       = 32

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K_DENSE    = 50   # dense retrieval candidates
TOP_K_SPARSE   = 50   # BM25 candidates
TOP_K_FUSED    = 20   # after RRF fusion
TOP_K_RERANK   = 8    # after reranking → sent to LLM
RRF_K          = 60   # RRF smoothing constant

# ── CRAG ───────────────────────────────────────────────────────────────────
CRAG_RELEVANCE_THRESHOLD = 0.35  # below → trigger web/supplemental search

# ── Database metadata ──────────────────────────────────────────────────────
DB_SOURCES = {
    "database1": {"lang": "en",    "type": "textbook", "title": "BCSC Vol.1"},
    "database2": {"lang": "en",    "type": "textbook", "title": "BCSC Vol.2"},
    "database3": {"lang": "en",    "type": "textbook", "title": "Smith & Nesi Oculoplastics"},
    "database4": {"lang": "zh",    "type": "textbook", "title": "实用泪器病学"},
    "database5": {"lang": "zh",    "type": "textbook", "title": "实用眼眶病学"},
    "database6": {"lang": "zh",    "type": "template", "title": "科室内部文书模板"},
}
