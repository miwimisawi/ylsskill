# 眼科住院医师 AI 助手

面向眼科住院医师的智能问答系统，覆盖**医学文书生成**、**急症辅助诊断**与**急症处理指导**三大场景。基于先进的检索增强生成（RAG）管道，融合科室内部文书模板与权威眼科教材，所有推理均通过云端 API 完成，本地无需 GPU。

---

## 目录

1. [功能特性](#功能特性)
2. [技术架构](#技术架构)
3. [知识库](#知识库)
4. [项目结构](#项目结构)
5. [环境要求](#环境要求)
6. [快速开始](#快速开始)
7. [构建与更新索引](#构建与更新索引)
8. [启动服务](#启动服务)
9. [使用界面](#使用界面)
10. [远程访问（手机演示）](#远程访问手机演示)
11. [API 参考](#api-参考)
12. [配置参数](#配置参数)
13. [调试脚本](#调试脚本)

---

## 功能特性

- **Parent-Child 分块检索**：父块（1024字）提供上下文，子块（256字）精确定位，兼顾精度与召回
- **混合搜索**：稠密向量（BGE-M3）+ 稀疏关键词（BM25）双路召回，RRF 融合排序
- **重排序**：BGE-Reranker-v2-M3 对 top-20 候选二次排序，取 top-8 送入 LLM
- **CRAG 置信度评估**：自动判断检索质量（HIGH/MEDIUM/LOW），低置信度时触发补充搜索
- **语义缓存**：相同或语义相近的问题直接命中缓存，显著提速
- **查询增强（可选）**：
  - HyDE — 生成假设答案段落再检索
  - Multi-Query — 多角度改写问题扩大召回
  - Step-back — 提炼背景问题先检索基础知识
- **中英文双语**：教材英文，文书模板中文，BGE-M3 原生支持双语
- **模型可选**：支持 Qwen3 / DeepSeek-V4 / GLM-5 等 13 个云端模型
- **流式输出**：SSE 实时推送，逐字显示，无需等待

---

## 技术架构

```
用户提问
    │
    ▼
[语义缓存查询] ──── 命中 ────▶ 直接返回缓存答案
    │ 未命中
    ▼
[查询增强]
  ├─ HyDE（假设文档嵌入）
  ├─ Multi-Query（多路改写）
  └─ Step-back（背景问题）
    │
    ▼
[混合检索]
  ├─ 稠密检索：BGE-M3 向量 → ChromaDB top-50
  ├─ 稀疏检索：BM25 → top-50
  └─ RRF 融合 → top-20
    │
    ▼
[重排序] BGE-Reranker-v2-M3 → top-8
    │
    ▼
[CRAG 置信度评估]
  ├─ HIGH (≥0.70)：直接使用
  ├─ MEDIUM (≥0.35)：结合原始检索
  └─ LOW (<0.35)：触发补充搜索
    │
    ▼
[LLM 生成] 云端模型 SSE 流式输出
    │
    ▼
[缓存存储] + 用户反馈收集
```

### 关键参数

| 参数 | 值 |
|------|-----|
| 嵌入模型 | BAAI/bge-m3（via SiliconFlow API） |
| 重排序模型 | BAAI/bge-reranker-v2-m3（via SiliconFlow API） |
| 父块大小 | 1024 字符 |
| 子块大小 | 256 字符（重叠 32） |
| 稠密检索 top-K | 50 |
| 稀疏检索 top-K | 50 |
| RRF 融合 top-K | 20 |
| 重排序 top-K | 8 |
| RRF 平滑常数 k | 60 |
| CRAG 置信度阈值 | 0.35 |
| 语义缓存阈值 | 0.92（余弦相似度） |

---

## 知识库

| 库 ID | 语言 | 类型 | 内容 |
|--------|------|------|------|
| database1 | EN | 教材 | BCSC Vol.1（Basic and Clinical Science Course） |
| database2 | EN | 教材 | BCSC Vol.2 |
| database3 | EN | 教材 | Smith & Nesi's Ophthalmic Plastic and Reconstructive Surgery |
| database4 | ZH | 教材 | 实用泪器病学 |
| database5 | ZH | 教材 | 实用眼眶病学 |
| database6 | ZH | 模板 | 科室内部文书模板 |

原始文件存放于 `data_given/`（已列入 .gitignore，不上传）。索引构建后存储于 `vector_db/`（同样不上传）。

---

## 项目结构

```
workspace1/
├── src/
│   ├── config.py          # 全局配置（API Key、模型、检索参数）
│   ├── indexer.py         # 文档读取 → 分块 → BGE-M3 嵌入 → ChromaDB 存储
│   ├── chunker.py         # Parent-Child 分块逻辑
│   ├── retriever.py       # 混合检索（稠密 + BM25 + RRF）
│   ├── reranker.py        # BGE-Reranker 重排序
│   ├── enhancer.py        # HyDE / Multi-Query / Step-back 查询增强
│   ├── crag.py            # CRAG 置信度评估
│   ├── generator.py       # LLM 调用（流式/非流式）
│   ├── cache.py           # 语义缓存（ChromaDB + SQLite）
│   ├── pipeline.py        # 完整 RAG 管道编排
│   └── api.py             # FastAPI 后端（SSE 流、/models、/stats 等）
├── frontend/
│   ├── index.html         # 用户界面（普通问答）
│   └── debug.html         # 调试界面（显示检索链路细节）
├── debug_indexer.py       # 构建 / 更新向量索引
├── debug_chunker.py       # 测试分块效果
├── debug_retriever.py     # 测试混合检索
├── debug_reranker.py      # 测试重排序
├── debug_enhancer.py      # 测试查询增强
├── debug_crag.py          # 测试 CRAG 置信度
├── debug_generator.py     # 测试 LLM 生成
├── debug_cache.py         # 测试语义缓存
├── debug_pipeline.py      # 端到端管道测试
├── streamlit_app.py       # Streamlit 版界面
├── run_server.py          # 启动 FastAPI 服务
├── requirements.txt       # Python 依赖
└── .gitignore
```

---

## 环境要求

- Python 3.10+（推荐使用 conda 环境）
- 无 GPU 要求（嵌入和重排序均通过 SiliconFlow API 云端完成）
- SiliconFlow API Key（已在 `src/config.py` 中预置）
- 网络连通 `api.siliconflow.cn`

---

## 快速开始

### 1. 创建环境

```bash
conda create -n openai python=3.11 -y
conda activate openai
pip install -r requirements.txt
```

### 2. 放置原始文档

将知识库文件按以下结构放入 `data_given/`：

```
data_given/
├── database1/   # BCSC Vol.1（Markdown 或 PDF）
├── database2/   # BCSC Vol.2（PDF）
├── database3/   # Smith & Nesi（EPUB/PDF）
├── database4/   # 实用泪器病学
├── database5/   # 实用眼眶病学
└── database6/   # 科室文书模板（Markdown/TXT）
```

### 3. 构建向量索引

```bash
python debug_indexer.py
```

首次构建会调用 SiliconFlow API 批量嵌入，时间取决于文档量（通常 5~30 分钟）。后续运行只处理变更文件（增量更新，基于 MD5 哈希）。

### 4. 启动服务

```bash
python run_server.py
```

服务启动后立即在后台加载向量索引（无需等待第一次请求）。

---

## 构建与更新索引

`debug_indexer.py` 支持增量更新——通过 `vector_db/manifest.json` 记录每个文件的 MD5，只重新嵌入发生变化的文件。

添加新知识库文件后，直接重新运行即可：

```bash
python debug_indexer.py
```

索引存储位置：
- `vector_db/`：ChromaDB 持久化数据库（向量 + 父块文本）
- `vector_db/bm25/`：BM25 索引（pickle 序列化）
- `vector_db/manifest.json`：文件哈希清单

---

## 启动服务

### FastAPI 后端（推荐）

```bash
python run_server.py
```

| 地址 | 说明 |
|------|------|
| http://127.0.0.1:8000/ | 用户界面 |
| http://127.0.0.1:8000/debug.html | 调试界面 |
| http://127.0.0.1:8000/docs | API 文档（Swagger） |
| http://127.0.0.1:8000/health | 健康检查（含索引就绪状态） |

### Streamlit 版本

```bash
python -m streamlit run streamlit_app.py
```

默认地址：http://localhost:8501

> **注意**：Streamlit Cloud 无法访问本地 `vector_db/` 和 `data_given/`，仅适合本地演示。

---

## 使用界面

### 用户界面（`index.html`）

- 输入问题，点击发送或按 Enter
- 点击页面底部的**建议问题**快速开始
- 顶部工具栏可选择：
  - **模型**：13 个可选模型（Qwen3 / DeepSeek-V4 / GLM-5 等）
  - **HyDE**：假设文档嵌入（提升语义匹配，建议开启）
  - **多查询**：多角度改写（提升召回率，略增延迟）
  - **背景提炼**：Step-back 背景知识检索
- 消息底部可给出反馈（点赞/点踩），影响语义缓存

### 调试界面（`debug.html`）

在用户界面基础上，右侧额外显示：

- **检索详情**：命中的文档来源、相关性分数
- **CRAG 状态**：置信度等级（HIGH/MEDIUM/LOW）及分数
- **计时信息**：各阶段耗时（检索、重排、生成等）
- **缓存状态**：是否命中缓存，缓存 ID
- **使用模型**：本次实际调用的模型名称
- **生成的查询变体**（Multi-Query 模式）

---

## 远程访问（手机演示）

当手机与电脑不在同一局域网时，使用 Cloudflare Tunnel 生成公网链接：

**1. 下载 cloudflared**

前往 https://github.com/cloudflare/cloudflared/releases 下载 `cloudflared-windows-amd64.exe`，重命名为 `cloudflared.exe`。

**2. 启动后端**

```bash
python run_server.py
```

**3. 开启隧道**

```bash
cloudflared.exe tunnel --url http://localhost:8000
```

输出中会出现类似 `https://xxxx-xxxx-xxxx.trycloudflare.com` 的公网地址，用手机浏览器打开即可。

> 免费、无需注册账号、每次隧道关闭后链接失效。

---

## API 参考

### POST `/chat`

流式问答（SSE）。

**请求体**

```json
{
  "query": "泪道阻塞术前如何评估？",
  "use_hyde": true,
  "use_multi_query": false,
  "use_step_back": false,
  "model": "Qwen/Qwen3.6-35B-A3B",
  "stream": true
}
```

**SSE 事件格式**

```
data: {"type": "token", "content": "泪道"}
data: {"type": "token", "content": "阻塞..."}
data: {"type": "done", "from_cache": false, "confidence": "HIGH",
       "crag_best_score": 0.82, "sources": [...], "timings": {...},
       "cache_id": "abc123", "model": "Qwen/Qwen3.6-35B-A3B"}
```

### GET `/models`

返回可用模型列表。

### GET `/health`

```json
{"status": "ok", "index_ready": true}
```

### GET `/stats`

返回缓存统计（总条目、命中次数、正负反馈数）。

### POST `/feedback`

```json
{"cache_id": "abc123", "positive": true}
```

---

## 配置参数

所有参数集中在 [`src/config.py`](src/config.py)：

```python
# API
SILICONFLOW_API_KEY  = "sk-..."
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL            = "Qwen/Qwen3.6-35B-A3B"   # 默认模型

# 嵌入 / 重排序模型（云端）
EMBED_MODEL   = "BAAI/bge-m3"
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"

# 分块
PARENT_CHUNK_SIZE = 1024
CHILD_CHUNK_SIZE  = 256
CHUNK_OVERLAP     = 32

# 检索
TOP_K_DENSE  = 50    # 稠密检索候选数
TOP_K_SPARSE = 50    # BM25 候选数
TOP_K_FUSED  = 20    # RRF 融合后保留数
TOP_K_RERANK = 8     # 重排序后送入 LLM 的数量
RRF_K        = 60    # RRF 平滑常数

# CRAG
CRAG_RELEVANCE_THRESHOLD = 0.35
```

---

## 调试脚本

各阶段均有独立调试脚本，便于单独测试：

| 脚本 | 测试内容 |
|------|---------|
| `debug_indexer.py` | 构建/更新向量索引 |
| `debug_chunker.py` | 分块效果（父/子块大小、重叠） |
| `debug_retriever.py` | 混合检索结果（稠密+BM25+RRF） |
| `debug_reranker.py` | 重排序前后对比 |
| `debug_enhancer.py` | HyDE / Multi-Query / Step-back 输出 |
| `debug_crag.py` | CRAG 置信度评分分布 |
| `debug_generator.py` | LLM 生成（含流式）测试 |
| `debug_cache.py` | 语义缓存读写、命中率 |
| `debug_pipeline.py` | 端到端完整管道测试 |

运行示例：

```bash
python debug_pipeline.py
# 输入测试问题，查看完整 RAG 链路输出及各阶段耗时
```

---

## 可选模型

| 模型 | 说明 |
|------|------|
| Qwen/Qwen3.6-35B-A3B | **默认**，平衡速度与质量 |
| Qwen/Qwen3.6-27B | 轻量版 Qwen3.6 |
| Qwen/Qwen3.5-397B-A17B | 旗舰模型，最高质量 |
| Qwen/Qwen3.5-122B-A10B | 高质量大模型 |
| Qwen/Qwen3.5-35B-A3B | 中等规模 |
| Qwen/Qwen3.5-27B | 轻量 |
| Qwen/Qwen3.5-4B | **最快**，适合快速草稿 |
| deepseek-ai/DeepSeek-V4-Pro | DeepSeek 旗舰 |
| deepseek-ai/DeepSeek-V4-Flash | DeepSeek 快速版 |
| deepseek-ai/DeepSeek-V3.2 | DeepSeek V3 系列 |
| THUDM/GLM-5.2 | GLM 最新版 |
| THUDM/GLM-Z1-9B-0414 | GLM 轻量推理 |
| THUDM/GLM-4-9B-0414 | GLM-4 轻量 |

---

## 注意事项

- `data_given/`、`vector_db/`、`cache_db/` 均已列入 `.gitignore`，不会上传至 GitHub
- API Key 已硬编码在 `src/config.py`，生产环境建议改用环境变量
- Streamlit Cloud 等无状态平台无法运行本项目（需要本地持久化索引）；推荐本地部署或自管云服务器
