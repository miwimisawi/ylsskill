# 调试指南 — 眼科助手 RAG Pipeline

> 环境：conda `openai`  
> 快速用法：`D:\miniconda\envs\openai\python.exe debug_xxx.py -i "参数"`  
> 建议先设置：`$env:PYTHONUTF8="1"` （避免 GBK 乱码）

---

## 调试顺序

```
1. debug_chunker   → 验证文本切块
2. debug_indexer   → 建立/检查索引（首次需 build，约 20-40 min）
3. debug_retriever → 验证混合检索
4. debug_enhancer  → 验证查询增强
5. debug_reranker  → 验证重排序
6. debug_crag      → 验证质量评估
7. debug_cache     → 验证语义缓存
8. debug_generator → 验证 LLM 生成
9. debug_pipeline  → 端到端流水线
```

---

## 1. debug_chunker.py — 切块模块

```bash
python debug_chunker.py -i "database6"   # 测试单个数据库
python debug_chunker.py -i "all"         # 测试全部 6 个数据库（约 41,005 chunks）
python debug_chunker.py -i "database1" --show 5   # 展示前 5 个 chunk
```

**策略：** Parent-Child 切块。Parent ~1024 字符保留上下文，Child ~256 字符用于向量检索。

---

## 2. debug_indexer.py — 索引模块

```bash
python debug_indexer.py -i "build"        # 首次建立索引（约50min）
python debug_indexer.py -i "incremental"  # 增量更新（只处理变更文件，节省token）
python debug_indexer.py -i "check"        # 查看索引状态 + manifest
python debug_indexer.py -i "force"        # 强制全量重建
```

**增量更新说明：**
- 用 MD5 哈希检测文件变更，只对变更/新增文件重新嵌入
- 首次运行自动 bootstrap manifest（不消耗 API）
- 之后每次只处理变化的文件，例如只更新 database4（1,245 chunks）可节省 97% token

---

## 3. debug_retriever.py — 混合检索模块

```bash
python debug_retriever.py -i "泪道阻塞的手术方法"
python debug_retriever.py -i "orbital cellulitis antibiotic treatment"
```

**需要索引就绪。** 执行稠密检索（BGE-M3）+ 稀疏检索（BM25）+ RRF 融合。

---

## 4. debug_enhancer.py — 查询增强模块

```bash
python debug_enhancer.py -i "青光眼的药物治疗"
python debug_enhancer.py -i "青光眼的药物治疗" --mode hyde      # 只测 HyDE
python debug_enhancer.py -i "青光眼的药物治疗" --mode multi     # 只测 Multi-Query
python debug_enhancer.py -i "青光眼的药物治疗" --mode stepback  # 只测 Step-back
python debug_enhancer.py -i "青光眼的药物治疗" --mode all       # 三种全测
```

**不需要索引。** 调用 SiliconFlow LLM 生成假设文档/改写查询。

---

## 5. debug_reranker.py — 重排序模块

```bash
python debug_reranker.py -i "泪囊炎的手术时机"
```

**需要索引就绪。** 用 BGE-Reranker-v2-m3 交叉编码器对检索结果重排，并进行上下文压缩。

---

## 6. debug_crag.py — CRAG 质量评估模块

```bash
python debug_crag.py -i "泪囊炎手术时机"          # 预期 HIGH
python debug_crag.py -i "量子计算在眼科的应用"     # 预期 LOW（知识库外）
```

**需要索引就绪。** 评估检索结果可信度：HIGH / MEDIUM / LOW，决定是否触发补充搜索。

---

## 7. debug_cache.py — 语义缓存模块

```bash
python debug_cache.py -i "急性泪囊炎的手术时机"   # 测试存储 + 检索
python debug_cache.py -i "stats"                   # 查看缓存统计
python debug_cache.py -i "clear"                   # 清空缓存
```

**不需要索引。** 使用 ChromaDB 向量相似度（阈值 0.92）命中语义相近问题的缓存。

---

## 8. debug_generator.py — LLM 生成模块

```bash
python debug_generator.py -i "VTE评分高危应如何处理"
python debug_generator.py -i "给我一个入院记录模板" --stream    # 流式输出
python debug_generator.py -i "急性闭角型青光眼的处理" --provider siliconflow
```

**不需要索引。** 直接调用 SiliconFlow Qwen2.5-72B，使用内置测试上下文。

---

## 9. debug_pipeline.py — 完整 RAG 流水线

```bash
python debug_pipeline.py -i "泪道阻塞的手术治疗"
python debug_pipeline.py -i "给我一个VTE病程模板"
python debug_pipeline.py -i "急性闭角型青光眼如何处理" --hyde --multi --stepback
python debug_pipeline.py -i "acute dacryocystitis treatment" --no-hyde
```

**需要索引就绪。** 执行完整流水线：缓存命中 → 查询增强 → 混合检索 → 重排序 → CRAG评估 → LLM生成 → 缓存存储 → 用户反馈。

---

## 关键参数速查

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `EMBED_MODEL` | config.py | `BAAI/bge-m3` | 嵌入模型 |
| `RERANK_MODEL` | config.py | `BAAI/bge-reranker-v2-m3` | 重排序模型 |
| `LLM_MODEL` | config.py | `Qwen/Qwen2.5-72B-Instruct` | LLM 模型 |
| `PARENT_CHUNK_SIZE` | config.py | 1024 | 父块字符数 |
| `CHILD_CHUNK_SIZE` | config.py | 256 | 子块字符数 |
| `TOP_K_RERANK` | config.py | 8 | 最终送入 LLM 的 chunk 数 |
| `CRAG_RELEVANCE_THRESHOLD` | config.py | 0.35 | CRAG 置信度阈值 |
| `CACHE_SIM_THRESHOLD` | cache.py | 0.92 | 缓存相似度阈值 |
