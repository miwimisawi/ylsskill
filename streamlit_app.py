"""
眼科住院助手 · Streamlit 界面

启动方式：
    $env:PYTHONUTF8="1"
    D:\miniconda\envs\openai\python.exe -m streamlit run streamlit_app.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.indexer import load_index
from src.pipeline import run
from src.config import AVAILABLE_MODELS, LLM_MODEL

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="眼科住院助手",
    page_icon="👁",
    layout="wide",
)

# ── Load index (cached across reruns) ───────────────────────────────────────
@st.cache_resource(show_spinner="正在加载知识库索引，首次约需 15 秒…")
def get_index():
    return load_index()

col_chat, col_debug = st.columns([3, 2])

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("👁 眼科住院助手")
    st.caption("可回答眼科诊疗问题、生成病历文书模板\n内容来自 BCSC、教材及科室内部规范")
    st.divider()

    st.subheader("模型选择")
    model_labels = [m["label"] for m in AVAILABLE_MODELS]
    model_ids    = [m["id"]    for m in AVAILABLE_MODELS]
    selected_idx = st.selectbox("LLM", range(len(model_labels)),
                                format_func=lambda i: model_labels[i], index=0)
    selected_model = model_ids[selected_idx]

    st.divider()
    st.subheader("查询增强")
    use_hyde       = st.checkbox("HyDE（假设性文档扩展）", value=True)
    use_multi      = st.checkbox("多查询", value=False)
    use_stepback   = st.checkbox("背景提炼（Step-back）", value=False)

    st.divider()
    st.subheader("调试模式")
    show_debug = st.checkbox("显示 Pipeline 详情", value=True)

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("v1.0 · SiliconFlow RAG")

# ── Chat history ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggestion chips (shown when chat is empty) ──────────────────────────────
SUGGESTIONS = [
    "泪道阻塞的手术治疗方案",
    "给我一个VTE评分病程模板",
    "急性闭角型青光眼如何紧急处理",
    "眼眶蜂窝织炎的抗生素选择",
]

with col_chat:
    st.header("对话", divider="gray")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "👁"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                m = msg["meta"]
                conf_color = {"high":"green","medium":"orange","low":"red","cached":"blue"}.get(m.get("confidence",""),"gray")
                src_text = " · ".join(
                    f"`{s.get('db','?')}`" for s in m.get("sources",[])
                ) or "—"
                st.caption(
                    f"置信度 :{conf_color}[**{m.get('confidence','—').upper()}**] · "
                    f"耗时 {m.get('total_time','—')}s · 来源：{src_text}"
                )

    # Suggestion chips when empty
    if not st.session_state.messages:
        st.markdown("**快速提问：**")
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            if cols[i % 2].button(s, use_container_width=True, key=f"sug_{i}"):
                st.session_state._pending_query = s
                st.rerun()

    # Handle pending suggestion
    pending = st.session_state.pop("_pending_query", None)

    # Chat input
    user_input = st.chat_input("输入问题，如：急性泪囊炎如何治疗？") or pending

with col_debug:
    if show_debug:
        st.header("Pipeline 详情", divider="gray")
        debug_placeholder = st.empty()
        if not st.session_state.messages:
            debug_placeholder.info("💡 发送问题后这里会显示 Pipeline 详情")
    else:
        debug_placeholder = None

# ── Process query ─────────────────────────────────────────────────────────────
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with col_chat:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="👁"):
            answer_placeholder = st.empty()
            answer_placeholder.markdown("*思考中…*")

    # Load index (cached)
    try:
        index = get_index()
        col, bm25, bm25_docs = index
    except Exception as e:
        st.error(f"索引加载失败：{e}")
        st.stop()

    # Run pipeline
    t0 = time.time()
    try:
        result = run(
            query=user_input,
            col=col, bm25=bm25, bm25_docs=bm25_docs,
            model=selected_model,
            use_hyde=use_hyde,
            use_multi_query=use_multi,
            use_step_back=use_stepback,
            debug=False,
        )
    except Exception as e:
        with col_chat:
            st.error(f"Pipeline 错误：{e}")
        st.stop()

    elapsed = round(time.time() - t0, 1)
    answer  = result.get("answer", "（无答案）")
    sources = result.get("sources", [])
    conf    = result.get("confidence", "—")
    from_cache = result.get("from_cache", False)
    di      = result.get("debug_info", {})
    timings = di.get("timings", {})

    # Display answer with simulated streaming
    with col_chat:
        answer_placeholder.markdown(answer)
        conf_color = {"high":"green","medium":"orange","low":"red","cached":"blue"}.get(conf,"gray")
        src_text = " · ".join(f"`{s.get('db','?')}`" for s in sources) or "—"
        st.caption(
            f"置信度 :{conf_color}[**{conf.upper()}**] · "
            f"耗时 **{elapsed}s** · {'✅ 缓存命中' if from_cache else '🔍 全流水线'} · "
            f"来源：{src_text}"
        )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {"confidence": conf, "sources": sources, "total_time": elapsed},
    })

    # Debug panel
    if show_debug and debug_placeholder is not None:
        with debug_placeholder.container():
            # Summary
            with st.expander("📋 摘要", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("置信度", conf.upper())
                c2.metric("CRAG 最高分", f"{di.get('crag_best_score', 0):.4f}" if di.get('crag_best_score') else "—")
                c1.metric("检索命中数", di.get("retrieval_count", "—"))
                c2.metric("来自缓存", "是" if from_cache else "否")
                st.caption(f"模型：{selected_model}")
                st.caption(f"Cache ID：{result.get('cache_id','—')}")

            # Timings
            if timings:
                with st.expander("⏱ 耗时分析", expanded=True):
                    import pandas as pd
                    rows = [(k, v) for k, v in timings.items()]
                    df = pd.DataFrame(rows, columns=["阶段", "耗时(s)"])
                    st.bar_chart(df.set_index("阶段"))
                    for k, v in timings.items():
                        st.caption(f"{k}: **{v}s**")

            # Search queries
            sq = di.get("search_queries", [])
            if sq:
                with st.expander(f"🔍 搜索查询（{len(sq)} 条）", expanded=False):
                    for i, q in enumerate(sq):
                        label = "原始" if i == 0 else f"改写 #{i}"
                        st.markdown(f"**{label}：** {q}")

            # Sources
            if sources:
                with st.expander(f"📚 引用来源（{len(sources)} 条）", expanded=True):
                    for i, s in enumerate(sources):
                        st.markdown(
                            f"**#{i+1}** `{s.get('db','?')}` — "
                            f"score=**{s.get('score',0):.4f}** · {s.get('source','')}"
                        )
