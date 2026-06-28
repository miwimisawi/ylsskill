"""
LLM Answer Generator.

Supports:
  - SiliconFlow API (default)
  - OpenAI-compatible API with custom BASE_URL
"""
from typing import Optional
from openai import OpenAI
from src.config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, LLM_MODEL,
    OPENAI_API_KEY, OPENAI_BASE_URL,
)
from src.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """你是一名面向初级眼科住院医师的AI助手，工作场景是病房日常文书和临床诊疗。

回答原则：
1. 优先使用检索到的参考资料，必须标注来源（如【参考1】）
2. 如果参考资料相关度低，明确告知"本地知识库未找到直接答案"
3. 对于药物剂量、手术指征等关键决策，提醒医师核对最新指南或请示上级
4. 回答简洁专业，适合病房快速查阅
5. 文书模板类请求直接输出格式化模板
6. 遇到紧急情况（如急性闭角型青光眼、眼外伤）优先给出处理流程
"""


def _make_client(provider: str = "siliconflow") -> tuple:
    if provider == "openai" and OPENAI_API_KEY:
        kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        return OpenAI(**kwargs), "gpt-4o"
    return OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL), LLM_MODEL


def generate(
    query: str,
    context: str,
    provider: str = "siliconflow",
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stream: bool = False,
    token_callback=None,
) -> str:
    if provider == "local":
        return "[本地模型接口预留 - 请配置 Ollama/vLLM]"

    client, default_model = _make_client(provider)
    model = model or default_model
    log.info("generate | model=%s provider=%s stream=%s callback=%s",
             model, provider, stream, token_callback is not None)

    user_content = (
        f"## 参考资料\n{context}\n\n---\n\n## 问题\n{query}\n\n"
        "请根据以上参考资料回答问题。如参考资料不足，请说明并给出你的建议。"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    if stream or token_callback is not None:
        full = ""
        token_count = 0
        extra = {"enable_thinking": False} if provider == "siliconflow" else {}
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                stream=True,
                extra_body=extra if extra else None,
            )
            for chunk in resp:
                # Some providers send empty-choices chunks as heartbeats
                if not chunk.choices:
                    continue
                delta_obj = chunk.choices[0].delta
                delta = delta_obj.content or ""
                if not delta:
                    # Fallback: Qwen3 thinking mode puts content in reasoning_content
                    delta = getattr(delta_obj, "reasoning_content", "") or ""
                if delta:
                    token_count += 1
                    if stream:
                        print(delta, end="", flush=True)
                    if token_callback is not None:
                        token_callback(delta)
                    full += delta
        except Exception:
            log.exception("generate | streaming error after %d tokens", token_count)
            raise
        if stream:
            print()
        log.info("generate | done, tokens=%d chars=%d", token_count, len(full))
        if not full:
            log.warning("generate | answer is empty — model=%s may have returned no content", model)
        return full
    else:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            answer = resp.choices[0].message.content.strip()
            log.info("generate | done (non-stream), chars=%d", len(answer))
            return answer
        except Exception:
            log.exception("generate | non-stream error")
            raise
