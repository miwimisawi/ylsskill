"""
Query Enhancement: HyDE + Multi-Query reformulation.

HyDE: Generate a hypothetical answer → embed that for retrieval.
Multi-Query: Generate N reformulations → retrieve for each → union.
"""
from typing import List
from src.config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, LLM_MODEL
from openai import OpenAI


def _llm(prompt: str, system: str = "", max_tokens: int = 512, model: str = LLM_MODEL) -> str:
    client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def hyde(query: str) -> str:
    """
    HyDE: Generate a hypothetical medical document passage that would answer the query.
    Embed this passage instead of the raw query for better semantic alignment.
    """
    system = (
        "你是一名眼科住院医师助手。请根据问题，生成一段简短的、像教科书一样的医学参考答案段落（100-200字）。"
        "只输出段落本身，不要解释。"
    )
    passage = _llm(query, system=system, max_tokens=300)
    return passage


def multi_query(query: str, n: int = 3) -> List[str]:
    """
    Generate N alternative query formulations to improve recall.
    Returns list including the original query.
    """
    system = (
        "你是一名眼科AI助手的查询优化器。请将用户问题改写为{n}个不同表述，"
        "涵盖不同角度（中文、英文、医学术语）。每行一个，不加序号。"
    ).format(n=n)
    raw = _llm(query, system=system, max_tokens=200)
    alternatives = [line.strip() for line in raw.split("\n") if line.strip()][:n]
    return [query] + alternatives  # original always first


def step_back(query: str) -> str:
    """
    Step-back prompting: generate a more general background question
    to retrieve foundational context first.
    """
    system = (
        "你是眼科教学专家。请将以下具体临床问题提炼为一个更宏观的背景知识问题，"
        "用于先检索基础医学知识。只输出这一个问题，不加解释。"
    )
    return _llm(query, system=system, max_tokens=100)
