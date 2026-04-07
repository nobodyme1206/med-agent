"""
统一LLM客户端：封装 OpenAI-compatible API
支持：chat completion（含 function calling）、text embedding
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# 全局 token 计数器（单次请求内累计）
_token_counter = {"total": 0}

def get_token_usage() -> int:
    return _token_counter["total"]

def reset_token_usage():
    _token_counter["total"] = 0


# 自动加载 .env
def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

API_KEY = os.environ.get("PARATERA_API_KEY", "")
BASE_URL = os.environ.get("PARATERA_BASE_URL", "https://ai.paratera.com/v1")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "Qwen2.5-7B-Instruct")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "GLM-Embedding-3")


def _get_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai：pip install openai")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ─────────────────────────────────────────────
# Chat Completion
# ─────────────────────────────────────────────

def chat(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    max_retries: int = 3,
    system: Optional[str] = None,
) -> Optional[str]:
    """调用chat completion，返回文本。失败返回None。"""
    client = _get_client()
    model = model or CHAT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if hasattr(resp, "usage") and resp.usage:
                _token_counter["total"] += getattr(resp.usage, "total_tokens", 0)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"chat() 失败（{e}），重试 {attempt+1}/{max_retries}")
            time.sleep(2 ** attempt)
    return None


def chat_with_messages(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    max_retries: int = 3,
    tools: Optional[List[Dict]] = None,
) -> Optional[Dict[str, Any]]:
    """
    多轮对话 + Function Calling 支持。
    返回完整的 message dict（含 content 和 tool_calls）。
    """
    client = _get_client()
    model = model or CHAT_MODEL

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            if hasattr(resp, "usage") and resp.usage:
                _token_counter["total"] += getattr(resp.usage, "total_tokens", 0)
            msg = resp.choices[0].message
            result = {"content": msg.content or "", "role": "assistant"}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return result
        except Exception as e:
            logger.warning(f"chat_with_messages() 失败（{e}），重试 {attempt+1}/{max_retries}")
            time.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────────
# Text Embedding
# ─────────────────────────────────────────────

def embed(texts: List[str], model: Optional[str] = None, batch_size: int = 64) -> List[List[float]]:
    """批量获取文本嵌入向量。"""
    client = _get_client()
    model = model or EMBED_MODEL
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                batch_embs = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_embs)
                break
            except Exception as e:
                logger.warning(f"embed() batch {i//batch_size} 失败（{e}），重试 {attempt+1}/3")
                time.sleep(2 ** attempt)
        else:
            logger.error(f"embed() batch {i//batch_size} 彻底失败，补零向量")
            all_embeddings.extend([[0.0] * 2048] * len(batch))

        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"  嵌入进度：{min(i+batch_size, len(texts))}/{len(texts)}")

    return all_embeddings


def embed_query(text: str, model: Optional[str] = None) -> List[float]:
    """单条查询嵌入（检索时使用）。"""
    return embed([text], model=model)[0]
