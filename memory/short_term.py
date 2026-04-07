"""
短期记忆：滑动窗口对话历史管理。
超出窗口时用 LLM 摘要压缩，保留关键上下文。
"""

import logging
from typing import List, Dict, Optional

from utils.llm_client import chat

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 10  # 最近 10 轮对话


class ShortTermMemory:
    """
    滑动窗口对话历史。
    - 保留最近 N 轮完整对话
    - 超出时将最早的对话压缩为摘要
    - 摘要作为 system message 前缀注入
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size
        self.messages: List[Dict[str, str]] = []
        self.summary: str = ""

    def add_message(self, role: str, content: str):
        """添加一条消息"""
        self.messages.append({"role": role, "content": content})

        # 超出窗口时压缩
        if len(self.messages) > self.window_size * 2:
            self._compress()

    def _compress(self):
        """将窗口外的消息压缩为摘要"""
        overflow = self.messages[:-self.window_size]
        self.messages = self.messages[-self.window_size:]

        # 构建压缩 prompt
        history_text = ""
        for msg in overflow:
            role = "患者" if msg["role"] == "user" else "医生"
            history_text += f"{role}：{msg['content']}\n"

        prompt = (
            f"请将以下对话历史压缩为一段简短的摘要，保留关键的医学信息"
            f"（症状、诊断、用药、检查结果等）：\n\n"
            f"{'之前的摘要：' + self.summary + chr(10) + chr(10) if self.summary else ''}"
            f"{history_text}\n"
            f"请输出摘要（不超过150字）："
        )

        new_summary = chat(prompt, temperature=0.1, max_tokens=256)
        if new_summary:
            self.summary = new_summary
            logger.info(f"ShortTermMemory: 对话压缩完成 ({len(overflow)} 条 → 摘要)")
        else:
            # LLM 失败时简单截断
            self.summary += " " + " | ".join(
                msg["content"][:50] for msg in overflow
            )

    def get_messages(self) -> List[Dict[str, str]]:
        """获取当前窗口内的消息（含摘要前缀）"""
        result = []
        if self.summary:
            result.append({
                "role": "system",
                "content": f"[对话历史摘要] {self.summary}",
            })
        result.extend(self.messages)
        return result

    def get_context_string(self) -> str:
        """获取完整上下文的字符串表示"""
        parts = []
        if self.summary:
            parts.append(f"[历史摘要] {self.summary}")
        for msg in self.messages:
            role = "患者" if msg["role"] == "user" else "医生"
            parts.append(f"{role}：{msg['content']}")
        return "\n".join(parts)

    def clear(self):
        """清空记忆"""
        self.messages = []
        self.summary = ""

    def __len__(self):
        return len(self.messages)
