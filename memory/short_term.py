"""
短期记忆：滑动窗口对话历史管理。
超出窗口时用 LLM 摘要压缩，保留关键上下文。
"""

import re
import logging
from typing import List, Dict, Optional, Set

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
        self._key_entities: Set[str] = set()

    def add_message(self, role: str, content: str):
        """添加一条消息"""
        self.messages.append({"role": role, "content": content})

        # 超出窗口时压缩
        if len(self.messages) > self.window_size * 2:
            self._compress()

    def _compress(self):
        """将窗口外的消息压缩为摘要（医学实体感知）"""
        overflow = self.messages[:-self.window_size]
        self.messages = self.messages[-self.window_size:]

        # 提取溢出部分的关键医学实体
        overflow_text = " ".join(msg["content"] for msg in overflow)
        entities = _extract_medical_entities(overflow_text)
        self._key_entities.update(entities)

        # 构建压缩 prompt（包含实体提示）
        history_text = ""
        for msg in overflow:
            role = "患者" if msg["role"] == "user" else "医生"
            history_text += f"{role}：{msg['content']}\n"

        entity_hint = ""
        if self._key_entities:
            entity_hint = (
                f"\n重要实体（必须保留）：{'、'.join(sorted(self._key_entities)[:15])}\n"
            )

        prompt = (
            f"请将以下对话历史压缩为一段简短的摘要，保留关键的医学信息"
            f"（症状、诊断、用药、检查结果等）：\n\n"
            f"{'之前的摘要：' + self.summary + chr(10) + chr(10) if self.summary else ''}"
            f"{entity_hint}"
            f"{history_text}\n"
            f"请输出摘要（不超过150字）："
        )

        new_summary = chat(prompt, temperature=0.1, max_tokens=256)
        if new_summary:
            self.summary = new_summary
            logger.info(f"ShortTermMemory: 对话压缩完成 ({len(overflow)} 条 → 摘要, "
                        f"{len(self._key_entities)} 个关键实体)")
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
        self._key_entities = set()

    def get_key_entities(self) -> Set[str]:
        """获取当前会话中提取的关键医学实体"""
        # 合并已提取的 + 当前窗口内的
        current_text = " ".join(msg["content"] for msg in self.messages)
        current_entities = _extract_medical_entities(current_text)
        return self._key_entities | current_entities

    def __len__(self):
        return len(self.messages)


# ─────────────────────────────────────────────
# 医学实体提取（基于规则）
# ─────────────────────────────────────────────

# 常见医学实体规则
MEDICAL_ENTITY_PATTERNS = [
    # 药物名（中文 + 英文后缀）
    r'[\u4e00-\u9fff]{2,6}(?:片|胶囊|注射液|粒|滴眼液|软膏|口服液)',
    # 检查项目
    r'(?:CT|MRI|X线|B超|血常规|尿常规|血生化|肺功能|肝功能|肾功能|心电图|ECG|'  
    r'甲状腺功能|血脂|血糖|糖化血红蛋白|HbA1c|电解质|凝血功能|D-二聚体)',
    # 诊断名称
    r'(?:糖尿病|高血压|冠心病|肺炎|哮喘|胃炎|胃溃疡|甲亢|甲减|痛风|贫血|气胸|肺栓塞|'
    r'心房颤动|心力衰竭|脊柱侧弯|腺癌|骨折|中风|脑梗|肾结石|胆结石|'
    r'肺结核|肥胖症|高血脂|动脉硬化)',
    # 症状
    r'(?:发热|咳嗽|头痛|胸闷|心悸|气促|腹痛|恶心|呕吐|眼花|头晕|'
    r'胸痛|关节痛|腰痛|乳房肿块|水肿|潴留|便血|血尿)',
    # 药品通用名
    r'(?:阿司匹林|美托洛尔|二甲双胍|尼美舒利|高血压药|降糖药|抗生素|'
    r'胰岛素|奥氮平|氨氯地平|贝那普利|左氧氟沙星|他汀)',
]

_ENTITY_RE = re.compile('|'.join(MEDICAL_ENTITY_PATTERNS))


def _extract_medical_entities(text: str) -> Set[str]:
    """从文本中提取关键医学实体"""
    if not text:
        return set()
    matches = _ENTITY_RE.findall(text)
    return set(m for m in matches if len(m) >= 2)
