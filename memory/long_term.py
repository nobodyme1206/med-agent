"""
长期记忆：基于向量数据库的患者历史存储。
每次会话结束后自动生成摘要并存入 FAISS，支持跨会话检索。
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from utils.llm_client import chat, embed_query

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).parent.parent / "data" / "memory"


class LongTermMemory:
    """
    长期记忆：FAISS 向量存储 + 会话摘要。
    - 每次会话结束时，生成摘要并存入向量库
    - 新会话开始时，检索相关历史摘要作为上下文
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self._index = None
        self._records: List[Dict] = []
        self._loaded = False

        self.index_path = self.memory_dir / "memory.index"
        self.records_path = self.memory_dir / "memory_records.json"

    def _ensure_loaded(self):
        """懒加载索引"""
        if self._loaded:
            return

        if self.index_path.exists() and self.records_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self.index_path))
                with open(self.records_path, "r", encoding="utf-8") as f:
                    self._records = json.load(f)
                logger.info(f"长期记忆加载: {len(self._records)} 条记录")
            except Exception as e:
                logger.warning(f"长期记忆加载失败: {e}")
                self._index = None
                self._records = []
        else:
            self._records = []

        self._loaded = True

    def _save(self):
        """持久化索引和记录"""
        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(self.index_path))
        with open(self.records_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def store_session(self, session_summary: str, metadata: Dict = None):
        """
        存储一次会话摘要到长期记忆。

        Args:
            session_summary: 会话摘要文本
            metadata: 可选元数据（时间、科室、诊断等）
        """
        self._ensure_loaded()

        # 嵌入
        try:
            embedding = embed_query(session_summary)
        except Exception as e:
            logger.error(f"长期记忆嵌入失败: {e}")
            return

        vec = np.array([embedding], dtype=np.float32)

        # 初始化或追加 FAISS 索引
        if self._index is None:
            import faiss
            dim = len(embedding)
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(vec)

        record = {
            "id": len(self._records),
            "summary": session_summary,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._records.append(record)

        self._save()
        logger.info(f"长期记忆存储: #{record['id']} ({len(session_summary)} 字)")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索与查询相关的历史会话摘要。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相关会话记录列表
        """
        self._ensure_loaded()

        if self._index is None or self._index.ntotal == 0:
            return []

        try:
            query_vec = np.array([embed_query(query)], dtype=np.float32)
        except Exception as e:
            logger.error(f"长期记忆检索嵌入失败: {e}")
            return []

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._records):
                record = self._records[idx].copy()
                record["relevance_score"] = float(score)
                results.append(record)

        return results

    def generate_session_summary(self, messages: List[Dict], department: str = "") -> str:
        """
        从对话历史生成会话摘要。

        Args:
            messages: 对话消息列表
            department: 就诊科室

        Returns:
            会话摘要文本
        """
        history = "\n".join(
            f"{'患者' if m['role'] == 'user' else '医生'}：{m['content']}"
            for m in messages
            if m.get("content")
        )

        prompt = (
            f"请将以下{'[' + department + ']' if department else ''}问诊对话总结为一段摘要，"
            f"保留关键信息（主诉、诊断、用药、检查建议）。不超过100字。\n\n"
            f"{history}\n\n摘要："
        )

        summary = chat(prompt, temperature=0.1, max_tokens=200)
        return summary or f"[{department}] 问诊记录（摘要生成失败）"

    def reset(self):
        """清空内存中的索引和记录（不影响磁盘文件），用于评测时隔离各 case"""
        self._index = None
        self._records = []
        self._loaded = True  # 标记已加载，避免从磁盘重新读取

    @property
    def total_records(self) -> int:
        self._ensure_loaded()
        return len(self._records)
