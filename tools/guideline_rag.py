"""
诊疗指南 RAG 检索工具：混合检索（BM25 + Dense + RRF + Reranker）。
复用 MedBench 的混合检索架构，chunk 粒度改为"知识点级别"。
"""

import sys
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_client import embed_query as api_embed_query

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "guidelines"


class GuidelineRetriever:
    """
    医学诊疗指南混合检索器：
      1. Dense: GLM-Embedding-3 语义检索（FAISS IndexFlatIP）
      2. Sparse: BM25 关键词检索
      3. Fusion: RRF (Reciprocal Rank Fusion, k=60)
      4. Rerank: BGE-Reranker-v2-m3 精排 + 分数过滤
    """

    def __init__(
        self,
        index_dir: Path = DATA_DIR,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        final_top_k: int = 3,
        score_threshold: float = 0.35,
    ):
        self.index_dir = index_dir
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.score_threshold = score_threshold

        self._faiss_index = None
        self._bm25 = None
        self._chunks = None
        self._reranker = None
        self._loaded = False

    def _ensure_loaded(self):
        """懒加载索引和模型"""
        if self._loaded:
            return

        index_path = self.index_dir / "knowledge.index"
        bm25_path = self.index_dir / "bm25.pkl"
        chunks_path = self.index_dir / "chunks.pkl"

        if not all(p.exists() for p in [index_path, bm25_path, chunks_path]):
            logger.warning(f"索引文件不完整: {self.index_dir}，检索将返回空结果")
            self._loaded = True
            return

        import faiss
        self._faiss_index = faiss.read_index(str(index_path))
        with open(bm25_path, "rb") as f:
            self._bm25 = pickle.load(f)
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)

        logger.info(
            f"索引加载完成: {len(self._chunks)} chunks, "
            f"FAISS dim={self._faiss_index.d}"
        )

        # 加载 Reranker（使用 sentence-transformers CrossEncoder 兼容 transformers 5.x）
        if self.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(
                    self.reranker_model, trust_remote_code=True
                )
                logger.info(f"Reranker 加载完成: {self.reranker_model}")
            except Exception as e:
                logger.warning(f"Reranker 加载失败: {e}，将跳过精排")

        self._loaded = True

    def _dense_search(self, query: str, top_k: int) -> List[int]:
        """密集向量检索"""
        if self._faiss_index is None:
            return []
        query_vec = np.array([api_embed_query(query)], dtype=np.float32)
        scores, indices = self._faiss_index.search(query_vec, top_k)
        return [int(idx) for idx in indices[0] if idx >= 0]

    def _sparse_search(self, query: str, top_k: int) -> List[int]:
        """BM25 稀疏检索"""
        if self._bm25 is None:
            return []
        tokens = list(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [int(idx) for idx in top_indices if scores[idx] > 0]

    def _rrf_fusion(self, dense_ids: List[int], sparse_ids: List[int], k: int = 60) -> List[int]:
        """Reciprocal Rank Fusion"""
        scores = {}
        for rank, idx in enumerate(dense_ids):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(sparse_ids):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids

    def _rerank(self, query: str, candidates: List[Dict]) -> List[Tuple[Dict, float]]:
        """BGE-Reranker 精排 + 分数过滤"""
        if not self._reranker or not candidates:
            return [(c, 0.0) for c in candidates]
        pairs = [[query, c["content"]] for c in candidates]
        scores = self._reranker.predict(pairs)
        if isinstance(scores, float):
            scores = [scores]
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        filtered = [(c, s) for s, c in ranked if s >= self.score_threshold]
        return filtered

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        执行混合检索，返回最相关的知识块列表。

        Args:
            query: 查询文本
            top_k: 返回数量（默认使用 self.final_top_k）

        Returns:
            知识块列表，每块含 source, content, score 等字段
        """
        self._ensure_loaded()
        if self._chunks is None:
            return []

        top_k = top_k or self.final_top_k

        # 1. 双通道检索
        dense_ids = self._dense_search(query, self.dense_top_k)
        sparse_ids = self._sparse_search(query, self.sparse_top_k)

        # 2. RRF 融合
        fused_ids = self._rrf_fusion(dense_ids, sparse_ids)
        candidates = [self._chunks[i] for i in fused_ids if i < len(self._chunks)]

        # 3. Reranker 精排 + 分数过滤
        if self.use_reranker and self._reranker:
            scored = self._rerank(query, candidates)
            candidates = [c for c, _ in scored]

        return candidates[:top_k]


# 全局检索器实例（懒加载）
_retriever: Optional[GuidelineRetriever] = None


def get_retriever() -> GuidelineRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GuidelineRetriever()
    return _retriever


def search_guidelines(query: str) -> Dict:
    """
    检索诊疗指南相关知识。

    Args:
        query: 医学问题或症状描述

    Returns:
        检索到的知识块列表，每块包含来源和内容。
    """
    retriever = get_retriever()
    chunks = retriever.retrieve(query)

    if not chunks:
        return {
            "found": False,
            "message": "未检索到相关诊疗指南内容",
            "chunks": [],
        }

    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "rank": i + 1,
            "source": chunk.get("source", ""),
            "content": chunk.get("content", ""),
        })

    return {
        "found": True,
        "num_results": len(results),
        "chunks": results,
    }


# ─────────────────────────────────────────────
# 工具注册定义
# ─────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_guidelines",
        "description": "检索医学诊疗指南和专业知识，输入症状或医学问题，返回相关的参考知识。用于辅助诊断和治疗方案制定。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "医学问题或症状描述，如'高血压头晕鉴别诊断'、'2型糖尿病一线用药'",
                },
            },
            "required": ["query"],
        },
        "handler": search_guidelines,
        "category": "rag",
    },
]
