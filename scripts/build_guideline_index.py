"""
诊疗指南 RAG 索引构建脚本。
将诊疗指南文本切块 → 嵌入 → 构建 FAISS + BM25 索引。

用法:
  python scripts/build_guideline_index.py --input data/guidelines/raw/ --output data/guidelines/
"""

import sys
import json
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_client import embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_texts(input_dir: Path) -> list:
    """加载原始文本文件"""
    texts = []
    for ext in ["*.txt", "*.md", "*.json"]:
        for f in input_dir.glob(ext):
            if f.suffix == ".json":
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                texts.append({
                                    "source": item.get("source", f.stem),
                                    "content": item.get("content", item.get("text", "")),
                                })
                            elif isinstance(item, str):
                                texts.append({"source": f.stem, "content": item})
            else:
                content = f.read_text(encoding="utf-8")
                texts.append({"source": f.stem, "content": content})
    logger.info(f"加载 {len(texts)} 个文档")
    return texts


def chunk_texts(texts: list, chunk_size: int = 512, overlap: int = 128) -> list:
    """知识点级别切块"""
    chunks = []
    for doc in texts:
        content = doc["content"]
        source = doc["source"]
        if not content.strip():
            continue

        # 按段落切分
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        "source": source,
                        "content": current_chunk.strip(),
                    })
                # 带重叠
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append({
                "source": source,
                "content": current_chunk.strip(),
            })

    logger.info(f"切块完成: {len(chunks)} 个 chunks")
    return chunks


def build_faiss_index(chunks: list, output_dir: Path):
    """构建 FAISS 向量索引"""
    import faiss

    texts = [c["content"] for c in chunks]
    logger.info(f"开始嵌入 {len(texts)} 个 chunks...")
    embeddings = embed(texts)

    dim = len(embeddings[0])
    vectors = np.array(embeddings, dtype=np.float32)

    # 归一化（用于余弦相似度）
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = output_dir / "knowledge.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 索引保存: {index_path} (dim={dim}, n={index.ntotal})")

    return index


def build_bm25_index(chunks: list, output_dir: Path):
    """构建 BM25 稀疏索引"""
    from rank_bm25 import BM25Okapi

    # 字符级分词（中文）
    tokenized = [list(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    bm25_path = output_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 索引保存: {bm25_path}")

    return bm25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/guidelines/raw/")
    parser.add_argument("--output", type=str, default="data/guidelines/")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.warning(f"输入目录不存在: {input_dir}，请先准备诊疗指南文本")
        return

    # 1. 加载文本
    texts = load_texts(input_dir)
    if not texts:
        logger.warning("无可用文本")
        return

    # 2. 切块
    chunks = chunk_texts(texts, args.chunk_size, args.overlap)

    # 3. 保存 chunks
    chunks_path = output_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Chunks 保存: {chunks_path}")

    # 4. 构建索引
    build_faiss_index(chunks, output_dir)
    build_bm25_index(chunks, output_dir)

    # 5. 元数据
    meta = {
        "num_chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "embedding_model": "GLM-Embedding-3",
    }
    with open(output_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"索引构建完成: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
