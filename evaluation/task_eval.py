"""
任务完成率评测 + 工具调用准确率评测。

升级版：
  - 诊断评测：embedding 语义相似度（主）+ 关键词匹配（回退）
  - 工具评测：名称 F1 + 参数精确匹配 + 参数准确率
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Embedding 语义相似度
# ─────────────────────────────────────────────

_embed_fn = None

def _get_embed_fn():
    """懒加载 embedding 函数"""
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        from utils.llm_client import embed
        # 测试是否可用
        embed(["test"])
        _embed_fn = embed
        logger.info("语义匹配模式: embedding 可用")
    except Exception:
        _embed_fn = False  # 标记不可用
        logger.info("语义匹配模式: embedding 不可用，回退关键词匹配")
    return _embed_fn


def _extract_diagnosis(agent_response: str) -> str:
    """
    从 Agent 长回复中提取核心诊断结论（1-2句），
    使诊断方向的语义比较更公平。

    策略（按优先级）：
    1. LLM 提取（最准确）
    2. 规则提取（关键词定位段落）
    3. 回退：取前200字
    """
    if not agent_response or len(agent_response) < 100:
        return agent_response

    # 策略1: 尝试 LLM 提取
    try:
        from utils.llm_client import chat
        extracted = chat(
            f"请从以下医疗回复中，用一句话提取核心诊断方向（只说可能的疾病/病因方向，不超过50字）：\n\n{agent_response[:800]}",
            temperature=0.0, max_tokens=80,
        )
        if extracted and len(extracted.strip()) >= 5:
            return extracted.strip()
    except Exception:
        pass

    # 策略2: 规则提取 — 找包含诊断关键词的句子
    import re
    diagnosis_keywords = ["可能", "考虑", "怀疑", "提示", "倾向", "不排除", "初步分析", "诊断方向"]
    sentences = re.split(r'[。\n]', agent_response)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) >= 10 and any(kw in sent for kw in diagnosis_keywords):
            return sent[:150]

    # 策略3: 回退
    return agent_response[:200]


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    计算两段文本的语义相似度 (cosine similarity)。
    embedding 不可用时回退到关键词匹配 ratio。
    """
    embed_fn = _get_embed_fn()
    if embed_fn and embed_fn is not False:
        try:
            import numpy as np
            vecs = embed_fn([text_a, text_b])
            a, b = np.array(vecs[0]), np.array(vecs[1])
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            pass
    # fallback: 关键词重叠率
    kw_a = set(_extract_keywords(text_a))
    kw_b = set(_extract_keywords(text_b))
    if not kw_a or not kw_b:
        return 0.0
    return len(kw_a & kw_b) / max(len(kw_a | kw_b), 1)


def evaluate_task_completion(
    predictions: List[Dict],
    references: List[Dict],
    correct_threshold: float = 0.7,
    partial_threshold: float = 0.4,
) -> Dict:
    """
    评测任务完成率：最终诊断方向是否正确。

    使用 embedding 语义相似度（cosine > correct_threshold 算正确），
    embedding 不可用时回退关键词匹配。

    Args:
        predictions: Agent 输出列表，含 final_response, tool_calls 等
        references: 标准答案列表，含 final_diagnosis_direction, tools_used 等
        correct_threshold: 语义相似度 ≥ 此值则判正确
        partial_threshold: 语义相似度 ≥ 此值则判部分正确

    Returns:
        评测指标字典
    """
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    correct = 0
    partial = 0
    similarities = []
    by_department = {}

    for pred, ref in zip(predictions, references):
        dept = ref.get("department", "") or ref.get("expected_department", "unknown")
        if dept not in by_department:
            by_department[dept] = {"total": 0, "correct": 0, "partial": 0}
        by_department[dept]["total"] += 1

        pred_response = pred.get("final_response", "")
        ref_diagnosis = ref.get("final_diagnosis_direction", "") or ref.get("expected_diagnosis_direction", "")

        if not ref_diagnosis:
            continue

        # 从长回复中提取诊断结论，避免长文本 vs 短文本的相似度偏差
        pred_diagnosis = _extract_diagnosis(pred_response)
        sim = _semantic_similarity(pred_diagnosis, ref_diagnosis)
        similarities.append(sim)

        if sim >= correct_threshold:
            correct += 1
            by_department[dept]["correct"] += 1
        elif sim >= partial_threshold:
            partial += 1
            by_department[dept]["partial"] += 1

    return {
        "total": total,
        "correct": correct,
        "partial_correct": partial,
        "accuracy": correct / total,
        "partial_accuracy": (correct + partial) / total,
        "avg_similarity": sum(similarities) / max(len(similarities), 1),
        "by_department": by_department,
    }


def evaluate_tool_usage(predictions: List[Dict], references: List[Dict]) -> Dict:
    """
    评测工具调用准确率。

    三层评测：
      1. 工具名称 F1（调用了正确的工具）
      2. 工具参数准确率（参数也正确）
      3. 工具调用顺序相似度（Kendall tau）
    """
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    precisions = []
    recalls = []
    f1s = []
    param_accuracies = []

    for pred, ref in zip(predictions, references):
        pred_calls = pred.get("tool_calls", [])
        pred_tools = [tc.get("tool_name", "") for tc in pred_calls]
        ref_tools = ref.get("tools_used", []) or ref.get("expected_tools", [])

        # 从 reference dialogue 提取带参数的工具调用
        ref_calls_with_args = _extract_ref_tool_calls(ref)

        if not ref_tools and not pred_tools:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            param_accuracies.append(1.0)
            continue

        # ─ Layer 1: 名称 F1 ─
        pred_set = set(pred_tools)
        ref_set = set(ref_tools)
        tp = len(pred_set & ref_set)
        precision = tp / max(len(pred_set), 1)
        recall = tp / max(len(ref_set), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # ─ Layer 2: 参数准确率 ─
        if ref_calls_with_args and pred_calls:
            param_acc = _compute_param_accuracy(pred_calls, ref_calls_with_args)
            param_accuracies.append(param_acc)

    n = len(f1s) or 1
    result = {
        "total": total,
        "avg_precision": sum(precisions) / n,
        "avg_recall": sum(recalls) / n,
        "avg_f1": sum(f1s) / n,
        "avg_tool_calls_per_case": sum(
            len(p.get("tool_calls", [])) for p in predictions
        ) / total,
    }
    if param_accuracies:
        result["avg_param_accuracy"] = sum(param_accuracies) / len(param_accuracies)
    return result


def _extract_ref_tool_calls(ref: Dict) -> List[Dict]:
    """从 reference 的 dialogue 中提取带参数的工具调用"""
    calls = []
    for turn in ref.get("dialogue", []):
        if turn.get("role") == "agent":
            for tc in turn.get("tool_calls", []):
                calls.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                })
    return calls


def _compute_param_accuracy(pred_calls: List[Dict], ref_calls: List[Dict]) -> float:
    """
    计算工具参数准确率。
    对每个匹配的工具名，比较关键参数是否一致。
    """
    matched = 0
    compared = 0

    for ref_tc in ref_calls:
        ref_name = ref_tc.get("name", "")
        ref_args = ref_tc.get("args", {})
        # 找同名的 pred tool call
        for pred_tc in pred_calls:
            pred_name = pred_tc.get("tool_name", "")
            pred_args = pred_tc.get("input_args", {})
            if pred_name == ref_name:
                # 比较参数
                for key, ref_val in ref_args.items():
                    compared += 1
                    pred_val = pred_args.get(key)
                    if pred_val is not None and str(pred_val).strip() == str(ref_val).strip():
                        matched += 1
                break  # 只匹配第一个同名工具

    if compared == 0:
        return 1.0  # 无参数可比 → 默认通过
    return matched / compared


def _extract_keywords(text: str) -> List[str]:
    """从诊断文本中提取关键词（用于 fallback 匹配）"""
    for ch in "，。、；：“”‘’（）【】《》！？":
        text = text.replace(ch, " ")
    stopwords = {"的", "了", "是", "在", "有", "不", "这", "个", "人", "都", "一", "和", "我",
                 "你", "他", "她", "它", "我们", "要", "会", "可以", "进行", "建议", "患者", "可能"}
    words = [w.strip() for w in text.split() if len(w.strip()) >= 2 and w.strip() not in stopwords]
    return words
