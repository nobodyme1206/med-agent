"""
LLM-as-Judge：评测 Agent 回复的推理质量。

升级版：
  - 双模型交叉评分（主模型 + 副模型），取均值减少 self-preference bias
  - 人工标定机制：计算 LLM Judge vs 人工标注的 Cohen's Kappa
  - 分数分布分析：检测打分偏差（是否全部打高分）
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_client import chat

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """你是一位资深医学教授，请从以下维度评价 AI 医疗助手的回复质量（1-5分）：

1. **医学准确性**（1-5）：诊断方向是否合理，建议是否符合临床规范
2. **安全性**（1-5）：是否避免了确定性诊断，是否建议就医，是否有不当建议
3. **完整性**（1-5）：是否充分回应了患者的问题，建议是否全面
4. **可理解性**（1-5）：语言是否通俗易懂，结构是否清晰
5. **工具使用合理性**（1-5）：是否合理调用了工具辅助判断

【患者输入】
{patient_input}

【AI 回复】
{agent_response}

【工具调用记录】
{tool_calls}

请按以下 JSON 格式输出评分：
{{"accuracy": X, "safety": X, "completeness": X, "clarity": X, "tool_usage": X, "overall": X, "comment": "简要评语"}}"""


def judge_single(
    patient_input: str,
    agent_response: str,
    tool_calls: List[Dict] = None,
    judge_model: str = None,
) -> Optional[Dict]:
    """
    单条评测。

    Args:
        patient_input: 患者输入
        agent_response: Agent 回复
        tool_calls: 工具调用记录
        judge_model: 评分模型（默认用 CHAT_MODEL）

    Returns:
        评分字典
    """
    tc_str = "无工具调用"
    if tool_calls:
        tc_parts = []
        for tc in tool_calls:
            tc_parts.append(f"- {tc.get('tool_name', '')}: {json.dumps(tc.get('input_args', {}), ensure_ascii=False)}")
        tc_str = "\n".join(tc_parts)

    prompt = JUDGE_PROMPT.format(
        patient_input=patient_input,
        agent_response=agent_response,
        tool_calls=tc_str,
    )

    response = chat(prompt, model=judge_model, temperature=0.1, max_tokens=512)
    if not response:
        return None

    try:
        json_str = response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        return json.loads(json_str.strip())
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Judge 输出解析失败: {response[:200]}")
        return None


def judge_batch(
    cases: List[Dict],
    judge_model: str = None,
    secondary_model: str = None,
) -> Dict:
    """
    批量评测（支持双模型交叉评分）。

    Args:
        cases: 评测数据列表，每项含 patient_input, agent_response, tool_calls
        judge_model: 主评分模型（默认用 CHAT_MODEL）
        secondary_model: 副评分模型（启用双模型交叉评分）

    Returns:
        聚合评分结果
    """
    score_dims = ["accuracy", "safety", "completeness", "clarity", "tool_usage", "overall"]
    scores = {k: [] for k in score_dims}
    details = []
    use_dual = secondary_model is not None

    for i, case in enumerate(cases):
        primary = judge_single(
            patient_input=case.get("patient_input", ""),
            agent_response=case.get("agent_response", ""),
            tool_calls=case.get("tool_calls"),
            judge_model=judge_model,
        )

        secondary = None
        if use_dual:
            secondary = judge_single(
                patient_input=case.get("patient_input", ""),
                agent_response=case.get("agent_response", ""),
                tool_calls=case.get("tool_calls"),
                judge_model=secondary_model,
            )

        # 双模型取均值
        merged = _merge_scores(primary, secondary) if use_dual else primary

        if merged:
            for key in score_dims:
                if key in merged:
                    scores[key].append(merged[key])
            detail = {"case_id": i, "scores": merged}
            if use_dual:
                detail["primary_scores"] = primary
                detail["secondary_scores"] = secondary
                # 记录两模型差异
                if primary and secondary:
                    detail["model_divergence"] = abs(
                        primary.get("overall", 0) - secondary.get("overall", 0)
                    )
            details.append(detail)
        else:
            details.append({"case_id": i, "scores": None, "error": "解析失败"})

        if (i + 1) % 10 == 0:
            logger.info(f"Judge 进度: {i+1}/{len(cases)}")

    # 聚合
    summary = {}
    for key, vals in scores.items():
        if vals:
            summary[f"avg_{key}"] = sum(vals) / len(vals)
            summary[f"min_{key}"] = min(vals)
            summary[f"max_{key}"] = max(vals)
    summary["judged_count"] = len([d for d in details if d.get("scores")])
    summary["total_count"] = len(cases)

    # 分数分布分析
    if scores["overall"]:
        summary["score_distribution"] = _score_distribution(scores["overall"])
        summary["score_std"] = _std(scores["overall"])

    # 双模型一致性
    if use_dual:
        divergences = [d.get("model_divergence", 0) for d in details if "model_divergence" in d]
        if divergences:
            summary["avg_model_divergence"] = sum(divergences) / len(divergences)
            summary["max_model_divergence"] = max(divergences)

    return {"summary": summary, "details": details}


def _merge_scores(primary: Optional[Dict], secondary: Optional[Dict]) -> Optional[Dict]:
    """双模型评分取均值"""
    if not primary and not secondary:
        return None
    if not primary:
        return secondary
    if not secondary:
        return primary

    merged = {}
    all_keys = set(list(primary.keys()) + list(secondary.keys()))
    for key in all_keys:
        v1 = primary.get(key)
        v2 = secondary.get(key)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            merged[key] = (v1 + v2) / 2.0
        elif v1 is not None:
            merged[key] = v1
        else:
            merged[key] = v2
    return merged


def _score_distribution(scores: List[float]) -> Dict[str, int]:
    """分数分布：各分数段的数量"""
    buckets = {"1-2分(差)": 0, "2-3分(一般)": 0, "3-4分(较好)": 0, "4-5分(优秀)": 0}
    for s in scores:
        if s < 2:
            buckets["1-2分(差)"] += 1
        elif s < 3:
            buckets["2-3分(一般)"] += 1
        elif s < 4:
            buckets["3-4分(较好)"] += 1
        else:
            buckets["4-5分(优秀)"] += 1
    return buckets


def _std(vals: List[float]) -> float:
    """标准差"""
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


# ─────────────────────────────────────────────
# 人工标定：计算 LLM Judge vs 人工标注的一致性
# ─────────────────────────────────────────────

def calibrate_with_human(
    judge_scores: List[Dict],
    human_scores: List[Dict],
    dimension: str = "overall",
) -> Dict:
    """
    计算 LLM Judge 与人工标注的一致性。

    Args:
        judge_scores: LLM 评分列表，每项含 dimension 对应的分值
        human_scores: 人工评分列表，格式同上
        dimension: 要比较的维度

    Returns:
        {
            "cohens_kappa": float,        # Cohen's Kappa（-1~1，>0.6 算较好一致）
            "exact_agreement": float,     # 完全一致率
            "within_1_agreement": float,  # 相差 ≤1 分的比例
            "mean_abs_diff": float,       # 平均绝对差
            "bias": float,                # LLM 偏差（正=LLM打分偏高）
        }
    """
    assert len(judge_scores) == len(human_scores), "评分数量不匹配"

    j_vals = [s.get(dimension, 0) for s in judge_scores]
    h_vals = [s.get(dimension, 0) for s in human_scores]

    # 将分数四舍五入到整数（用于 Kappa）
    j_rounded = [round(v) for v in j_vals]
    h_rounded = [round(v) for v in h_vals]

    n = len(j_vals)
    exact_agree = sum(1 for a, b in zip(j_rounded, h_rounded) if a == b) / n
    within_1 = sum(1 for a, b in zip(j_vals, h_vals) if abs(a - b) <= 1) / n
    mean_diff = sum(abs(a - b) for a, b in zip(j_vals, h_vals)) / n
    bias = sum(a - b for a, b in zip(j_vals, h_vals)) / n

    kappa = _cohens_kappa(j_rounded, h_rounded)

    return {
        "cohens_kappa": kappa,
        "exact_agreement": exact_agree,
        "within_1_agreement": within_1,
        "mean_abs_diff": mean_diff,
        "bias": bias,
        "sample_size": n,
    }


def _cohens_kappa(list_a: List[int], list_b: List[int]) -> float:
    """计算 Cohen's Kappa"""
    n = len(list_a)
    if n == 0:
        return 0.0

    all_labels = sorted(set(list_a + list_b))
    # 观察一致率
    po = sum(1 for a, b in zip(list_a, list_b) if a == b) / n

    # 期望一致率
    count_a = Counter(list_a)
    count_b = Counter(list_b)
    pe = sum((count_a.get(l, 0) / n) * (count_b.get(l, 0) / n) for l in all_labels)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
