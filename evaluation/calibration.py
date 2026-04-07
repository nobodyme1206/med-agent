"""
置信度标定模块：分析 Agent 输出置信度与实际正确率的校准关系。

功能：
  1. 按置信度分桶，计算每个桶内的实际正确率 → 校准曲线
  2. 计算 ECE（Expected Calibration Error）
  3. 寻找最优阈值（最大化 F1 / 最小化误诊+漏诊）
  4. 输出校准报告 + 建议阈值

用法:
  from evaluation.calibration import CalibrationAnalyzer
  analyzer = CalibrationAnalyzer(predictions, references)
  report = analyzer.run()
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """
    置信度校准分析器。

    输入：
        predictions: Agent 输出列表，每项含 confidence, final_response 等
        references: 标准答案列表，含 final_diagnosis_direction 等
        correctness: 可选，预计算的正确性列表 (bool)
    """

    def __init__(
        self,
        predictions: List[Dict],
        references: List[Dict],
        correctness: List[bool] = None,
        num_bins: int = 10,
    ):
        self.predictions = predictions
        self.references = references
        self.num_bins = num_bins

        # 提取置信度
        self.confidences = [
            p.get("confidence", 0.0) for p in predictions
        ]

        # 如果没有提供正确性，用简单匹配判断
        if correctness is not None:
            self.correctness = correctness
        else:
            self.correctness = self._compute_correctness()

    def _compute_correctness(self) -> List[bool]:
        """简单判断每个 case 是否正确（关键词重叠）"""
        results = []
        for pred, ref in zip(self.predictions, self.references):
            pred_text = pred.get("final_response", "")
            ref_text = ref.get("final_diagnosis_direction", "")
            if not ref_text:
                results.append(True)  # 无标准答案时默认正确
                continue
            # 简单重叠检查
            ref_chars = set(ref_text) - set("，。、；：的了是在有不")
            pred_chars = set(pred_text)
            overlap = len(ref_chars & pred_chars) / max(len(ref_chars), 1)
            results.append(overlap >= 0.4)
        return results

    def calibration_curve(self) -> List[Dict]:
        """
        按置信度分桶，计算每桶的实际正确率。

        Returns:
            [{"bin_lower": 0.0, "bin_upper": 0.1, "avg_confidence": 0.05,
              "accuracy": 0.3, "count": 20}, ...]
        """
        bins = []
        bin_width = 1.0 / self.num_bins

        for i in range(self.num_bins):
            lower = i * bin_width
            upper = (i + 1) * bin_width

            indices = [
                j for j, c in enumerate(self.confidences)
                if lower <= c < upper or (i == self.num_bins - 1 and c == upper)
            ]

            if not indices:
                bins.append({
                    "bin_lower": round(lower, 2),
                    "bin_upper": round(upper, 2),
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                    "count": 0,
                })
                continue

            avg_conf = sum(self.confidences[j] for j in indices) / len(indices)
            acc = sum(1 for j in indices if self.correctness[j]) / len(indices)

            bins.append({
                "bin_lower": round(lower, 2),
                "bin_upper": round(upper, 2),
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(acc, 4),
                "count": len(indices),
            })

        return bins

    def expected_calibration_error(self) -> float:
        """
        计算 ECE（Expected Calibration Error）。
        ECE = Σ (|accuracy_i - confidence_i| * n_i / N)
        ECE 越小越好，0 = 完美校准。
        """
        curve = self.calibration_curve()
        n_total = len(self.confidences)
        if n_total == 0:
            return 0.0

        ece = 0.0
        for b in curve:
            if b["count"] > 0:
                ece += abs(b["accuracy"] - b["avg_confidence"]) * b["count"] / n_total
        return round(ece, 4)

    def find_optimal_threshold(self) -> Dict:
        """
        寻找最优置信度阈值。

        策略：遍历阈值候选，选择使 F1 最大的阈值。
        - confidence >= threshold → 正常输出（预测为"可信"）
        - confidence < threshold → 建议就医 / 人工接管（预测为"不可信"）

        Returns:
            {"optimal_threshold": float, "best_f1": float, "all_thresholds": [...]}
        """
        candidates = [i / 20.0 for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95
        results = []

        for thresh in candidates:
            tp = fp = tn = fn = 0
            for conf, correct in zip(self.confidences, self.correctness):
                predicted_trustworthy = conf >= thresh
                if predicted_trustworthy and correct:
                    tp += 1
                elif predicted_trustworthy and not correct:
                    fp += 1  # 高置信但错了（最危险）
                elif not predicted_trustworthy and not correct:
                    tn += 1  # 低置信且确实错了（正确兜底）
                else:
                    fn += 1  # 低置信但其实对了（过度保守）

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            escalation_rate = (tn + fn) / max(len(self.confidences), 1)

            results.append({
                "threshold": thresh,
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "escalation_rate": round(escalation_rate, 4),
                "dangerous_errors": fp,  # 高置信但错误的数量
            })

        # 选 F1 最高的
        best = max(results, key=lambda x: x["f1"])

        return {
            "optimal_threshold": best["threshold"],
            "best_f1": best["f1"],
            "escalation_rate_at_optimal": best["escalation_rate"],
            "dangerous_errors_at_optimal": best["dangerous_errors"],
            "all_thresholds": results,
        }

    def run(self) -> Dict:
        """运行完整校准分析"""
        curve = self.calibration_curve()
        ece = self.expected_calibration_error()
        threshold = self.find_optimal_threshold()

        report = {
            "calibration_curve": curve,
            "ece": ece,
            "threshold_analysis": threshold,
            "summary": {
                "total_cases": len(self.confidences),
                "avg_confidence": round(
                    sum(self.confidences) / max(len(self.confidences), 1), 4
                ),
                "overall_accuracy": round(
                    sum(self.correctness) / max(len(self.correctness), 1), 4
                ),
                "ece": ece,
                "optimal_threshold": threshold["optimal_threshold"],
                "current_threshold": 0.6,  # 当前硬编码阈值
                "recommendation": self._recommendation(ece, threshold),
            },
        }

        logger.info(f"校准分析完成: ECE={ece:.4f}, 最优阈值={threshold['optimal_threshold']}")
        return report

    def _recommendation(self, ece: float, threshold_result: Dict) -> str:
        """生成校准建议"""
        optimal = threshold_result["optimal_threshold"]
        parts = []

        if ece < 0.05:
            parts.append("校准良好 (ECE < 0.05)")
        elif ece < 0.15:
            parts.append(f"校准一般 (ECE={ece:.3f})，建议进行温度缩放")
        else:
            parts.append(f"校准较差 (ECE={ece:.3f})，置信度不可靠，需重新训练或后处理")

        if abs(optimal - 0.6) > 0.1:
            parts.append(f"建议将兜底阈值从 0.6 调整为 {optimal}")
        else:
            parts.append(f"当前阈值 0.6 接近最优值 {optimal}，无需调整")

        return "；".join(parts)

    def save(self, path: str):
        """保存校准报告"""
        report = self.run()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"校准报告保存: {path}")
