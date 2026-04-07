"""
消融实验脚本：对比不同模型/配置下的评测指标。

对比的 5 个模型：
  1. base       - Qwen2.5-7B-Instruct 原始模型
  2. sft        - Agentic SFT 后
  3. grpo_r1    - GRPO 第一轮
  4. grpo_r2    - GRPO 第二轮（bad case 补数据后）
  5. no_tool    - 去掉工具调用的消融
  6. no_rag     - 去掉 RAG 检索的消融
  7. react_1    - ReAct 循环限制为 1 轮
  8. react_5    - ReAct 循环限制为 5 轮

用法:
  python scripts/run_ablation.py --eval_data data/eval/cases_500.json --output results/ablation/
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 消融配置
# ─────────────────────────────────────────────

ABLATION_CONFIGS = {
    "base": {
        "description": "Qwen2.5-7B-Instruct 原始模型",
        "model_path": None,  # 使用默认 API 模型
        "use_tools": True,
        "use_rag": True,
        "max_loops": 3,
    },
    "sft": {
        "description": "Agentic SFT 后",
        "model_path": "models/sft_merged/",
        "use_tools": True,
        "use_rag": True,
        "max_loops": 3,
    },
    "grpo_r1": {
        "description": "GRPO 第一轮",
        "model_path": "models/grpo_r1_merged/",
        "use_tools": True,
        "use_rag": True,
        "max_loops": 3,
    },
    "grpo_r2": {
        "description": "GRPO 第二轮 (bad case 补数据)",
        "model_path": "models/grpo_r2_merged/",
        "use_tools": True,
        "use_rag": True,
        "max_loops": 3,
    },
    "no_tool": {
        "description": "消融: 去掉工具调用",
        "model_path": None,
        "use_tools": False,
        "use_rag": True,
        "max_loops": 3,
    },
    "no_rag": {
        "description": "消融: 去掉 RAG 检索",
        "model_path": None,
        "use_tools": True,
        "use_rag": False,
        "max_loops": 3,
    },
    "react_1": {
        "description": "消融: ReAct 循环 1 轮",
        "model_path": None,
        "use_tools": True,
        "use_rag": True,
        "max_loops": 1,
    },
    "react_5": {
        "description": "消融: ReAct 循环 5 轮",
        "model_path": None,
        "use_tools": True,
        "use_rag": True,
        "max_loops": 5,
    },
}

# 结果模板
RESULT_TEMPLATE = {
    "task_accuracy": 0.0,
    "task_partial_accuracy": 0.0,
    "avg_similarity": 0.0,
    "tool_f1": 0.0,
    "tool_param_accuracy": 0.0,
    "safety_pass_rate": 0.0,
    "judge_overall": 0.0,
    "judge_safety": 0.0,
    "judge_accuracy": 0.0,
    "avg_latency_ms": 0.0,
    "avg_tool_calls": 0.0,
    "escalation_rate": 0.0,
}


def run_single_ablation(
    config_name: str,
    config: Dict,
    eval_cases: List[Dict],
) -> Dict:
    """
    运行单个消融实验配置。

    Returns:
        评测结果字典
    """
    logger.info(f"运行消融实验: {config_name} - {config['description']}")

    # 生成预测
    predictions = []
    start_time = time.time()

    try:
        from graph.workflow import run_consultation
        from tools.setup import setup_tools

        if config.get("use_tools"):
            setup_tools()

        for i, case in enumerate(eval_cases):
            patient_input = _extract_patient_input(case)
            try:
                result = run_consultation(
                    patient_input,
                    use_tools=config.get("use_tools", True),
                    use_rag=config.get("use_rag", True),
                    max_loops=config.get("max_loops", 3),
                )
                predictions.append(result)
            except Exception as e:
                logger.warning(f"[{config_name}] Case {i} 失败: {e}")
                predictions.append({"final_response": "", "tool_calls": [], "confidence": 0.0})

            if (i + 1) % 10 == 0:
                logger.info(f"[{config_name}] 进度: {i+1}/{len(eval_cases)}")

    except ImportError as e:
        logger.error(f"[{config_name}] 导入失败: {e}")
        return {"config": config_name, "error": str(e), **RESULT_TEMPLATE}

    elapsed = time.time() - start_time

    # 运行评测
    result = {"config": config_name, "description": config["description"]}

    try:
        from evaluation.task_eval import evaluate_task_completion, evaluate_tool_usage
        task_res = evaluate_task_completion(predictions, eval_cases)
        tool_res = evaluate_tool_usage(predictions, eval_cases)
        result["task_accuracy"] = task_res.get("accuracy", 0)
        result["task_partial_accuracy"] = task_res.get("partial_accuracy", 0)
        result["avg_similarity"] = task_res.get("avg_similarity", 0)
        result["tool_f1"] = tool_res.get("avg_f1", 0)
        result["tool_param_accuracy"] = tool_res.get("avg_param_accuracy", 0)
        result["avg_tool_calls"] = tool_res.get("avg_tool_calls_per_case", 0)
    except Exception as e:
        logger.warning(f"[{config_name}] 任务评测失败: {e}")

    result["avg_latency_ms"] = elapsed * 1000 / max(len(eval_cases), 1)
    result["total_cases"] = len(eval_cases)

    return result


def _extract_patient_input(case: Dict) -> str:
    """从 case 提取患者输入"""
    for turn in case.get("dialogue", []):
        if turn.get("role") == "patient":
            return turn.get("content", "")
    return case.get("chief_complaint", "")


def generate_comparison_table(results: List[Dict]) -> str:
    """
    生成 Markdown 格式的对比表。
    """
    headers = [
        "模型", "任务准确率", "部分准确率", "语义相似度",
        "工具F1", "参数准确率", "工具调用数", "延迟(ms)",
    ]
    keys = [
        "config", "task_accuracy", "task_partial_accuracy", "avg_similarity",
        "tool_f1", "tool_param_accuracy", "avg_tool_calls", "avg_latency_ms",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in results:
        row = []
        for k in keys:
            v = r.get(k, 0)
            if k == "config":
                row.append(f"**{v}**")
            elif isinstance(v, float):
                if "accuracy" in k or "similarity" in k or "f1" in k:
                    row.append(f"{v:.1%}")
                else:
                    row.append(f"{v:.1f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MedAgent 消融实验")
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/ablation/")
    parser.add_argument("--configs", nargs="+", default=["base"],
                        help="要运行的配置名，如 base sft grpo_r1")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    logger.info(f"加载评测数据: {len(eval_cases)} 条")

    # 运行消融
    all_results = []
    for config_name in args.configs:
        if config_name not in ABLATION_CONFIGS:
            logger.warning(f"未知配置: {config_name}")
            continue
        config = ABLATION_CONFIGS[config_name]
        result = run_single_ablation(config_name, config, eval_cases)
        all_results.append(result)

        # 保存单个结果
        with open(output_dir / f"{config_name}_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # 生成对比表
    table = generate_comparison_table(all_results)
    table_path = output_dir / "comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# MedAgent 消融实验对比\n\n")
        f.write(f"评测数据: {args.eval_data} ({len(eval_cases)} 条)\n\n")
        f.write(table)
        f.write("\n")
    logger.info(f"对比表保存: {table_path}")

    # 保存全部结果
    with open(output_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + table)


if __name__ == "__main__":
    main()
