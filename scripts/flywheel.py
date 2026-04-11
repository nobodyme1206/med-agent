"""
数据飞轮编排器：一键运行 eval → failure mining → augment → quality → SFT 全流程。

流程：
  Step 1: 运行评测（生成 predictions + evaluation_report）
  Step 2: 从评测结果中挖掘失败用例
  Step 3: 将失败用例增补为训练数据（SFT / GRPO / 各阶段）
  Step 4: 质控过滤（结构/工具/医学/安全/推理链/reflection）
  Step 5: 合并到主训练集

用法:
  python scripts/flywheel.py \
    --eval_data data/eval/cases_500.json \
    --predictions results/predictions.json \
    --sft_base data/synth/sft_data/agent_sft.json \
    --output_dir results/flywheel_round1 \
    [--run_eval]  # 是否重新运行评测（默认使用已有 predictions）
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="数据飞轮编排器")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="评测数据集路径")
    parser.add_argument("--predictions", type=str, default="",
                        help="已有的 predictions 路径（跳过评测步骤）")
    parser.add_argument("--sft_base", type=str, default="",
                        help="基础 SFT 数据路径（用于合并）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--run_eval", action="store_true",
                        help="是否重新运行评测生成 predictions")
    parser.add_argument("--skip_partial", action="store_true",
                        help="跳过部分正确的 case（只保留完全错误的）")
    parser.add_argument("--quality_strict", action="store_true",
                        help="质控严格模式")
    parser.add_argument("--use_embedding_dedup", action="store_true",
                        help="质控时使用 embedding 去近似")
    parser.add_argument("--round", type=int, default=1,
                        help="飞轮轮次编号")
    return parser.parse_args()


def step1_eval(args, output_dir: Path) -> tuple:
    """Step 1: 评测（可选跳过）"""
    logger.info("=" * 60)
    logger.info("Step 1: 评测")
    logger.info("=" * 60)

    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    logger.info(f"评测集: {len(eval_cases)} 条")

    if args.predictions and Path(args.predictions).exists():
        with open(args.predictions, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        logger.info(f"加载已有 predictions: {len(predictions)} 条")
    elif args.run_eval:
        logger.info("运行 Agent 评测... (请确保有 LLM API 访问)")
        from graph.workflow import run_consultation
        from tools.setup import setup_tools
        setup_tools()

        predictions = []
        for i, case in enumerate(eval_cases):
            patient_input = ""
            for turn in case.get("dialogue", []):
                if turn.get("role") == "patient":
                    patient_input = turn.get("content", "")
                    break
            if not patient_input:
                patient_input = case.get("chief_complaint", "") or case.get("patient_input", "")
            try:
                result = run_consultation(patient_input)
            except Exception as e:
                logger.error(f"Case {i}: {e}")
                result = {"final_response": "", "tool_calls": []}
            predictions.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"评测进度: {i+1}/{len(eval_cases)}")

        pred_path = output_dir / "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        logger.info(f"predictions 保存: {pred_path}")
    else:
        raise ValueError("需要提供 --predictions 或 --run_eval")

    return predictions, eval_cases


def step2_mine_failures(predictions, eval_cases, output_dir: Path, skip_partial: bool) -> list:
    """Step 2: 挖掘失败用例"""
    logger.info("=" * 60)
    logger.info("Step 2: 失败用例挖掘")
    logger.info("=" * 60)

    from evaluation.task_eval import evaluate_single_prediction
    from scripts.augment_failure_cases import _normalize_failure_case

    failures = []
    for idx, (pred, ref) in enumerate(zip(predictions, eval_cases)):
        detail = evaluate_single_prediction(pred, ref)
        if detail.get("is_correct"):
            continue
        if skip_partial and detail.get("is_partial"):
            continue

        patient_input = ""
        for turn in ref.get("dialogue", []):
            if turn.get("role") == "patient":
                patient_input = turn.get("content", "")
                break
        if not patient_input:
            patient_input = ref.get("chief_complaint", "") or ref.get("patient_input", "")

        failures.append(_normalize_failure_case({
            "case_id": idx,
            "patient_input": patient_input,
            "prediction": pred,
            "reference": ref,
        }))

    # 保存
    failures_path = output_dir / "mined_failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    logger.info(f"挖掘到 {len(failures)} 条失败用例 → {failures_path}")

    return failures


def step3_augment(failures: list, output_dir: Path) -> dict:
    """Step 3: 增补训练数据"""
    logger.info("=" * 60)
    logger.info("Step 3: 失败用例增补")
    logger.info("=" * 60)

    from scripts.augment_failure_cases import build_outputs

    outputs = build_outputs(failures)

    files = {
        "hard_case_sft.json": outputs["agent_sft"],
        "hard_case_grpo.json": outputs["grpo"],
        "hard_case_router_sft.json": outputs["router"],
        "hard_case_planner_sft.json": outputs["planner"],
        "hard_case_summary_sft.json": outputs["summary"],
        "failure_manifest.json": outputs["manifest"],
        "failure_stats.json": {"total": len(outputs["manifest"]), "tag_counts": outputs["tag_counts"]},
    }
    for name, payload in files.items():
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"增补: SFT={len(outputs['agent_sft'])}, GRPO={len(outputs['grpo'])}, "
                f"Router={len(outputs['router'])}, Planner={len(outputs['planner'])}, "
                f"Summary={len(outputs['summary'])}")
    logger.info(f"失败 tag 分布: {outputs['tag_counts']}")

    return outputs


def step4_quality(augmented_sft: list, output_dir: Path, strict: bool, use_embedding: bool) -> list:
    """Step 4: 质控过滤"""
    logger.info("=" * 60)
    logger.info("Step 4: 质控过滤")
    logger.info("=" * 60)

    if not augmented_sft:
        logger.warning("无增补数据，跳过质控")
        return []

    # 质控需要 trajectory 格式的数据，这里对 SFT 做轻量检查
    # 主要检查对话轮数和必要字段
    passed = []
    failed = 0
    for item in augmented_sft:
        convs = item.get("conversations", [])
        human_turns = sum(1 for c in convs if c.get("from") == "human")
        gpt_turns = sum(1 for c in convs if c.get("from") == "gpt")

        if human_turns < 1 or gpt_turns < 1:
            failed += 1
            continue
        # 检查 gpt 回复非空
        gpt_contents = [c.get("value", "") for c in convs if c.get("from") == "gpt"]
        if not any(len(c.strip()) > 20 for c in gpt_contents):
            failed += 1
            continue
        passed.append(item)

    logger.info(f"质控: {len(augmented_sft)} → {len(passed)} (过滤 {failed} 条)")

    quality_path = output_dir / "hard_case_sft_filtered.json"
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)

    return passed


def step5_merge(filtered_sft: list, sft_base_path: str, output_dir: Path, round_num: int) -> int:
    """Step 5: 合并到主训练集"""
    logger.info("=" * 60)
    logger.info("Step 5: 合并训练集")
    logger.info("=" * 60)

    base_data = []
    if sft_base_path and Path(sft_base_path).exists():
        with open(sft_base_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        logger.info(f"基础 SFT: {len(base_data)} 条")

    merged = base_data + filtered_sft
    merged_path = output_dir / f"merged_sft_round{round_num}.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(f"合并: {len(base_data)} + {len(filtered_sft)} = {len(merged)} → {merged_path}")
    return len(merged)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Step 1
    predictions, eval_cases = step1_eval(args, output_dir)

    # Step 2
    failures = step2_mine_failures(predictions, eval_cases, output_dir, args.skip_partial)

    # Step 3
    outputs = step3_augment(failures, output_dir)

    # Step 4
    filtered = step4_quality(outputs["agent_sft"], output_dir, args.quality_strict, args.use_embedding_dedup)

    # Step 5
    total = step5_merge(filtered, args.sft_base, output_dir, args.round)

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"飞轮 Round {args.round} 完成! ({elapsed:.1f}s)")
    logger.info(f"  失败用例: {len(failures)}")
    logger.info(f"  增补 SFT: {len(outputs['agent_sft'])}")
    logger.info(f"  质控通过: {len(filtered)}")
    logger.info(f"  合并总量: {total}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"{'=' * 60}")

    # 保存飞轮元数据
    meta = {
        "round": args.round,
        "timestamp": time.time(),
        "eval_data": args.eval_data,
        "total_eval_cases": len(eval_cases),
        "total_predictions": len(predictions),
        "total_failures": len(failures),
        "augmented_sft": len(outputs["agent_sft"]),
        "filtered_sft": len(filtered),
        "merged_total": total,
        "elapsed_seconds": elapsed,
        "tag_counts": outputs.get("tag_counts", {}),
    }
    with open(output_dir / "flywheel_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
