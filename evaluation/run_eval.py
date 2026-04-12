"""
一键评测入口：运行所有评测模块并生成报告。

用法:
  python evaluation/run_eval.py --eval_data data/eval/cases_500.json --output results/
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _extract_patient_input(case: dict) -> str:
    for turn in case.get("dialogue", []):
        if turn.get("role") == "patient":
            return turn.get("content", "")
    return case.get("chief_complaint", "") or case.get("patient_input", "")


def _build_failure_cases(predictions, references, task_result):
    details = task_result.get("details", []) if isinstance(task_result, dict) else []
    failures = []
    tag_counts = {}
    for idx, (pred, ref, detail) in enumerate(zip(predictions, references, details)):
        if detail.get("is_correct"):
            continue

        expected_tools = ref.get("preferred_tool_sequence") or ref.get("expected_tools") or ref.get("tools_used") or []
        predicted_tools = [tc.get("tool_name", "") for tc in pred.get("tool_calls", []) if tc.get("tool_name")]
        expected_first_tool = ref.get("expected_first_tool") or (expected_tools[0] if expected_tools else "")
        predicted_first_tool = predicted_tools[0] if predicted_tools else ""

        tags = []
        if not detail.get("department_match"):
            tags.append("department_mismatch")
        if detail.get("diagnosis_similarity", 0.0) < 0.7:
            tags.append("diagnosis_mismatch")
        test_recall = detail.get("test_recall")
        if test_recall is not None and test_recall < 1.0:
            tags.append("missing_tests")
        if expected_first_tool and predicted_first_tool != expected_first_tool:
            tags.append("wrong_first_tool")
        if any(tool not in set(expected_tools) for tool in predicted_tools):
            tags.append("offplan_tool")
        if len(predicted_tools) > len(set(predicted_tools)):
            tags.append("duplicate_tool")

        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        failures.append({
            "case_id": idx,
            "patient_input": _extract_patient_input(ref),
            "expected_department": ref.get("expected_department") or ref.get("department", ""),
            "predicted_department": detail.get("pred_department", ""),
            "expected_diagnosis_direction": ref.get("expected_diagnosis_direction") or ref.get("final_diagnosis_direction", ""),
            "predicted_diagnosis_direction": pred.get("structured_output", {}).get("diagnosis_direction") or pred.get("final_response", ""),
            "expected_tools": expected_tools,
            "predicted_tools": predicted_tools,
            "expected_first_tool": expected_first_tool,
            "predicted_first_tool": predicted_first_tool,
            "recommended_tests": ref.get("recommended_tests", []),
            "combined_score": detail.get("combined_score", 0.0),
            "diagnosis_similarity": detail.get("diagnosis_similarity", 0.0),
            "department_match": detail.get("department_match", 0.0),
            "test_recall": test_recall,
            "failure_tags": tags,
            "tool_plan": ref.get("preferred_tool_sequence") or ref.get("expected_tools") or [],
            "need_pharmacist": ref.get("need_pharmacist", False),
            "prediction": pred,
            "reference": ref,
        })

    return {
        "total_failures": len(failures),
        "failure_rate": len(failures) / max(len(predictions), 1),
        "tag_counts": tag_counts,
        "cases": failures,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MedAgent 一键评测")
    parser.add_argument("--eval_data", type=str, required=True, help="评测数据集路径")
    parser.add_argument("--predictions", type=str, default="", help="预测结果路径（如已有）")
    parser.add_argument("--output", type=str, default="results/", help="报告输出目录")
    parser.add_argument("--run_agent", action="store_true", help="运行 Agent 生成预测")
    parser.add_argument("--run_safety", action="store_true", help="运行安全红队测试")
    parser.add_argument("--red_team_data", type=str, default="", help="外部红队测试集路径（默认使用内置 case）")
    parser.add_argument("--run_judge", action="store_true", help="运行 LLM-as-Judge")
    parser.add_argument("--judge_model", type=str, default=None, help="Judge 模型")
    parser.add_argument("--secondary_judge_model", type=str, default=None, help="副 Judge 模型")
    parser.add_argument("--disable_tools", action="store_true", help="消融：禁用工具调用")
    parser.add_argument("--disable_rag", action="store_true", help="消融：禁用 RAG")
    parser.add_argument("--use_memory", action="store_true", help="消融：启用长期记忆")
    parser.add_argument("--max_tool_calls", type=int, default=0, help="每个 case 的总工具调用上限（0=默认）")
    parser.add_argument("--max_calls_per_tool", type=int, default=0, help="每个工具单独调用上限（0=默认）")
    parser.add_argument("--max_cases", type=int, default=0, help="最多评测几条（0=全部）")
    parser.add_argument("--safety_sample", type=int, default=0,
                        help="每类红队攻击采样几条（0=全部，建议2=每类取最低+最高强度）")
    parser.add_argument("--eval_source", type=str, default="",
                        help="评测数据来源标注，如 'synth' / 'CMB-Exam' / 'CMB-Clin' / 'hard_case'")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检测数据来源
    eval_source = args.eval_source
    if not eval_source:
        # 自动推断
        eval_data_name = Path(args.eval_data).stem
        if "cmb" in eval_data_name.lower():
            eval_source = "CMB"
        elif "hard" in eval_data_name.lower():
            eval_source = "hard_case"
        else:
            eval_source = "synth"

    report = {
        "timestamp": time.time(),
        "eval_data": args.eval_data,
        "eval_source": eval_source,
        "runtime": {
            "use_tools": not args.disable_tools,
            "use_rag": not args.disable_rag,
            "use_memory": args.use_memory,
            "max_tool_calls": args.max_tool_calls,
            "max_calls_per_tool": args.max_calls_per_tool,
            "judge_model": args.judge_model,
            "secondary_judge_model": args.secondary_judge_model,
        },
    }

    # 加载评测数据
    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    logger.info(f"评测数据加载: {len(eval_cases)} 条 (来源: {eval_source})")

    # 统计各数据来源分布
    source_dist = {}
    for c in eval_cases:
        src = c.get("source", eval_source)
        source_dist[src] = source_dist.get(src, 0) + 1
    if source_dist:
        report["data_source_distribution"] = source_dist
        logger.info(f"数据来源分布: {source_dist}")

    # 限制评测数量
    if args.max_cases > 0:
        eval_cases = eval_cases[:args.max_cases]
        logger.info(f"限制评测: 取前 {args.max_cases} 条")

    # ─── 运行 Agent 生成预测（断点续跑） ───
    predictions = []
    if args.run_agent:
        from graph.workflow import run_consultation, _long_term_memory
        from tools.setup import setup_tools
        setup_tools()

        checkpoint_path = output_dir / "pred_checkpoint.jsonl"
        done_count = 0
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))
            done_count = len(predictions)
            logger.info(f"从 checkpoint 恢复: {done_count} 条已完成")

        logger.info(f"开始运行 Agent 生成预测... ({done_count}/{len(eval_cases)})")
        with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
            for i, case in enumerate(eval_cases):
                if i < done_count:
                    continue

                # 评测隔离：每个 case 清空长期记忆，避免跨 case 污染
                _long_term_memory.reset()

                patient_input = ""
                dialogue = case.get("dialogue", [])
                for turn in dialogue:
                    if turn.get("role") == "patient":
                        patient_input = turn.get("content", "")
                        break
                if not patient_input:
                    patient_input = case.get("chief_complaint", "") or case.get("patient_input", "")

                try:
                    result = run_consultation(
                        patient_input,
                        use_tools=not args.disable_tools,
                        use_rag=not args.disable_rag,
                        use_memory=args.use_memory,
                        max_tool_calls=args.max_tool_calls or None,
                        max_calls_per_tool=args.max_calls_per_tool or None,
                    )
                except Exception as e:
                    logger.error(f"Case {i}: Agent 运行失败 - {e}")
                    result = {"final_response": "", "tool_calls": []}

                predictions.append(result)
                ckpt_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                ckpt_f.flush()
                logger.info(f"Agent 进度: {i+1}/{len(eval_cases)}")

        # 保存预测
        pred_path = output_dir / "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"预测保存: {pred_path}")

    elif args.predictions:
        with open(args.predictions, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        logger.info(f"加载已有预测: {len(predictions)} 条")

    # ─── 任务完成率 + 工具 F1 ───
    if predictions and eval_cases:
        from evaluation.task_eval import evaluate_task_completion, evaluate_tool_usage
        task_result = evaluate_task_completion(predictions, eval_cases)
        tool_result = evaluate_tool_usage(predictions, eval_cases)
        report["task_completion"] = task_result
        report["tool_usage"] = tool_result
        failure_result = _build_failure_cases(predictions, eval_cases, task_result)
        report["failure_analysis"] = {
            "total_failures": failure_result["total_failures"],
            "failure_rate": failure_result["failure_rate"],
            "tag_counts": failure_result["tag_counts"],
        }
        with open(output_dir / "failure_cases.json", "w", encoding="utf-8") as f:
            json.dump(failure_result["cases"], f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"任务完成率: {task_result.get('accuracy', 0):.2%}")
        logger.info(f"工具 F1: {tool_result.get('avg_f1', 0):.3f}")

    # ─── 轨迹效率 ───
    if predictions:
        from evaluation.trajectory_eval import evaluate_trajectory_efficiency
        traj_result = evaluate_trajectory_efficiency(predictions, eval_cases)
        report["trajectory_efficiency"] = traj_result
        logger.info(f"效率分数: {traj_result.get('efficiency_score', 0):.3f}")

    # ─── 推理链评测 ───
    if predictions:
        from evaluation.reasoning_eval import evaluate_reasoning
        reasoning_result = evaluate_reasoning(predictions, eval_cases if eval_cases else None)
        report["reasoning"] = reasoning_result
        logger.info(f"推理链分数: {reasoning_result.get('overall_reasoning_score', 0):.3f}")

    # ─── 置信度校准 ───
    if predictions:
        from evaluation.calibration import CalibrationAnalyzer
        try:
            analyzer = CalibrationAnalyzer(predictions, eval_cases)
            cal_report = analyzer.run()
            report["calibration"] = cal_report["summary"]
            analyzer.save(str(output_dir / "calibration_report.json"))
            logger.info(f"ECE: {cal_report['summary']['ece']:.4f}, "
                        f"最优阈值: {cal_report['summary']['optimal_threshold']}")
        except Exception as e:
            logger.warning(f"校准分析失败: {e}")

    # ─── 安全红队测试（断点续跑）───
    if args.run_safety:
        from evaluation.safety_eval import evaluate_safety, RED_TEAM_CASES
        from graph.workflow import run_consultation
        from tools.setup import setup_tools
        setup_tools()

        # 加载红队测试集
        if args.red_team_data and Path(args.red_team_data).exists():
            with open(args.red_team_data, "r", encoding="utf-8") as f:
                safety_cases = json.load(f)
            logger.info(f"加载外部红队测试集: {args.red_team_data} ({len(safety_cases)} 条)")
        else:
            safety_cases = RED_TEAM_CASES
            logger.info(f"使用内置红队测试集: {len(safety_cases)} 条)")

        # 按类别采样（每类取最低+最高强度）
        if args.safety_sample and args.safety_sample > 0:
            from collections import defaultdict
            by_cat = defaultdict(list)
            for c in safety_cases:
                by_cat[c.get("category", "unknown")].append(c)
            sampled = []
            for cat, cases in by_cat.items():
                sorted_cases = sorted(cases, key=lambda x: x.get("severity", 0))
                if args.safety_sample >= len(sorted_cases):
                    sampled.extend(sorted_cases)
                else:
                    # 取最低和最高强度
                    sampled.append(sorted_cases[0])
                    sampled.append(sorted_cases[-1])
                    # 如果要更多，从中间取
                    remaining = args.safety_sample - 2
                    mid = sorted_cases[1:-1]
                    sampled.extend(mid[:remaining])
            safety_cases = sampled
            logger.info(f"红队采样: 每类 {args.safety_sample} 条 → 共 {len(safety_cases)} 条")

        # 断点续跑
        safety_ckpt_path = output_dir / "safety_checkpoint.jsonl"
        safety_responses = []
        safety_done = 0
        if safety_ckpt_path.exists():
            with open(safety_ckpt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        safety_responses.append(json.loads(line))
            safety_done = len(safety_responses)
            logger.info(f"安全从 checkpoint 恢复: {safety_done}/{len(safety_cases)}")

        logger.info(f"运行安全红队测试... ({safety_done}/{len(safety_cases)})")
        with open(safety_ckpt_path, "a", encoding="utf-8") as ckpt_f:
            for i, case in enumerate(safety_cases):
                if i < safety_done:
                    continue
                try:
                    result = run_consultation(case["input"])
                    safety_responses.append(result)
                except Exception as e:
                    result = {"final_response": str(e)}
                    safety_responses.append(result)
                ckpt_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                ckpt_f.flush()
                logger.info(f"安全进度: {i+1}/{len(safety_cases)}")

        safety_result = evaluate_safety(
            safety_responses,
            test_cases=safety_cases,
            use_llm_judge=not args.disable_tools,
            include_impossible=True,
        )
        report["safety"] = safety_result
        logger.info(f"安全通过率: {safety_result.get('pass_rate', 0):.2%}")
        if safety_result.get("dose_response_curve"):
            logger.info(f"剂量-反应曲线: {safety_result['dose_response_curve']}")

    # ─── LLM-as-Judge（断点续跑，支持双模型交叉评分） ───
    if args.run_judge and predictions:
        from evaluation.llm_judge import judge_single, _merge_scores

        judge_ckpt_path = output_dir / "judge_checkpoint.jsonl"
        judge_scores = []
        judge_done = 0
        if judge_ckpt_path.exists():
            with open(judge_ckpt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        judge_scores.append(json.loads(line))
            judge_done = len(judge_scores)
            logger.info(f"Judge 从 checkpoint 恢复: {judge_done}/{len(predictions)}")

        judge_model = args.judge_model
        secondary_model = args.secondary_judge_model
        logger.info(
            f"运行 LLM-as-Judge (primary={judge_model or 'default'}, secondary={secondary_model or 'none'})... "
            f"({judge_done}/{len(predictions)})"
        )
        with open(judge_ckpt_path, "a", encoding="utf-8") as ckpt_f:
            for i, (pred, ref) in enumerate(zip(predictions, eval_cases)):
                if i < judge_done:
                    continue
                patient_input = ""
                for turn in ref.get("dialogue", []):
                    if turn.get("role") == "patient":
                        patient_input = turn.get("content", "")
                        break
                patient_input = patient_input or ref.get("chief_complaint", "") or ref.get("patient_input", "")
                try:
                    primary = judge_single(
                        patient_input=patient_input,
                        agent_response=pred.get("final_response", ""),
                        tool_calls=pred.get("tool_calls"),
                        judge_model=judge_model,
                    ) or {}
                    if secondary_model:
                        secondary = judge_single(
                            patient_input=patient_input,
                            agent_response=pred.get("final_response", ""),
                            tool_calls=pred.get("tool_calls"),
                            judge_model=secondary_model,
                        ) or {}
                        score = _merge_scores(primary, secondary) or {}
                    else:
                        score = primary
                except Exception as e:
                    logger.warning(f"Judge case {i} 失败: {e}")
                    score = {}
                judge_scores.append(score)
                ckpt_f.write(json.dumps(score, ensure_ascii=False, default=str) + "\n")
                ckpt_f.flush()
                logger.info(f"Judge 进度: {i+1}/{len(predictions)}")

        # 汇总 Judge 结果
        if judge_scores:
            keys = ["accuracy", "safety", "completeness", "clarity", "tool_usage", "overall"]
            summary = {}
            for k in keys:
                vals = [s.get(k, 0) for s in judge_scores if s.get(k) is not None]
                summary[f"avg_{k}"] = sum(vals) / len(vals) if vals else 0
            summary["n"] = len(judge_scores)

            # 主指标：基于 Judge accuracy 维度的诊断准确率（对齐 AgentClinic 范式）
            acc_vals = [s.get("accuracy", 0) for s in judge_scores if s.get("accuracy") is not None]
            if acc_vals:
                correct = sum(1 for v in acc_vals if v >= 4)
                partial = sum(1 for v in acc_vals if 3 <= v < 4)
                total_j = len(acc_vals)
                summary["diagnostic_accuracy"] = correct / total_j
                summary["diagnostic_partial"] = (correct + partial) / total_j
                summary["diagnostic_correct"] = correct
                summary["diagnostic_partial_correct"] = partial
                summary["diagnostic_total"] = total_j
                logger.info(f"Judge 诊断准确率: {correct}/{total_j} = {correct/total_j:.1%}")

            report["llm_judge"] = summary
            logger.info(f"Judge 平均分: {summary.get('avg_overall', 0):.2f}")

    # ─── 保存报告 ───
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"评测报告保存: {report_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"MedAgent 评测报告摘要 [数据来源: {report.get('eval_source', 'unknown')}]")
    print("=" * 60)
    if "data_source_distribution" in report:
        print(f"  数据来源分布: {report['data_source_distribution']}")
    if "llm_judge" in report:
        lj = report["llm_judge"]
        print(f"  ★ 诊断准确率 (Judge≥4):  {lj.get('diagnostic_accuracy', 0):.1%} "
              f"({lj.get('diagnostic_correct', 0)}/{lj.get('diagnostic_total', 0)})")
        print(f"    含部分正确 (Judge≥3): {lj.get('diagnostic_partial', 0):.1%}")
        print(f"    Judge 准确性均分:     {lj.get('avg_accuracy', 0):.2f}/5.0")
        print(f"    Judge 安全性均分:     {lj.get('avg_safety', 0):.2f}/5.0")
        print(f"    Judge 总分:           {lj.get('avg_overall', 0):.2f}/5.0")
    if "task_completion" in report:
        tc = report["task_completion"]
        print(f"  · 结构化组合分数:       {tc.get('avg_combined_score', 0):.3f}")
        print(f"  · 科室准确率:           {tc.get('department_accuracy', 0):.1%}")
        print(f"  · 诊断方向准确率:       {tc.get('diagnosis_accuracy', 0):.1%}")
    if "tool_usage" in report:
        tu = report["tool_usage"]
        print(f"  · 工具调用 F1:          {tu.get('avg_f1', 0):.3f}")
        print(f"  · First Tool 准确率:    {tu.get('first_tool_accuracy', 0):.1%}")
        print(f"  · 重复工具率:           {tu.get('duplicate_tool_rate', 0):.3f}")
    if "trajectory_efficiency" in report:
        te = report["trajectory_efficiency"]
        print(f"  · 效率分数:             {te.get('efficiency_score', 0):.3f}")
    if "reasoning" in report:
        re_ = report["reasoning"]
        print(f"  · 推理链综合分数:       {re_.get('overall_reasoning_score', 0):.3f}")
        print(f"    完整性:               {re_.get('avg_completeness', 0):.3f}")
        print(f"    证据锚定率:           {re_.get('avg_evidence_grounding', 0):.3f}")
        print(f"    自洽性:               {re_.get('avg_consistency', 0):.3f}")
        print(f"    工具归因:             {re_.get('avg_tool_attribution', 0):.3f}")
    if "safety" in report:
        sa = report["safety"]
        print(f"  · 安全通过率:           {sa.get('pass_rate', 0):.2%}")
        curve = sa.get("dose_response_curve", [])
        if curve:
            print(f"    强度梯度通过率:       ", end="")
            for pt in curve:
                if pt["pass_rate"] is not None:
                    print(f"L{pt['severity']}={pt['pass_rate']:.0%} ", end="")
            print()
    print("=" * 60)


if __name__ == "__main__":
    main()
