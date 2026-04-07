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
    parser.add_argument("--max_cases", type=int, default=0, help="最多评测几条（0=全部）")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"timestamp": time.time(), "eval_data": args.eval_data}

    # 加载评测数据
    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    logger.info(f"评测数据加载: {len(eval_cases)} 条")

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
                    result = run_consultation(patient_input)
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
    if predictions:
        from evaluation.task_eval import evaluate_task_completion, evaluate_tool_usage
        task_result = evaluate_task_completion(predictions, eval_cases)
        tool_result = evaluate_tool_usage(predictions, eval_cases)
        report["task_completion"] = task_result
        report["tool_usage"] = tool_result
        logger.info(f"任务完成率: {task_result.get('accuracy', 0):.2%}")
        logger.info(f"工具 F1: {tool_result.get('avg_f1', 0):.3f}")

    # ─── 轨迹效率 ───
    if predictions:
        from evaluation.trajectory_eval import evaluate_trajectory_efficiency
        traj_result = evaluate_trajectory_efficiency(predictions)
        report["trajectory_efficiency"] = traj_result
        logger.info(f"效率分数: {traj_result.get('efficiency_score', 0):.3f}")

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

        safety_result = evaluate_safety(safety_responses, test_cases=safety_cases)
        report["safety"] = safety_result
        logger.info(f"安全通过率: {safety_result.get('pass_rate', 0):.2%}")

    # ─── LLM-as-Judge（断点续跑）───
    if args.run_judge and predictions:
        from evaluation.llm_judge import judge_single

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

        logger.info(f"运行 LLM-as-Judge... ({judge_done}/{len(predictions)})")
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
                    score = judge_single(
                        patient_input=patient_input,
                        agent_response=pred.get("final_response", ""),
                        tool_calls=pred.get("tool_calls"),
                    ) or {}
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
            report["llm_judge"] = summary
            logger.info(f"Judge 平均分: {summary.get('avg_overall', 0):.2f}")

    # ─── 保存报告 ───
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"评测报告保存: {report_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("MedAgent 评测报告摘要")
    print("=" * 60)
    if "task_completion" in report:
        tc = report["task_completion"]
        print(f"  任务完成率:    {tc.get('accuracy', 0):.2%} ({tc.get('correct', 0)}/{tc.get('total', 0)})")
    if "tool_usage" in report:
        tu = report["tool_usage"]
        print(f"  工具调用 F1:   {tu.get('avg_f1', 0):.3f}")
    if "trajectory_efficiency" in report:
        te = report["trajectory_efficiency"]
        print(f"  效率分数:      {te.get('efficiency_score', 0):.3f}")
        print(f"  升级率:        {te.get('escalation_rate', 0):.2%}")
    if "safety" in report:
        sa = report["safety"]
        print(f"  安全通过率:    {sa.get('pass_rate', 0):.2%}")
    if "llm_judge" in report:
        lj = report["llm_judge"]
        print(f"  LLM Judge 均分: {lj.get('avg_overall', 0):.2f}/5.0")
    print("=" * 60)


if __name__ == "__main__":
    main()
