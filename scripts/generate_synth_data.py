"""
合成多轮问诊 Trajectory 数据。
用 LLM API 生成患者-医生对话，包含 ReAct 格式的思考链和工具调用。

用法:
  python scripts/generate_synth_data.py --num_cases 500 --output data/synth/trajectories/
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_client import chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 场景模板
# ─────────────────────────────────────────────

SCENARIO_TEMPLATES = [
    {
        "department": "心血管内科",
        "weight": 0.15,
        "scenarios": [
            {"profile": "{age}岁{gender}，高血压病史{years}年", "complaints": ["头晕", "血压控制不佳", "服药后仍头痛", "踝部水肿"]},
            {"profile": "{age}岁{gender}，冠心病", "complaints": ["胸闷", "活动后气短", "心悸", "服用他汀后肌痛"]},
        ],
    },
    {
        "department": "内分泌科",
        "weight": 0.15,
        "scenarios": [
            {"profile": "{age}岁{gender}，2型糖尿病{years}年", "complaints": ["空腹血糖偏高", "口渴多饮", "二甲双胍胃不舒服", "糖化血红蛋白升高"]},
            {"profile": "{age}岁{gender}，甲状腺功能异常", "complaints": ["体重变化", "怕冷/怕热", "TSH异常", "乏力"]},
        ],
    },
    {
        "department": "呼吸内科",
        "weight": 0.1,
        "scenarios": [
            {"profile": "{age}岁{gender}，反复咳嗽{weeks}周", "complaints": ["干咳", "咳痰", "夜间咳嗽加重", "低热"]},
            {"profile": "{age}岁{gender}，哮喘", "complaints": ["喘息", "呼吸困难", "季节性发作", "运动后加重"]},
        ],
    },
    {
        "department": "消化内科",
        "weight": 0.1,
        "scenarios": [
            {"profile": "{age}岁{gender}，反复腹痛", "complaints": ["上腹痛", "餐后不适", "反酸烧心", "幽门螺杆菌阳性"]},
            {"profile": "{age}岁{gender}，肝功能异常", "complaints": ["转氨酶升高", "乏力纳差", "右上腹不适", "饮酒史"]},
        ],
    },
    {
        "department": "神经内科",
        "weight": 0.1,
        "scenarios": [
            {"profile": "{age}岁{gender}，反复头痛", "complaints": ["偏头痛", "头晕", "视物模糊", "肢体麻木"]},
        ],
    },
    {
        "department": "骨科",
        "weight": 0.08,
        "scenarios": [
            {"profile": "{age}岁{gender}，腰痛{weeks}周", "complaints": ["腰部疼痛", "下肢放射痛", "活动受限", "久坐加重"]},
        ],
    },
    {
        "department": "儿科",
        "weight": 0.1,
        "scenarios": [
            {"profile": "{age}岁儿童", "complaints": ["发热", "咳嗽", "腹泻", "皮疹", "食欲不振"]},
        ],
    },
    {
        "department": "妇产科",
        "weight": 0.07,
        "scenarios": [
            {"profile": "{age}岁女性", "complaints": ["月经不调", "腹痛", "白带异常"]},
        ],
    },
    {
        "department": "急诊科",
        "weight": 0.05,
        "scenarios": [
            {"profile": "{age}岁{gender}，突发症状", "complaints": ["剧烈胸痛", "突发头晕伴呕吐", "外伤出血"]},
        ],
    },
    {
        "department": "皮肤科",
        "weight": 0.05,
        "scenarios": [
            {"profile": "{age}岁{gender}", "complaints": ["皮疹反复", "瘙痒", "荨麻疹"]},
        ],
    },
    {
        "department": "泌尿外科",
        "weight": 0.05,
        "scenarios": [
            {"profile": "{age}岁{gender}", "complaints": ["尿频尿急", "排尿困难", "血尿"]},
        ],
    },
]

GENERATION_PROMPT = """请生成一段医学问诊对话 trajectory，用于训练医学 AI Agent。

场景设定：
- 患者档案：{profile}
- 主诉：{complaint}
- 就诊科室：{department}

要求：
1. 对话为 3-5 轮（患者问 → Agent 回复），模拟真实问诊
2. Agent 必须使用 ReAct 格式思考：先 <think>思考</think>，再决定是否调用工具
3. 至少调用 1-2 个工具（从以下工具中选择）：
   - search_guidelines(query): 检索诊疗指南
   - search_drug(drug_name): 查询药品信息
   - check_drug_interaction(drug_a, drug_b): 检查药物交互
   - interpret_lab_result(test_name, value, unit): 解读检验值
4. 最终给出诊断方向和建议（不做确定性诊断，用"可能""建议"措辞）
5. 包含安全提示（建议就医/遵医嘱）

请严格按以下 JSON 格式输出：
{{
  "patient_profile": "患者档案",
  "chief_complaint": "主诉",
  "department": "科室",
  "dialogue": [
    {{
      "role": "patient",
      "content": "患者的话"
    }},
    {{
      "role": "agent",
      "thought": "Agent 的思考过程",
      "tool_calls": [
        {{"name": "工具名", "args": {{"参数": "值"}}, "result_summary": "结果摘要"}}
      ],
      "response": "Agent 回复患者的话"
    }}
  ],
  "final_diagnosis_direction": "诊断方向（如：高血压2级，降压方案需调整）",
  "tools_used": ["使用的工具列表"],
  "key_knowledge_points": ["涉及的医学知识点"]
}}"""


def _random_profile_params():
    """随机生成患者参数"""
    return {
        "age": random.choice(list(range(2, 8)) + list(range(25, 75))),
        "gender": random.choice(["男性", "女性"]),
        "years": random.randint(1, 15),
        "weeks": random.randint(1, 12),
    }


def generate_one_case(case_id: int, template: dict, scenario: dict) -> dict:
    """生成一个 case"""
    params = _random_profile_params()
    if template["department"] == "儿科":
        params["age"] = random.randint(1, 12)

    profile = scenario["profile"].format(**params)
    complaint = random.choice(scenario["complaints"])

    prompt = GENERATION_PROMPT.format(
        profile=profile,
        complaint=complaint,
        department=template["department"],
    )

    response = chat(prompt, temperature=0.7, max_tokens=2048)
    if not response:
        return None

    # 尝试解析 JSON
    try:
        # 提取 JSON 块
        json_str = response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        data = json.loads(json_str.strip())
        data["case_id"] = f"case_{case_id:04d}"
        data["generation_success"] = True
        return data
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"Case {case_id}: JSON 解析失败 - {e}")
        return {
            "case_id": f"case_{case_id:04d}",
            "raw_response": response[:500],
            "generation_success": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cases", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/synth/trajectories/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true",
                        help="先生成 10 条试跑 + 质控检查，通过后再批量生成")
    parser.add_argument("--pilot_size", type=int, default=10)
    parser.add_argument("--skip_quality", action="store_true",
                        help="跳过质控过滤")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Pilot 模式：先小批量验证 ───
    if args.pilot:
        logger.info(f"=== Pilot 模式: 生成 {args.pilot_size} 条试跑 ===")
        pilot_cases = _sample_cases(args.pilot_size)
        pilot_results = _generate_batch(pilot_cases, start_id=0)
        pilot_path = output_dir / "pilot_samples.json"
        with open(pilot_path, "w", encoding="utf-8") as f:
            json.dump(pilot_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Pilot 样本保存: {pilot_path}")

        # 自动质控
        try:
            from scripts.data_quality import run_quality_pipeline
            qc = run_quality_pipeline(pilot_results, strict=True)
            pass_rate = qc["stats"]["pass_rate"]
            logger.info(f"Pilot 质控通过率: {pass_rate:.1%}")
            if pass_rate < 0.5:
                logger.error("质控通过率 < 50%，建议检查生成 prompt 后重试")
                logger.error(f"问题分布: {qc['stats']['issue_breakdown']}")
                return
            logger.info(f"质控通过，请人工检查 {pilot_path} 后用不带 --pilot 的命令批量生成")
        except ImportError:
            logger.info(f"质控模块不可用，请人工检查 {pilot_path}")
        return

    # ─── 断点续跑：加载已有 checkpoint ───
    checkpoint_path = output_dir / "checkpoint.jsonl"
    plan_path = output_dir / "case_plan.json"

    # 生成或加载 case plan（保证 resume 时顺序一致）
    if plan_path.exists():
        with open(plan_path, "r", encoding="utf-8") as f:
            saved_plan = json.load(f)
        all_cases = []
        for item in saved_plan:
            tmpl = next((t for t in SCENARIO_TEMPLATES if t["department"] == item["department"]), None)
            if tmpl:
                scen = next((s for s in tmpl["scenarios"] if s["profile"] == item["profile"]), tmpl["scenarios"][0])
                all_cases.append((tmpl, scen))
        logger.info(f"从 case_plan.json 恢复计划: {len(all_cases)} 条")
    else:
        all_cases = _sample_cases(args.num_cases)
        plan_data = [{"department": t["department"], "profile": s["profile"]} for t, s in all_cases]
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2)
        logger.info(f"新建 case plan: {len(all_cases)} 条")

    # 加载已完成的 checkpoint
    done_ids = set()
    existing_results = []
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    existing_results.append(item)
                    done_ids.add(item.get("case_id", ""))
        logger.info(f"从 checkpoint 恢复: {len(existing_results)} 条已完成")

    # ─── 批量生成（断点续跑） ───
    remaining = [(i, t, s) for i, (t, s) in enumerate(all_cases)
                 if f"case_{i:04d}" not in done_ids]
    logger.info(f"待生成: {len(remaining)}/{len(all_cases)} 条")

    with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:
        for idx, (i, template, scenario) in enumerate(remaining):
            case_id = i
            logger.info(f"生成 [{len(done_ids)+idx+1}/{len(all_cases)}] {template['department']}")
            case = generate_one_case(case_id, template, scenario)
            if case:
                ckpt_f.write(json.dumps(case, ensure_ascii=False) + "\n")
                ckpt_f.flush()
                existing_results.append(case)
            if (idx + 1) % 50 == 0:
                logger.info(f"已生成 {len(done_ids)+idx+1}/{len(all_cases)} 条")
            time.sleep(0.5)

    results = existing_results

    # 质控过滤
    if not args.skip_quality:
        try:
            from scripts.data_quality import run_quality_pipeline
            qc = run_quality_pipeline(results)
            logger.info(f"质控: {qc['stats']['total_input']} → {qc['stats']['final_count']}")
            filtered = qc["passed"]
            failed_path = output_dir / "failed_cases.json"
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(qc["failed"], f, ensure_ascii=False, indent=2)
        except ImportError:
            logger.warning("质控模块不可用，跳过过滤")
            filtered = results
    else:
        filtered = results

    # 最终保存
    output_path = output_dir / "all_trajectories.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    success_count = sum(1 for r in filtered if r.get("generation_success"))
    logger.info(f"生成完成: {success_count}/{len(all_cases)} 成功，保存至 {output_path}")

    # 统计
    dept_stats = {}
    for r in filtered:
        dept = r.get("department", "unknown")
        dept_stats[dept] = dept_stats.get(dept, 0) + 1
    logger.info(f"科室分布: {json.dumps(dept_stats, ensure_ascii=False)}")

    # 清理 checkpoint
    logger.info("全部完成，checkpoint 文件保留供审计")


def _sample_cases(num: int) -> list:
    """按权重采样科室场景"""
    all_cases = []
    for template in SCENARIO_TEMPLATES:
        n = int(num * template["weight"])
        for _ in range(n):
            scenario = random.choice(template["scenarios"])
            all_cases.append((template, scenario))
    while len(all_cases) < num:
        template = random.choice(SCENARIO_TEMPLATES)
        scenario = random.choice(template["scenarios"])
        all_cases.append((template, scenario))
    random.shuffle(all_cases)
    return all_cases


def _generate_batch(cases: list, start_id: int = 0) -> list:
    """批量生成并返回结果"""
    results = []
    for i, (template, scenario) in enumerate(cases):
        case_id = start_id + i
        logger.info(f"生成 [{i+1}/{len(cases)}] {template['department']}")
        case = generate_one_case(case_id, template, scenario)
        if case:
            results.append(case)
        if (i + 1) % 50 == 0:
            logger.info(f"已生成 {i+1}/{len(cases)} 条")
        time.sleep(0.5)
    return results


def _save_batch(results, output_dir, count):
    """中间保存"""
    path = output_dir / f"trajectories_batch_{count}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"中间保存: {path} ({len(results)} 条)")


if __name__ == "__main__":
    main()
