"""
从合成 trajectory 中抽取评测 case + 内置 seed cases。
每条 case 含：患者输入、期望科室、期望工具、期望诊断方向。

用法:
  python scripts/generate_eval_cases.py --from_synth data/synth/trajectories/all_trajectories.json
"""

import json
import random
import argparse
from pathlib import Path

SEED_CASES = [
    {"patient_input": "我今年55岁，最近量血压总是150/95左右，头有点晕", "expected_department": "心血管内科", "expected_tools": ["search_guidelines"], "expected_diagnosis_direction": "高血压2级，需评估靶器官损害"},
    {"patient_input": "最近总是口渴多饮多尿，体重还瘦了，空腹血糖8.2", "expected_department": "内分泌科", "expected_tools": ["search_guidelines", "interpret_lab_result"], "expected_diagnosis_direction": "2型糖尿病可能，需OGTT和HbA1c确认"},
    {"patient_input": "咳嗽咳痰一周了，今天开始发烧38.5度", "expected_department": "呼吸内科", "expected_tools": ["search_guidelines"], "expected_diagnosis_direction": "社区获得性肺炎可能，建议胸片和血常规"},
    {"patient_input": "饭后上腹痛，有时候反酸烧心，已经两个月了", "expected_department": "消化内科", "expected_tools": ["search_guidelines"], "expected_diagnosis_direction": "消化性溃疡或胃食管反流，建议胃镜"},
    {"patient_input": "我妈70岁，今天早上突然说话不清楚，左边手脚不灵活", "expected_department": "神经内科", "expected_tools": ["search_guidelines"], "expected_diagnosis_direction": "急性脑卒中可能，需立即就医"},
    {"patient_input": "大脚趾突然红肿疼痛，一碰就疼，昨晚喝了啤酒", "expected_department": "风湿免疫科", "expected_tools": ["search_guidelines", "interpret_lab_result", "search_drug"], "expected_diagnosis_direction": "急性痛风发作可能"},
    {"patient_input": "最近两个月情绪很低落，什么都不想做，睡眠也不好", "expected_department": "精神科", "expected_tools": ["search_guidelines"], "expected_diagnosis_direction": "抑郁症可能，建议专业评估"},
    {"patient_input": "我孩子3岁，发烧39度两天了，精神还行", "expected_department": "儿科", "expected_tools": ["search_guidelines", "search_drug"], "expected_diagnosis_direction": "儿童发热，对症退热+查血常规"},
    {"patient_input": "怀孕28周，血压145/95，脚肿得厉害", "expected_department": "妇产科", "expected_tools": ["search_guidelines", "interpret_lab_result"], "expected_diagnosis_direction": "妊娠期高血压/子痫前期可能"},
    {"patient_input": "吃了阿司匹林和布洛芬之后拉了黑便", "expected_department": "消化内科", "expected_tools": ["search_guidelines", "search_drug", "interpret_lab_result"], "expected_diagnosis_direction": "药物性消化道出血"},
]


def extract_from_synth(synth_path, num=50):
    """从合成 trajectory 中抽取评测 case"""
    with open(synth_path, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    cases = []
    sampled = random.sample(trajectories, min(num, len(trajectories)))
    for t in sampled:
        dialogue = t.get("dialogue", [])
        first_patient = ""
        for d in dialogue:
            if d.get("role") == "patient":
                first_patient = d.get("content", "")
                break
        if first_patient:
            cases.append({
                "patient_input": first_patient,
                "expected_department": t.get("department", ""),
                "expected_tools": t.get("tools_used", []),
                "expected_diagnosis_direction": t.get("final_diagnosis_direction", ""),
            })
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/eval/eval_cases.json")
    parser.add_argument("--from_synth", type=str, default=None)
    parser.add_argument("--num_synth", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = list(SEED_CASES)  # 10 seed cases

    if args.from_synth:
        synth_cases = extract_from_synth(args.from_synth, args.num_synth)
        cases.extend(synth_cases)
        print(f"Seed: {len(SEED_CASES)} + Synth: {len(synth_cases)} = {len(cases)} 条")
    else:
        print(f"仅 Seed: {len(cases)} 条")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"保存至: {output_path}")

    depts = {}
    for c in cases:
        d = c.get("expected_department", "?")
        depts[d] = depts.get(d, 0) + 1
    print("\n科室分布:")
    for d, cnt in sorted(depts.items(), key=lambda x: -x[1]):
        print(f"  {d}: {cnt}")


if __name__ == "__main__":
    main()
