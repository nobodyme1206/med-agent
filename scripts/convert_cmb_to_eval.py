"""
将 CMB (Chinese Medical Benchmark) 数据集转换为 MedAgent eval_cases.json 格式。

支持两种数据源：
  1. CMB-Exam (选择题) → 提取临床场景题，转为患者描述 + 预期诊断
  2. CMB-Clin (案例分析) → 直接用病史描述 + 诊断/鉴别作为 ground truth

用法:
  # 方式 1: 从 HuggingFace 下载 (需要 datasets 库)
  python scripts/convert_cmb_to_eval.py --source huggingface --output data/eval/cmb_eval.json

  # 方式 2: 从本地 JSON 文件
  python scripts/convert_cmb_to_eval.py --source local \
    --exam_path /path/to/cmb_exam_val.json \
    --clin_path /path/to/cmb_clin.json \
    --output data/eval/cmb_eval.json

  # 只要 CMB-Clin (推荐，case 质量更高)
  python scripts/convert_cmb_to_eval.py --source huggingface --clin_only --output data/eval/cmb_clin_eval.json

  # 控制 Exam 采样数量
  python scripts/convert_cmb_to_eval.py --source huggingface --exam_sample 100 --output data/eval/cmb_eval.json
"""

import json
import re
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional


# ─────────────────────────────────────────────
# 科室映射：CMB exam_subject → MedAgent department
# ─────────────────────────────────────────────

SUBJECT_TO_DEPARTMENT = {
    # 内科系统
    "临床执业医师": "内科",
    "内科学": "内科",
    "呼吸内科": "呼吸内科",
    "心血管内科": "心血管内科",
    "消化内科": "消化内科",
    "内分泌科": "内分泌科",
    "肾内科": "肾内科",
    "神经内科": "神经内科",
    "血液内科": "血液内科",
    "风湿免疫科": "风湿免疫科",
    # 外科系统
    "外科学": "外科",
    "普通外科": "普通外科",
    "骨科": "骨科",
    "泌尿外科": "泌尿外科",
    "神经外科": "神经外科",
    "胸外科": "胸外科",
    # 妇产儿
    "妇产科学": "妇产科",
    "妇产科": "妇产科",
    "儿科学": "儿科",
    "儿科": "儿科",
    # 其他
    "精神科": "精神科",
    "皮肤科": "皮肤科",
    "眼科": "眼科",
    "耳鼻喉科": "耳鼻喉科",
    "口腔执业医师": "口腔科",
    "口腔科": "口腔科",
    "中医执业医师": "中医科",
    "中医科": "中医科",
    "急诊科": "急诊科",
    "全科医学": "全科",
}

# CMB-Clin title 中常见的疾病 → 科室推断
DISEASE_TO_DEPARTMENT = {
    "肺炎": "呼吸内科", "肺结核": "呼吸内科", "哮喘": "呼吸内科", "肺癌": "呼吸内科",
    "支气管": "呼吸内科", "气胸": "呼吸内科", "胸腔积液": "呼吸内科",
    "冠心病": "心血管内科", "心肌梗": "心血管内科", "高血压": "心血管内科",
    "心力衰竭": "心血管内科", "心房颤动": "心血管内科", "心律失常": "心血管内科",
    "胃": "消化内科", "肝": "消化内科", "胰腺": "消化内科", "肠": "消化内科",
    "溃疡": "消化内科", "消化": "消化内科",
    "糖尿病": "内分泌科", "甲状腺": "内分泌科", "甲亢": "内分泌科",
    "肾": "肾内科", "尿毒症": "肾内科",
    "脑": "神经内科", "卒中": "神经内科", "癫痫": "神经内科", "帕金森": "神经内科",
    "贫血": "血液内科", "白血病": "血液内科", "淋巴瘤": "血液内科",
    "关节": "风湿免疫科", "痛风": "风湿免疫科", "红斑狼疮": "风湿免疫科",
    "骨折": "骨科", "脊柱": "骨科",
    "疝": "普通外科", "阑尾": "普通外科", "胆囊": "普通外科",
    "前列腺": "泌尿外科", "膀胱": "泌尿外科",
    "产": "妇产科", "妊娠": "妇产科", "子宫": "妇产科", "卵巢": "妇产科",
    "小儿": "儿科", "新生儿": "儿科",
}


def _infer_department_from_title(title: str) -> str:
    """从 CMB-Clin title 推断科室"""
    for keyword, dept in DISEASE_TO_DEPARTMENT.items():
        if keyword in title:
            return dept
    return "内科"


def _infer_department_from_subject(subject: str) -> str:
    """从 CMB-Exam exam_subject 推断科室"""
    for key, dept in SUBJECT_TO_DEPARTMENT.items():
        if key in subject:
            return dept
    return "内科"


def _is_clinical_scenario(question: str) -> bool:
    """判断 CMB-Exam 题目是否为临床场景题（而非纯知识题）"""
    clinical_markers = [
        "患者", "病人", "男性", "女性", "岁",
        "入院", "就诊", "主诉", "查体", "检查",
        "天前", "小时前", "个月", "年来",
    ]
    return sum(1 for m in clinical_markers if m in question) >= 2


def _extract_patient_description(question: str) -> str:
    """从 CMB-Exam 选择题中提取患者描述部分"""
    # 去掉题目末尾的提问部分
    # 常见模式: "...该患者最可能的诊断是" / "...首选的治疗是"
    cut_patterns = [
        r'[，。](?:该|此|其|本)[患病]?[者人]?(?:最)?(?:可能|应该|首选|需要|适合)',
        r'[，。](?:最可能|最恰当|最合适|首先应|应首先|下列哪项)',
        r'[，。]诊断(?:为|是|考虑)',
        r'[，。]治疗(?:应|首选|方案)',
    ]
    text = question
    for pat in cut_patterns:
        m = re.search(pat, text)
        if m:
            text = text[:m.start() + 1]  # 保留标点
            break
    return text.strip()


def _build_diagnosis_from_answer(question: str, answer: str, options: Dict) -> str:
    """从选择题答案构建预期诊断方向"""
    if answer and answer in options:
        return options[answer]
    return ""


# ─────────────────────────────────────────────
# CMB-Exam 转换
# ─────────────────────────────────────────────

def convert_exam_items(items: List[Dict], max_items: int = 100) -> List[Dict]:
    """将 CMB-Exam 选择题转为 eval_cases"""
    # 只保留临床场景题 + 单选题
    clinical = [
        item for item in items
        if item.get("question_type") == "单项选择题"
        and _is_clinical_scenario(item.get("question", ""))
    ]

    if len(clinical) > max_items:
        random.seed(42)
        clinical = random.sample(clinical, max_items)

    results = []
    for i, item in enumerate(clinical):
        question = item.get("question", "")
        answer = item.get("answer", "")
        options = item.get("option", {})
        subject = item.get("exam_subject", "")

        patient_desc = _extract_patient_description(question)
        if len(patient_desc) < 15:
            continue

        diagnosis = _build_diagnosis_from_answer(question, answer, options)
        if not diagnosis:
            continue

        department = _infer_department_from_subject(subject)

        case = {
            "patient_input": patient_desc,
            "expected_department": department,
            "expected_tools": ["search_guidelines"],
            "expected_diagnosis_direction": diagnosis,
            "source": "CMB-Exam",
            "source_meta": {
                "exam_type": item.get("exam_type", ""),
                "exam_class": item.get("exam_class", ""),
                "exam_subject": subject,
                "original_answer": answer,
                "original_options": options,
            },
        }
        results.append(case)

    return results


# ─────────────────────────────────────────────
# CMB-Clin 转换
# ─────────────────────────────────────────────

def convert_clin_items(items: List[Dict]) -> List[Dict]:
    """将 CMB-Clin 案例分析转为 eval_cases"""
    results = []
    for item in items:
        title = item.get("title", "")
        description = item.get("description", "")
        qa_pairs = item.get("QA_pairs", [])

        if not description or len(description) < 20:
            continue

        # 从 description 提取主诉作为 patient_input
        # CMB-Clin 的 description 包含现病史、体格检查、辅助检查
        patient_input = _simplify_clin_description(description)

        # 从 QA_pairs 提取诊断信息
        diagnosis = ""
        for qa in qa_pairs:
            q = qa.get("question", "")
            s = qa.get("solution", "")
            if "诊断" in q and "鉴别" not in q:
                # 提取诊断结论（通常在 solution 的第一行）
                diagnosis = _extract_diagnosis_from_solution(s)
                break

        if not diagnosis:
            # fallback: 从 title 提取
            diagnosis = title.replace("案例分析-", "").strip()

        department = _infer_department_from_title(title)

        # 确定预期工具
        expected_tools = ["search_guidelines"]
        if any(kw in description for kw in ["血常规", "血糖", "血压", "化验", "检验"]):
            expected_tools.append("interpret_lab_result")
        if any(kw in description for kw in ["用药", "服用", "药物"]):
            expected_tools.append("search_drug")

        case = {
            "patient_input": patient_input,
            "expected_department": department,
            "expected_tools": expected_tools,
            "expected_diagnosis_direction": diagnosis,
            "source": "CMB-Clin",
            "source_meta": {
                "original_id": item.get("id", ""),
                "title": title,
                "qa_count": len(qa_pairs),
            },
        }
        results.append(case)

    return results


def _simplify_clin_description(description: str) -> str:
    """
    将 CMB-Clin 完整病历简化为患者自述风格。
    保留主诉和关键症状，去掉过于专业的体格检查描述。
    """
    lines = description.strip().split("\n")
    # 提取主诉
    chief = ""
    history = ""
    for i, line in enumerate(lines):
        line = line.strip()
        if "主诉" in line:
            # 主诉通常在下一行或同行
            if "：" in line or ":" in line:
                chief = re.split(r'[：:]', line, 1)[-1].strip()
            elif i + 1 < len(lines):
                chief = lines[i + 1].strip()
        if "病史摘要" in line or "现病史" in line:
            # 收集后续几行作为病史
            history_lines = []
            for j in range(i + 1, min(i + 6, len(lines))):
                l = lines[j].strip()
                if l and "体格检查" not in l and "辅助检查" not in l and "主诉" not in l:
                    history_lines.append(l)
                elif "体格检查" in l or "辅助检查" in l:
                    break
            history = " ".join(history_lines)

    if chief:
        return chief
    if history and len(history) > 10:
        # 截取前 200 字符，保持可读
        return history[:200].strip()
    # fallback：取前 200 字符
    return description[:200].strip()


def _extract_diagnosis_from_solution(solution: str) -> str:
    """从 CMB-Clin solution 提取诊断结论"""
    lines = solution.strip().split("\n")
    # 通常第一行是诊断
    first_line = lines[0].strip()

    # 去掉 "诊断：" 前缀
    first_line = re.sub(r'^诊断[：:]?\s*', '', first_line)

    # 如果太长（包含了依据），截断到句号
    if len(first_line) > 50:
        m = re.search(r'[。；]', first_line)
        if m:
            first_line = first_line[:m.start()]

    return first_line.strip()


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_from_huggingface() -> tuple:
    """从 HuggingFace 加载 CMB 数据集"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("需要安装 datasets 库: pip install datasets")

    print("从 HuggingFace 下载 CMB 数据集...")
    exam_ds = load_dataset("FreedomIntelligence/CMB", "exam", trust_remote_code=True)
    clin_ds = load_dataset("FreedomIntelligence/CMB", "clin", trust_remote_code=True)

    # CMB-Exam val split (有解析，质量更高)
    exam_items = []
    for split in ["val", "test"]:
        if split in exam_ds:
            exam_items.extend([dict(item) for item in exam_ds[split]])

    # CMB-Clin
    clin_items = []
    for split in clin_ds:
        clin_items.extend([dict(item) for item in clin_ds[split]])

    print(f"  CMB-Exam: {len(exam_items)} 条")
    print(f"  CMB-Clin: {len(clin_items)} 条")
    return exam_items, clin_items


def load_from_local(exam_path: Optional[str], clin_path: Optional[str]) -> tuple:
    """从本地 JSON 文件加载"""
    exam_items = []
    clin_items = []

    if exam_path and Path(exam_path).exists():
        with open(exam_path, "r", encoding="utf-8") as f:
            exam_items = json.load(f)
        print(f"  CMB-Exam: {len(exam_items)} 条 (from {exam_path})")

    if clin_path and Path(clin_path).exists():
        with open(clin_path, "r", encoding="utf-8") as f:
            clin_items = json.load(f)
        print(f"  CMB-Clin: {len(clin_items)} 条 (from {clin_path})")

    return exam_items, clin_items


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CMB → MedAgent eval_cases 转换")
    parser.add_argument("--source", choices=["huggingface", "local"], default="huggingface")
    parser.add_argument("--exam_path", type=str, help="本地 CMB-Exam JSON 路径")
    parser.add_argument("--clin_path", type=str, help="本地 CMB-Clin JSON 路径")
    parser.add_argument("--output", type=str, default="data/eval/cmb_eval.json")
    parser.add_argument("--exam_sample", type=int, default=100,
                        help="从 CMB-Exam 中采样的最大数量")
    parser.add_argument("--clin_only", action="store_true",
                        help="只使用 CMB-Clin（推荐，case 质量更高）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载数据
    if args.source == "huggingface":
        exam_items, clin_items = load_from_huggingface()
    else:
        exam_items, clin_items = load_from_local(args.exam_path, args.clin_path)

    all_cases = []

    # 转换 CMB-Clin（优先，case 质量高）
    if clin_items:
        clin_cases = convert_clin_items(clin_items)
        all_cases.extend(clin_cases)
        print(f"  CMB-Clin → {len(clin_cases)} eval cases")

    # 转换 CMB-Exam
    if not args.clin_only and exam_items:
        exam_cases = convert_exam_items(exam_items, max_items=args.exam_sample)
        all_cases.extend(exam_cases)
        print(f"  CMB-Exam → {len(exam_cases)} eval cases")

    if not all_cases:
        print("警告: 没有生成任何 eval case")
        return

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"\n总计 {len(all_cases)} eval cases → {output_path}")

    # 统计
    sources = {}
    depts = {}
    for c in all_cases:
        src = c.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        dept = c.get("expected_department", "unknown")
        depts[dept] = depts.get(dept, 0) + 1

    print("\n来源分布:")
    for k, v in sorted(sources.items()):
        print(f"  {k}: {v}")
    print("\n科室分布:")
    for k, v in sorted(depts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
