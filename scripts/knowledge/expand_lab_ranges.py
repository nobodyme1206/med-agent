"""
扩充检验值参考范围：从 16 项 → 40+ 项。
补充：凝血功能、甲状腺全套、血脂全套、肿瘤标志物、尿常规、心肌标志物、
      铁代谢、电解质、炎症指标等。

用法:
  python scripts/expand_lab_ranges.py
  python scripts/expand_lab_ranges.py --output data/lab_ranges/lab_ranges.json
"""

import json
import argparse
from pathlib import Path

NEW_LAB_RANGES = {
    # ─── 凝血功能 ───
    "凝血酶原时间": {
        "aliases": ["PT", "Prothrombin Time"],
        "unit": "秒",
        "low": 11.0,
        "high": 13.5,
        "low_causes": ["高凝状态", "DIC早期"],
        "high_causes": ["华法林治疗", "肝功能不全", "维生素K缺乏", "DIC", "凝血因子缺乏"],
        "significance": "外源性凝血途径筛查指标，监测华法林疗效（INR）"
    },
    "INR": {
        "aliases": ["国际标准化比值", "International Normalized Ratio"],
        "unit": "",
        "low": 0.8,
        "high": 1.2,
        "low_causes": ["高凝状态"],
        "high_causes": ["华法林过量", "肝功能不全", "维生素K缺乏", "DIC"],
        "significance": "标准化PT，用于华法林抗凝监测。房颤/DVT目标2-3，机械瓣目标2.5-3.5"
    },
    "APTT": {
        "aliases": ["活化部分凝血活酶时间", "Activated Partial Thromboplastin Time"],
        "unit": "秒",
        "low": 25,
        "high": 37,
        "low_causes": ["高凝状态", "DIC早期"],
        "high_causes": ["肝素治疗", "血友病", "DIC", "狼疮抗凝物"],
        "significance": "内源性凝血途径筛查，监测肝素疗效"
    },
    "D-二聚体": {
        "aliases": ["D-Dimer", "D-D"],
        "unit": "mg/L FEU",
        "low": 0,
        "high": 0.5,
        "low_causes": [],
        "high_causes": ["深静脉血栓", "肺栓塞", "DIC", "手术后", "恶性肿瘤", "感染"],
        "significance": "纤溶活性标志物，阴性可排除VTE；阳性需结合临床"
    },
    "纤维蛋白原": {
        "aliases": ["FIB", "Fibrinogen", "Fg"],
        "unit": "g/L",
        "low": 2.0,
        "high": 4.0,
        "low_causes": ["DIC消耗期", "严重肝病", "纤溶亢进"],
        "high_causes": ["急性感染", "炎症", "恶性肿瘤", "妊娠", "心血管事件"],
        "significance": "凝血最终底物，也是急性时相蛋白"
    },
    # ─── 甲状腺全套 ───
    "FT3": {
        "aliases": ["游离三碘甲状腺原氨酸", "Free T3"],
        "unit": "pmol/L",
        "low": 3.1,
        "high": 6.8,
        "low_causes": ["甲状腺功能减退", "严重全身性疾病", "低T3综合征"],
        "high_causes": ["甲状腺功能亢进", "T3型甲亢"],
        "significance": "生物活性最强的甲状腺激素，甲亢时FT3常先于FT4升高"
    },
    "FT4": {
        "aliases": ["游离甲状腺素", "Free T4", "游离T4"],
        "unit": "pmol/L",
        "low": 12.0,
        "high": 22.0,
        "low_causes": ["甲状腺功能减退", "垂体功能低下"],
        "high_causes": ["甲状腺功能亢进", "亚急性甲状腺炎", "外源性甲状腺素过量"],
        "significance": "不受TBG影响，比总T4更准确反映甲状腺功能"
    },
    # ─── 血脂补充 ───
    "高密度脂蛋白": {
        "aliases": ["HDL", "HDL-C", "高密度脂蛋白胆固醇"],
        "unit": "mmol/L",
        "low": 1.0,
        "high": 99,
        "low_causes": ["代谢综合征", "糖尿病", "吸烟", "肥胖", "遗传因素"],
        "high_causes": [],
        "significance": "抗动脉粥样硬化的保护因子，<1.0mmol/L为心血管危险因素"
    },
    "甘油三酯": {
        "aliases": ["TG", "Triglyceride"],
        "unit": "mmol/L",
        "low": 0.3,
        "high": 1.7,
        "low_causes": ["甲亢", "营养不良", "肝功能衰竭"],
        "high_causes": ["高脂饮食", "糖尿病", "肥胖", "酒精性", "肾病综合征", "急性胰腺炎风险"],
        "significance": "心血管风险因素，>5.6mmol/L时急性胰腺炎风险显著增加"
    },
    # ─── 心肌标志物 ───
    "肌钙蛋白I": {
        "aliases": ["cTnI", "Troponin I", "心肌肌钙蛋白I"],
        "unit": "ng/mL",
        "low": 0,
        "high": 0.04,
        "low_causes": [],
        "high_causes": ["急性心肌梗死", "心肌炎", "心力衰竭", "肺栓塞", "肾功能不全", "脓毒症"],
        "significance": "心肌损伤最特异的标志物，AMI诊断金标准"
    },
    "BNP": {
        "aliases": ["脑钠肽", "B-type Natriuretic Peptide", "B型钠尿肽"],
        "unit": "pg/mL",
        "low": 0,
        "high": 100,
        "low_causes": [],
        "high_causes": ["心力衰竭", "急性冠脉综合征", "肺栓塞", "房颤", "肾功能不全"],
        "significance": "心衰筛查和预后评估指标。<100pg/mL基本排除心衰"
    },
    "NT-proBNP": {
        "aliases": ["N端脑钠肽前体", "N-terminal pro-BNP"],
        "unit": "pg/mL",
        "low": 0,
        "high": 125,
        "low_causes": [],
        "high_causes": ["心力衰竭", "急性冠脉综合征", "肾功能不全", "房颤"],
        "significance": "心衰标志物，半衰期长于BNP，阈值随年龄增加（>75岁：<450pg/mL）"
    },
    "CK-MB": {
        "aliases": ["肌酸激酶同工酶MB", "Creatine Kinase MB"],
        "unit": "U/L",
        "low": 0,
        "high": 25,
        "low_causes": [],
        "high_causes": ["急性心肌梗死", "心肌炎", "心脏手术后", "横纹肌溶解（少量）"],
        "significance": "传统心肌损伤标志物，升高提示心肌损伤，但特异性不如肌钙蛋白"
    },
    # ─── 肝功能补充 ───
    "总胆红素": {
        "aliases": ["TBIL", "Total Bilirubin", "胆红素"],
        "unit": "μmol/L",
        "low": 3.4,
        "high": 17.1,
        "low_causes": [],
        "high_causes": ["肝细胞损伤", "胆道梗阻", "溶血性贫血", "Gilbert综合征", "新生儿黄疸"],
        "significance": "黄疸的直接指标，结合直接/间接胆红素区分黄疸类型"
    },
    "直接胆红素": {
        "aliases": ["DBIL", "Direct Bilirubin", "结合胆红素"],
        "unit": "μmol/L",
        "low": 0,
        "high": 6.8,
        "low_causes": [],
        "high_causes": ["胆道梗阻", "肝内胆汁淤积", "Dubin-Johnson综合征"],
        "significance": "升高提示梗阻性或肝细胞性黄疸"
    },
    "白蛋白": {
        "aliases": ["ALB", "Albumin", "血清白蛋白"],
        "unit": "g/L",
        "low": 35,
        "high": 55,
        "low_causes": ["肝硬化", "肾病综合征", "营养不良", "慢性消耗性疾病", "烧伤"],
        "high_causes": ["脱水"],
        "significance": "反映肝脏合成功能和营养状态，<30g/L为显著低白蛋白血症"
    },
    "GGT": {
        "aliases": ["γ-谷氨酰转肽酶", "Gamma-Glutamyl Transferase", "γ-GT"],
        "unit": "U/L",
        "low": 0,
        "high": 50,
        "low_causes": [],
        "high_causes": ["酒精性肝病", "胆道疾病", "药物性肝损伤", "脂肪肝", "胰腺疾病"],
        "significance": "胆道和肝脏损伤指标，对酒精性肝损伤敏感"
    },
    "碱性磷酸酶": {
        "aliases": ["ALP", "Alkaline Phosphatase"],
        "unit": "U/L",
        "low": 45,
        "high": 125,
        "low_causes": ["甲减", "贫血", "营养不良"],
        "high_causes": ["胆道梗阻", "骨疾病（Paget病）", "甲亢", "妊娠", "儿童生长期"],
        "significance": "胆道梗阻和骨代谢标志物，需结合GGT区分来源"
    },
    # ─── 电解质补充 ───
    "钙": {
        "aliases": ["Ca", "Ca2+", "血钙", "Calcium"],
        "unit": "mmol/L",
        "low": 2.1,
        "high": 2.55,
        "low_causes": ["甲状旁腺功能减退", "维生素D缺乏", "慢性肾病", "低白蛋白血症", "急性胰腺炎"],
        "high_causes": ["原发性甲旁亢", "恶性肿瘤骨转移", "维生素D过量", "结节病"],
        "significance": "维持神经肌肉兴奋性和骨骼健康，异常可致心律失常"
    },
    "磷": {
        "aliases": ["P", "血磷", "Phosphorus", "Phosphate"],
        "unit": "mmol/L",
        "low": 0.8,
        "high": 1.45,
        "low_causes": ["甲旁亢", "维生素D缺乏", "呼吸性碱中毒", "重新喂养综合征"],
        "high_causes": ["慢性肾病", "甲状旁腺功能减退", "横纹肌溶解", "肿瘤溶解综合征"],
        "significance": "与钙代谢密切相关，慢性肾病管理的重要指标"
    },
    "镁": {
        "aliases": ["Mg", "Mg2+", "血镁", "Magnesium"],
        "unit": "mmol/L",
        "low": 0.75,
        "high": 1.02,
        "low_causes": ["长期PPI使用", "利尿剂", "酒精中毒", "腹泻", "糖尿病"],
        "high_causes": ["肾功能不全", "镁剂过量", "甲减"],
        "significance": "参与300+酶反应，低镁可致心律失常和难治性低钾"
    },
    # ─── 铁代谢 ───
    "血清铁": {
        "aliases": ["Fe", "Iron", "Serum Iron"],
        "unit": "μmol/L",
        "low": 10.6,
        "high": 28.3,
        "low_causes": ["缺铁性贫血", "慢性病贫血", "感染"],
        "high_causes": ["血色病", "铁剂过量", "溶血性贫血", "再生障碍性贫血"],
        "significance": "需结合铁蛋白和TIBC综合评估铁代谢状态"
    },
    "铁蛋白": {
        "aliases": ["Ferritin", "SF", "血清铁蛋白"],
        "unit": "μg/L",
        "low": 20,
        "high": 200,
        "low_causes": ["缺铁性贫血（最早改变的指标）"],
        "high_causes": ["铁过载", "炎症", "肝病", "恶性肿瘤", "噬血细胞综合征"],
        "significance": "体内铁储备最可靠的指标。<20μg/L即可诊断缺铁"
    },
    # ─── 炎症指标 ───
    "降钙素原": {
        "aliases": ["PCT", "Procalcitonin"],
        "unit": "ng/mL",
        "low": 0,
        "high": 0.05,
        "low_causes": [],
        "high_causes": ["细菌感染（尤其脓毒症）", "严重创伤", "大手术后", "器官移植排斥"],
        "significance": "细菌感染特异性高于CRP。>0.5ng/mL高度提示细菌感染，>2ng/mL提示脓毒症"
    },
    "红细胞沉降率": {
        "aliases": ["ESR", "血沉", "Erythrocyte Sedimentation Rate"],
        "unit": "mm/h",
        "low": 0,
        "high": 20,
        "low_causes": ["红细胞增多症", "DIC"],
        "high_causes": ["感染", "自身免疫性疾病", "恶性肿瘤", "贫血", "妊娠"],
        "significance": "非特异性炎症指标，常用于风湿病活动度监测"
    },
    # ─── 尿常规 ───
    "尿蛋白": {
        "aliases": ["PRO", "Urine Protein", "尿蛋白定性"],
        "unit": "",
        "low": 0,
        "high": 0,
        "low_causes": [],
        "high_causes": ["肾小球肾炎", "糖尿病肾病", "肾病综合征", "发热", "剧烈运动后"],
        "significance": "肾脏损伤筛查指标，阳性需进一步定量和分型"
    },
    "尿糖": {
        "aliases": ["GLU（尿）", "Urine Glucose"],
        "unit": "",
        "low": 0,
        "high": 0,
        "low_causes": [],
        "high_causes": ["糖尿病血糖>10mmol/L", "肾性糖尿", "妊娠", "SGLT2抑制剂使用"],
        "significance": "血糖超过肾糖阈时出现，用药SGLT2i后尿糖阳性为正常"
    },
    # ─── 肿瘤标志物 ───
    "AFP": {
        "aliases": ["甲胎蛋白", "Alpha-Fetoprotein"],
        "unit": "ng/mL",
        "low": 0,
        "high": 20,
        "low_causes": [],
        "high_causes": ["肝细胞癌", "生殖细胞肿瘤", "肝炎/肝硬化活动期", "妊娠"],
        "significance": "肝癌筛查首选标志物，联合超声用于肝硬化患者监测"
    },
    "CEA": {
        "aliases": ["癌胚抗原", "Carcinoembryonic Antigen"],
        "unit": "ng/mL",
        "low": 0,
        "high": 5.0,
        "low_causes": [],
        "high_causes": ["结直肠癌", "肺癌", "胃癌", "乳腺癌", "吸烟", "炎症性肠病"],
        "significance": "广谱肿瘤标志物，主要用于结直肠癌疗效监测和复发监测"
    },
    "PSA": {
        "aliases": ["前列腺特异性抗原", "Prostate Specific Antigen"],
        "unit": "ng/mL",
        "low": 0,
        "high": 4.0,
        "low_causes": [],
        "high_causes": ["前列腺癌", "良性前列腺增生", "前列腺炎", "前列腺活检后"],
        "significance": "前列腺癌筛查指标，>10ng/mL癌症概率>50%"
    },
    "CA125": {
        "aliases": ["糖类抗原125", "Cancer Antigen 125"],
        "unit": "U/mL",
        "low": 0,
        "high": 35,
        "low_causes": [],
        "high_causes": ["卵巢癌", "子宫内膜异位症", "盆腔炎", "肝硬化腹水", "妊娠早期"],
        "significance": "卵巢癌首选标志物，也用于子宫内膜异位症评估"
    },
    "CA19-9": {
        "aliases": ["糖类抗原19-9", "Cancer Antigen 19-9"],
        "unit": "U/mL",
        "low": 0,
        "high": 37,
        "low_causes": [],
        "high_causes": ["胰腺癌", "胆管癌", "胃癌", "结直肠癌", "胆道梗阻", "胰腺炎"],
        "significance": "胰腺癌最有价值的标志物，也用于胆道肿瘤评估"
    },
}


def main():
    parser = argparse.ArgumentParser(description="扩充检验值参考范围")
    parser.add_argument("--output", type=str,
                        default="data/lab_ranges/lab_ranges.json")
    args = parser.parse_args()

    output_path = Path(args.output)

    # 读取已有数据
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    print(f"已有检验项: {len(existing)} 项")

    # 合并（去重）
    added = 0
    for name, info in NEW_LAB_RANGES.items():
        if name not in existing:
            existing[name] = info
            added += 1

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"新增: {added} 项，总计: {len(existing)} 项")
    print(f"保存至: {output_path}")

    # 列出所有项
    print("\n全部检验项:")
    for i, name in enumerate(existing.keys(), 1):
        aliases = existing[name].get("aliases", [])
        print(f"  {i:2d}. {name} ({', '.join(aliases[:2])})")


if __name__ == "__main__":
    main()
