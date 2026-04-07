"""
扩充药品知识库：从 10 种 → 50+ 种常用药品。
覆盖：降压、降糖、抗生素、解热镇痛、抗凝、消化系统、呼吸系统、
      精神类、心血管、内分泌、抗过敏、泌尿系统等。

用法:
  python scripts/expand_drug_kb.py
  python scripts/expand_drug_kb.py --output data/drug_kb/drug_database.json
"""

import json
import argparse
from pathlib import Path

# 新增 40+ 种药品（保持与已有 10 种相同 schema）
NEW_DRUGS = [
    # ─── 降压药 ───
    {
        "generic_name": "缬沙坦",
        "trade_name": "代文",
        "aliases": ["Valsartan"],
        "category": "血管紧张素II受体拮抗剂（ARB）",
        "indications": ["高血压", "心力衰竭", "心肌梗死后"],
        "contraindications": ["妊娠", "双侧肾动脉狭窄", "高钾血症"],
        "adverse_reactions": ["头晕", "高钾血症", "肾功能下降", "咳嗽（少见）"],
        "dosage": "80-160mg，每日1次",
        "interactions": [
            {"drug": "保钾利尿剂", "severity": "严重", "description": "增加高钾血症风险", "recommendation": "联用时监测血钾"},
            {"drug": "NSAIDs", "severity": "中度", "description": "减弱降压效果并增加肾损害风险", "recommendation": "监测血压和肾功能"},
            {"drug": "锂盐", "severity": "中度", "description": "升高锂血药浓度", "recommendation": "监测锂浓度"}
        ]
    },
    {
        "generic_name": "依那普利",
        "trade_name": "悦宁定",
        "aliases": ["Enalapril"],
        "category": "血管紧张素转换酶抑制剂（ACEI）",
        "indications": ["高血压", "心力衰竭", "糖尿病肾病"],
        "contraindications": ["妊娠", "血管性水肿史", "双侧肾动脉狭窄"],
        "adverse_reactions": ["干咳", "高钾血症", "血管性水肿", "头晕"],
        "dosage": "5-20mg，每日1-2次",
        "interactions": [
            {"drug": "保钾利尿剂", "severity": "严重", "description": "增加高钾血症风险", "recommendation": "监测血钾"},
            {"drug": "NSAIDs", "severity": "中度", "description": "减弱降压效果", "recommendation": "监测血压和肾功能"}
        ]
    },
    {
        "generic_name": "美托洛尔",
        "trade_name": "倍他乐克",
        "aliases": ["Metoprolol"],
        "category": "β受体阻滞剂",
        "indications": ["高血压", "心绞痛", "心力衰竭", "心房颤动", "室上性心动过速"],
        "contraindications": ["严重心动过缓", "二度及以上房室传导阻滞", "心源性休克", "失代偿心衰"],
        "adverse_reactions": ["心动过缓", "低血压", "疲劳", "支气管痉挛", "四肢发冷"],
        "dosage": "缓释片：47.5-190mg，每日1次；普通片：25-100mg，每日2次",
        "interactions": [
            {"drug": "维拉帕米", "severity": "严重", "description": "严重心动过缓和低血压", "recommendation": "避免联用"},
            {"drug": "地尔硫卓", "severity": "严重", "description": "可致严重心动过缓", "recommendation": "联用时密切监测心率"},
            {"drug": "胰岛素", "severity": "中度", "description": "掩盖低血糖症状", "recommendation": "监测血糖"}
        ]
    },
    {
        "generic_name": "氢氯噻嗪",
        "trade_name": "双氢克尿噻",
        "aliases": ["Hydrochlorothiazide", "HCTZ"],
        "category": "噻嗪类利尿剂",
        "indications": ["高血压", "水肿", "心力衰竭"],
        "contraindications": ["无尿", "严重肾功能不全", "严重低钾血症"],
        "adverse_reactions": ["低钾血症", "高尿酸血症", "低钠血症", "高血糖", "光敏感"],
        "dosage": "12.5-25mg，每日1次",
        "interactions": [
            {"drug": "地高辛", "severity": "严重", "description": "低钾增加地高辛毒性", "recommendation": "监测血钾和地高辛浓度"},
            {"drug": "锂盐", "severity": "严重", "description": "减少锂排泄致中毒", "recommendation": "避免联用或监测锂浓度"},
            {"drug": "NSAIDs", "severity": "中度", "description": "减弱利尿降压效果", "recommendation": "监测血压"}
        ]
    },
    {
        "generic_name": "螺内酯",
        "trade_name": "安体舒通",
        "aliases": ["Spironolactone"],
        "category": "醛固酮受体拮抗剂",
        "indications": ["心力衰竭", "肝硬化腹水", "原发性醛固酮增多症", "难治性高血压"],
        "contraindications": ["高钾血症", "严重肾功能不全", "Addison病"],
        "adverse_reactions": ["高钾血症", "男性乳房发育", "月经不规则", "胃肠不适"],
        "dosage": "20-100mg，每日1次",
        "interactions": [
            {"drug": "ACEI/ARB", "severity": "严重", "description": "显著增加高钾血症风险", "recommendation": "联用时密切监测血钾"},
            {"drug": "地高辛", "severity": "中度", "description": "升高地高辛浓度", "recommendation": "监测地高辛浓度"}
        ]
    },
    # ─── 降糖药 ───
    {
        "generic_name": "格列美脲",
        "trade_name": "亚莫利",
        "aliases": ["Glimepiride"],
        "category": "磺脲类促泌剂",
        "indications": ["2型糖尿病"],
        "contraindications": ["1型糖尿病", "糖尿病酮症酸中毒", "严重肝肾功能不全"],
        "adverse_reactions": ["低血糖", "体重增加", "胃肠不适", "皮疹"],
        "dosage": "起始1mg，每日1次；最大6mg/日",
        "interactions": [
            {"drug": "氟康唑", "severity": "严重", "description": "抑制代谢增加低血糖风险", "recommendation": "监测血糖并减量"},
            {"drug": "β受体阻滞剂", "severity": "中度", "description": "掩盖低血糖症状", "recommendation": "加强血糖监测"}
        ]
    },
    {
        "generic_name": "阿卡波糖",
        "trade_name": "拜唐苹",
        "aliases": ["Acarbose"],
        "category": "α-糖苷酶抑制剂",
        "indications": ["2型糖尿病（餐后血糖控制）"],
        "contraindications": ["炎症性肠病", "肠梗阻", "严重肾功能不全"],
        "adverse_reactions": ["腹胀", "排气增多", "腹泻", "转氨酶升高"],
        "dosage": "50mg，每日3次，随第一口饭嚼服；最大100mg/次",
        "interactions": [
            {"drug": "消化酶制剂", "severity": "中度", "description": "降低阿卡波糖疗效", "recommendation": "避免联用"},
            {"drug": "胰岛素", "severity": "中度", "description": "增加低血糖风险", "recommendation": "低血糖时需口服葡萄糖而非蔗糖"}
        ]
    },
    {
        "generic_name": "达格列净",
        "trade_name": "安达唐",
        "aliases": ["Dapagliflozin", "SGLT2抑制剂"],
        "category": "钠-葡萄糖协同转运蛋白2抑制剂（SGLT2i）",
        "indications": ["2型糖尿病", "心力衰竭", "慢性肾脏病"],
        "contraindications": ["1型糖尿病", "严重肾功能不全（eGFR<25）", "糖尿病酮症酸中毒"],
        "adverse_reactions": ["泌尿生殖道感染", "低血压", "酮症酸中毒（罕见）", "脱水"],
        "dosage": "10mg，每日1次",
        "interactions": [
            {"drug": "利尿剂", "severity": "中度", "description": "增加脱水和低血压风险", "recommendation": "联用时注意补液和监测血压"},
            {"drug": "胰岛素", "severity": "中度", "description": "增加低血糖风险", "recommendation": "考虑减少胰岛素剂量"}
        ]
    },
    {
        "generic_name": "西格列汀",
        "trade_name": "捷诺维",
        "aliases": ["Sitagliptin", "DPP-4抑制剂"],
        "category": "二肽基肽酶4抑制剂（DPP-4i）",
        "indications": ["2型糖尿病"],
        "contraindications": ["1型糖尿病", "对本品过敏"],
        "adverse_reactions": ["上呼吸道感染", "头痛", "胰腺炎（罕见）"],
        "dosage": "100mg，每日1次",
        "interactions": [
            {"drug": "磺脲类", "severity": "中度", "description": "增加低血糖风险", "recommendation": "联用时考虑减少磺脲类剂量"},
            {"drug": "地高辛", "severity": "轻度", "description": "轻微升高地高辛浓度", "recommendation": "必要时监测"}
        ]
    },
    # ─── 抗生素 ───
    {
        "generic_name": "阿奇霉素",
        "trade_name": "希舒美",
        "aliases": ["Azithromycin"],
        "category": "大环内酯类抗生素",
        "indications": ["上呼吸道感染", "下呼吸道感染", "皮肤软组织感染", "非淋菌性尿道炎"],
        "contraindications": ["对大环内酯类过敏", "严重肝功能不全"],
        "adverse_reactions": ["腹泻", "恶心", "腹痛", "QT间期延长"],
        "dosage": "500mg 首日，250mg 第2-5日",
        "interactions": [
            {"drug": "华法林", "severity": "中度", "description": "可能增强抗凝效果", "recommendation": "监测INR"},
            {"drug": "地高辛", "severity": "中度", "description": "升高地高辛浓度", "recommendation": "监测地高辛浓度"},
            {"drug": "他汀类", "severity": "中度", "description": "增加横纹肌溶解风险", "recommendation": "联用时注意肌痛症状"}
        ]
    },
    {
        "generic_name": "左氧氟沙星",
        "trade_name": "可乐必妥",
        "aliases": ["Levofloxacin"],
        "category": "氟喹诺酮类抗生素",
        "indications": ["泌尿道感染", "呼吸道感染", "皮肤软组织感染", "前列腺炎"],
        "contraindications": ["妊娠", "18岁以下", "癫痫", "QT间期延长", "肌腱炎/肌腱断裂史"],
        "adverse_reactions": ["恶心", "腹泻", "头晕", "肌腱炎", "QT间期延长", "光敏感"],
        "dosage": "250-750mg，每日1次",
        "interactions": [
            {"drug": "含铝/镁抗酸药", "severity": "中度", "description": "降低左氧氟沙星吸收", "recommendation": "间隔2小时以上服用"},
            {"drug": "华法林", "severity": "中度", "description": "增强抗凝效果", "recommendation": "监测INR"},
            {"drug": "NSAIDs", "severity": "中度", "description": "增加中枢神经系统不良反应", "recommendation": "联用时注意癫痫发作风险"}
        ]
    },
    {
        "generic_name": "头孢地尼",
        "trade_name": "欧意",
        "aliases": ["Cefdinir"],
        "category": "第三代头孢菌素",
        "indications": ["上呼吸道感染", "中耳炎", "皮肤软组织感染", "社区获得性肺炎"],
        "contraindications": ["头孢菌素过敏"],
        "adverse_reactions": ["腹泻", "恶心", "皮疹", "阴道炎"],
        "dosage": "成人：300mg，每日2次或600mg每日1次",
        "interactions": [
            {"drug": "铁剂", "severity": "中度", "description": "铁剂降低头孢地尼吸收", "recommendation": "间隔2小时以上服用"},
            {"drug": "丙磺舒", "severity": "轻度", "description": "升高血药浓度", "recommendation": "注意不良反应"}
        ]
    },
    {
        "generic_name": "甲硝唑",
        "trade_name": "灭滴灵",
        "aliases": ["Metronidazole"],
        "category": "硝基咪唑类抗菌药",
        "indications": ["厌氧菌感染", "滴虫感染", "幽门螺杆菌根除（联合用药）", "牙周炎"],
        "contraindications": ["妊娠早期", "对硝基咪唑类过敏", "活动性中枢神经系统疾病"],
        "adverse_reactions": ["恶心", "金属味", "头痛", "周围神经病变（长期使用）"],
        "dosage": "200-400mg，每日3次",
        "interactions": [
            {"drug": "酒精", "severity": "严重", "description": "双硫仑样反应", "recommendation": "用药期间及停药后3天禁酒"},
            {"drug": "华法林", "severity": "严重", "description": "显著增强抗凝效果", "recommendation": "密切监测INR并减量"},
            {"drug": "锂盐", "severity": "中度", "description": "升高锂浓度", "recommendation": "监测锂浓度"}
        ]
    },
    # ─── 抗凝/抗血栓 ───
    {
        "generic_name": "华法林",
        "trade_name": "华法林钠片",
        "aliases": ["Warfarin"],
        "category": "维生素K拮抗剂",
        "indications": ["心房颤动", "深静脉血栓", "肺栓塞", "机械瓣膜置换术后"],
        "contraindications": ["活动性出血", "严重肝肾功能不全", "妊娠", "近期脑出血"],
        "adverse_reactions": ["出血", "皮肤坏死（罕见）", "紫趾综合征"],
        "dosage": "初始2.5-5mg/日，根据INR调整（目标INR 2-3）",
        "interactions": [
            {"drug": "阿司匹林", "severity": "严重", "description": "显著增加出血风险", "recommendation": "联用时需严格适应证并监测"},
            {"drug": "抗生素", "severity": "中度", "description": "影响维生素K代谢", "recommendation": "抗生素治疗期间加强INR监测"},
            {"drug": "维生素K", "severity": "中度", "description": "拮抗华法林效果", "recommendation": "保持饮食中维生素K摄入稳定"}
        ]
    },
    {
        "generic_name": "利伐沙班",
        "trade_name": "拜瑞妥",
        "aliases": ["Rivaroxaban"],
        "category": "直接口服抗凝药（Xa因子抑制剂）",
        "indications": ["非瓣膜性房颤", "深静脉血栓", "肺栓塞", "髋/膝关节置换术后预防"],
        "contraindications": ["活动性出血", "严重肝病", "妊娠"],
        "adverse_reactions": ["出血", "贫血", "恶心", "转氨酶升高"],
        "dosage": "房颤：20mg/日随餐；DVT/PE：15mg bid×21天后20mg/日",
        "interactions": [
            {"drug": "CYP3A4/P-gp强抑制剂", "severity": "严重", "description": "显著升高利伐沙班浓度", "recommendation": "避免联用酮康唑、伊曲康唑等"},
            {"drug": "抗血小板药", "severity": "严重", "description": "增加出血风险", "recommendation": "评估获益/风险比"}
        ]
    },
    {
        "generic_name": "阿司匹林",
        "trade_name": "拜阿司匹灵",
        "aliases": ["Aspirin", "乙酰水杨酸"],
        "category": "抗血小板药 / 非甾体抗炎药",
        "indications": ["冠心病二级预防", "缺血性脑卒中预防", "急性心肌梗死", "解热镇痛"],
        "contraindications": ["活动性消化道溃疡", "出血倾向", "严重肝肾功能不全", "儿童病毒感染（Reye综合征）"],
        "adverse_reactions": ["胃肠刺激", "消化道出血", "过敏反应", "耳鸣（大剂量）"],
        "dosage": "抗血小板：75-100mg/日；解热：300-600mg/次",
        "interactions": [
            {"drug": "华法林", "severity": "严重", "description": "增加出血风险", "recommendation": "联用时需严格指征"},
            {"drug": "布洛芬", "severity": "中度", "description": "布洛芬减弱阿司匹林抗血小板作用", "recommendation": "阿司匹林服后至少2小时再服布洛芬"},
            {"drug": "ACEI", "severity": "轻度", "description": "大剂量时可能减弱降压效果", "recommendation": "低剂量阿司匹林通常无影响"}
        ]
    },
    # ─── 消化系统 ───
    {
        "generic_name": "泮托拉唑",
        "trade_name": "泮托拉唑钠肠溶片",
        "aliases": ["Pantoprazole"],
        "category": "质子泵抑制剂（PPI）",
        "indications": ["胃食管反流病", "消化性溃疡", "Zollinger-Ellison综合征"],
        "contraindications": ["对PPI过敏"],
        "adverse_reactions": ["头痛", "腹泻", "恶心", "低镁血症（长期）"],
        "dosage": "40mg，每日1次",
        "interactions": [
            {"drug": "甲氨蝶呤", "severity": "中度", "description": "可能升高甲氨蝶呤浓度", "recommendation": "大剂量MTX时暂停PPI"},
            {"drug": "氯吡格雷", "severity": "轻度", "description": "对氯吡格雷影响较小（优于奥美拉唑）", "recommendation": "需联用PPI时优选泮托拉唑"}
        ]
    },
    {
        "generic_name": "多潘立酮",
        "trade_name": "吗丁啉",
        "aliases": ["Domperidone"],
        "category": "多巴胺受体拮抗剂（促胃动力药）",
        "indications": ["功能性消化不良", "恶心呕吐", "胃排空延迟"],
        "contraindications": ["催乳素瘤", "QT间期延长", "严重肝功能不全"],
        "adverse_reactions": ["口干", "头痛", "催乳素升高", "QT间期延长"],
        "dosage": "10mg，每日3次，餐前15-30分钟",
        "interactions": [
            {"drug": "酮康唑", "severity": "严重", "description": "CYP3A4抑制升高多潘立酮浓度", "recommendation": "避免联用"},
            {"drug": "红霉素", "severity": "严重", "description": "增加QT间期延长风险", "recommendation": "避免联用"}
        ]
    },
    {
        "generic_name": "蒙脱石散",
        "trade_name": "思密达",
        "aliases": ["Smectite"],
        "category": "肠道吸附剂",
        "indications": ["急性腹泻", "慢性腹泻", "食管/胃/十二指肠疾病辅助"],
        "contraindications": ["肠梗阻"],
        "adverse_reactions": ["便秘", "腹胀"],
        "dosage": "成人：3g/次，每日3次",
        "interactions": [
            {"drug": "其他口服药", "severity": "中度", "description": "可吸附其他药物降低疗效", "recommendation": "与其他药物间隔2小时以上"}
        ]
    },
    # ─── 呼吸系统 ───
    {
        "generic_name": "沙丁胺醇",
        "trade_name": "万托林",
        "aliases": ["Salbutamol", "Albuterol"],
        "category": "短效β2受体激动剂（SABA）",
        "indications": ["支气管哮喘急性发作", "慢性阻塞性肺疾病", "运动诱发性支气管痉挛"],
        "contraindications": ["对本品过敏"],
        "adverse_reactions": ["心悸", "手抖", "头痛", "低钾血症（大剂量）"],
        "dosage": "吸入：100-200μg/次，按需使用；雾化：2.5-5mg/次",
        "interactions": [
            {"drug": "β受体阻滞剂", "severity": "严重", "description": "β阻滞剂拮抗沙丁胺醇作用", "recommendation": "哮喘患者避免非选择性β阻滞剂"},
            {"drug": "利尿剂", "severity": "中度", "description": "增加低钾血症风险", "recommendation": "联用时监测血钾"}
        ]
    },
    {
        "generic_name": "布地奈德",
        "trade_name": "普米克",
        "aliases": ["Budesonide"],
        "category": "吸入性糖皮质激素（ICS）",
        "indications": ["支气管哮喘", "慢性阻塞性肺疾病"],
        "contraindications": ["对本品过敏", "活动性肺结核"],
        "adverse_reactions": ["口腔念珠菌感染", "声音嘶哑", "咽部不适"],
        "dosage": "200-800μg/日，分1-2次吸入",
        "interactions": [
            {"drug": "CYP3A4抑制剂", "severity": "中度", "description": "升高布地奈德全身暴露", "recommendation": "联用酮康唑/伊曲康唑时注意全身副作用"},
            {"drug": "利托那韦", "severity": "严重", "description": "显著增加全身激素效应", "recommendation": "避免联用"}
        ]
    },
    {
        "generic_name": "氨溴索",
        "trade_name": "沐舒坦",
        "aliases": ["Ambroxol"],
        "category": "黏液溶解剂（祛痰药）",
        "indications": ["急慢性支气管炎", "肺炎", "支气管哮喘痰液黏稠"],
        "contraindications": ["妊娠早期"],
        "adverse_reactions": ["胃肠不适", "皮疹", "过敏反应（罕见）"],
        "dosage": "30mg，每日3次",
        "interactions": [
            {"drug": "抗生素", "severity": "轻度", "description": "可增加抗生素在肺组织的浓度", "recommendation": "协同作用，通常有益"}
        ]
    },
    {
        "generic_name": "孟鲁司特",
        "trade_name": "顺尔宁",
        "aliases": ["Montelukast"],
        "category": "白三烯受体拮抗剂",
        "indications": ["支气管哮喘", "过敏性鼻炎", "运动诱发支气管痉挛"],
        "contraindications": ["对本品过敏"],
        "adverse_reactions": ["头痛", "腹痛", "情绪变化", "失眠"],
        "dosage": "成人：10mg，每晚1次",
        "interactions": [
            {"drug": "苯巴比妥", "severity": "中度", "description": "CYP诱导降低孟鲁司特浓度", "recommendation": "必要时调整剂量"},
            {"drug": "吉非罗齐", "severity": "轻度", "description": "升高孟鲁司特浓度", "recommendation": "通常无需调整"}
        ]
    },
    # ─── 精神/神经系统 ───
    {
        "generic_name": "舍曲林",
        "trade_name": "左洛复",
        "aliases": ["Sertraline"],
        "category": "选择性5-羟色胺再摄取抑制剂（SSRI）",
        "indications": ["抑郁症", "强迫症", "恐慌症", "社交焦虑症", "创伤后应激障碍"],
        "contraindications": ["MAO抑制剂使用14天内", "合用匹莫齐特"],
        "adverse_reactions": ["恶心", "腹泻", "失眠", "头痛", "性功能障碍"],
        "dosage": "起始50mg/日，最大200mg/日",
        "interactions": [
            {"drug": "MAO抑制剂", "severity": "严重", "description": "可致5-HT综合征", "recommendation": "禁止联用，需间隔≥14天"},
            {"drug": "华法林", "severity": "中度", "description": "增加出血风险", "recommendation": "联用时监测INR"},
            {"drug": "曲马多", "severity": "严重", "description": "增加癫痫发作和5-HT综合征风险", "recommendation": "避免联用"}
        ]
    },
    {
        "generic_name": "艾司西酞普兰",
        "trade_name": "来士普",
        "aliases": ["Escitalopram"],
        "category": "选择性5-羟色胺再摄取抑制剂（SSRI）",
        "indications": ["抑郁症", "广泛性焦虑症"],
        "contraindications": ["MAO抑制剂使用14天内", "QT间期延长"],
        "adverse_reactions": ["恶心", "头痛", "失眠", "嗜睡", "性功能障碍"],
        "dosage": "10-20mg，每日1次",
        "interactions": [
            {"drug": "MAO抑制剂", "severity": "严重", "description": "可致5-HT综合征", "recommendation": "禁止联用"},
            {"drug": "西咪替丁", "severity": "中度", "description": "升高艾司西酞普兰浓度", "recommendation": "考虑减量"}
        ]
    },
    {
        "generic_name": "阿普唑仑",
        "trade_name": "佳静安定",
        "aliases": ["Alprazolam"],
        "category": "苯二氮卓类镇静催眠药",
        "indications": ["焦虑症", "惊恐障碍", "失眠"],
        "contraindications": ["重症肌无力", "严重呼吸功能不全", "睡眠呼吸暂停"],
        "adverse_reactions": ["嗜睡", "头晕", "依赖性", "反跳性焦虑", "认知功能下降"],
        "dosage": "0.25-0.5mg，每日3次；最大4mg/日",
        "interactions": [
            {"drug": "阿片类", "severity": "严重", "description": "增加呼吸抑制和死亡风险", "recommendation": "避免联用"},
            {"drug": "CYP3A4抑制剂", "severity": "严重", "description": "升高阿普唑仑浓度", "recommendation": "联用酮康唑等时大幅减量"},
            {"drug": "酒精", "severity": "严重", "description": "增强中枢抑制作用", "recommendation": "禁止联用"}
        ]
    },
    {
        "generic_name": "卡马西平",
        "trade_name": "得理多",
        "aliases": ["Carbamazepine"],
        "category": "抗癫痫药 / 情绪稳定剂",
        "indications": ["癫痫", "三叉神经痛", "双相情感障碍"],
        "contraindications": ["骨髓抑制", "房室传导阻滞", "HLA-B*1502阳性（亚裔需筛查）"],
        "adverse_reactions": ["头晕", "嗜睡", "复视", "低钠血症", "Stevens-Johnson综合征"],
        "dosage": "起始100-200mg bid，缓慢递增",
        "interactions": [
            {"drug": "口服避孕药", "severity": "严重", "description": "CYP诱导降低避孕药浓度", "recommendation": "需更换避孕方法"},
            {"drug": "华法林", "severity": "严重", "description": "加速华法林代谢", "recommendation": "联用时增加INR监测频率"},
            {"drug": "红霉素", "severity": "严重", "description": "抑制卡马西平代谢致中毒", "recommendation": "避免联用或密切监测"}
        ]
    },
    # ─── 抗过敏 ───
    {
        "generic_name": "氯雷他定",
        "trade_name": "开瑞坦",
        "aliases": ["Loratadine"],
        "category": "第二代H1抗组胺药",
        "indications": ["过敏性鼻炎", "荨麻疹", "过敏性皮肤病"],
        "contraindications": ["对本品过敏"],
        "adverse_reactions": ["嗜睡（少见）", "头痛", "口干", "疲劳"],
        "dosage": "10mg，每日1次",
        "interactions": [
            {"drug": "酮康唑", "severity": "中度", "description": "升高氯雷他定浓度", "recommendation": "通常无需调整剂量"},
            {"drug": "红霉素", "severity": "轻度", "description": "轻度升高浓度", "recommendation": "无需特殊处理"}
        ]
    },
    {
        "generic_name": "西替利嗪",
        "trade_name": "仙特明",
        "aliases": ["Cetirizine"],
        "category": "第二代H1抗组胺药",
        "indications": ["过敏性鼻炎", "慢性荨麻疹", "过敏性结膜炎"],
        "contraindications": ["严重肾功能不全", "对本品过敏"],
        "adverse_reactions": ["嗜睡", "口干", "头痛", "疲劳"],
        "dosage": "10mg，每日1次",
        "interactions": [
            {"drug": "中枢抑制剂", "severity": "中度", "description": "增强镇静作用", "recommendation": "联用时注意嗜睡"},
            {"drug": "茶碱", "severity": "轻度", "description": "茶碱可降低西替利嗪清除率", "recommendation": "通常无临床意义"}
        ]
    },
    # ─── 心血管 ───
    {
        "generic_name": "地高辛",
        "trade_name": "地高辛片",
        "aliases": ["Digoxin"],
        "category": "强心苷类",
        "indications": ["心力衰竭", "心房颤动（控制心室率）"],
        "contraindications": ["肥厚梗阻性心肌病", "室性心动过速", "低钾血症"],
        "adverse_reactions": ["恶心", "呕吐", "心律失常", "视觉异常（黄视）"],
        "dosage": "0.125-0.25mg/日",
        "interactions": [
            {"drug": "胺碘酮", "severity": "严重", "description": "升高地高辛浓度50-100%", "recommendation": "联用时地高辛减量50%"},
            {"drug": "利尿剂", "severity": "严重", "description": "低钾增加地高辛毒性", "recommendation": "监测血钾"},
            {"drug": "维拉帕米", "severity": "严重", "description": "升高地高辛浓度", "recommendation": "减量并监测"}
        ]
    },
    {
        "generic_name": "胺碘酮",
        "trade_name": "可达龙",
        "aliases": ["Amiodarone"],
        "category": "III类抗心律失常药",
        "indications": ["室性心动过速", "心室颤动", "心房颤动"],
        "contraindications": ["甲状腺功能异常", "严重传导阻滞", "碘过敏"],
        "adverse_reactions": ["甲状腺功能异常", "肺纤维化", "角膜微沉着", "光敏感", "肝功能损害"],
        "dosage": "负荷：200mg tid×1周→200mg bid×1周→维持200mg/日",
        "interactions": [
            {"drug": "华法林", "severity": "严重", "description": "抑制华法林代谢增强抗凝", "recommendation": "华法林减量30-50%并监测INR"},
            {"drug": "地高辛", "severity": "严重", "description": "升高地高辛浓度", "recommendation": "地高辛减量50%"},
            {"drug": "他汀类", "severity": "严重", "description": "增加横纹肌溶解风险", "recommendation": "辛伐他汀不超过20mg"}
        ]
    },
    {
        "generic_name": "硝酸甘油",
        "trade_name": "硝酸甘油片",
        "aliases": ["Nitroglycerin", "NTG"],
        "category": "硝酸酯类",
        "indications": ["心绞痛急性发作", "急性心力衰竭", "高血压急症"],
        "contraindications": ["严重低血压", "肥厚梗阻性心肌病", "磷酸二酯酶5抑制剂使用24h内"],
        "adverse_reactions": ["头痛", "低血压", "面部潮红", "心动过速"],
        "dosage": "舌下含服：0.5mg/次；贴片：5-10mg/24h",
        "interactions": [
            {"drug": "西地那非", "severity": "严重", "description": "严重低血压甚至死亡", "recommendation": "禁止联用"},
            {"drug": "抗高血压药", "severity": "中度", "description": "增强降压效果", "recommendation": "注意体位性低血压"}
        ]
    },
    # ─── 解热镇痛 ───
    {
        "generic_name": "对乙酰氨基酚",
        "trade_name": "泰诺林",
        "aliases": ["Acetaminophen", "扑热息痛", "Paracetamol"],
        "category": "解热镇痛药",
        "indications": ["发热", "头痛", "关节痛", "肌肉痛", "痛经"],
        "contraindications": ["严重肝功能不全"],
        "adverse_reactions": ["肝毒性（过量）", "皮疹（罕见）"],
        "dosage": "500-1000mg/次，每4-6小时1次，最大4g/日",
        "interactions": [
            {"drug": "华法林", "severity": "中度", "description": "大剂量长期使用增强抗凝效果", "recommendation": "长期联用时监测INR"},
            {"drug": "酒精", "severity": "严重", "description": "增加肝毒性风险", "recommendation": "避免酗酒"},
            {"drug": "卡马西平", "severity": "中度", "description": "加速对乙酰氨基酚代谢增加肝毒性", "recommendation": "注意肝功能"}
        ]
    },
    {
        "generic_name": "双氯芬酸",
        "trade_name": "扶他林",
        "aliases": ["Diclofenac"],
        "category": "非甾体抗炎药（NSAIDs）",
        "indications": ["类风湿关节炎", "骨关节炎", "强直性脊柱炎", "急性痛风", "术后疼痛"],
        "contraindications": ["活动性消化性溃疡", "冠脉搭桥术围手术期", "严重心衰"],
        "adverse_reactions": ["胃肠不适", "转氨酶升高", "肾功能损害", "心血管事件（长期使用）"],
        "dosage": "75-150mg/日，分2-3次",
        "interactions": [
            {"drug": "华法林", "severity": "严重", "description": "增加出血风险", "recommendation": "联用时监测INR"},
            {"drug": "ACEI/ARB", "severity": "中度", "description": "减弱降压效果和肾功能损害", "recommendation": "监测血压和肾功能"},
            {"drug": "锂盐", "severity": "中度", "description": "升高锂浓度", "recommendation": "监测锂浓度"}
        ]
    },
    # ─── 内分泌 ───
    {
        "generic_name": "左甲状腺素钠",
        "trade_name": "优甲乐",
        "aliases": ["Levothyroxine", "L-T4"],
        "category": "甲状腺激素",
        "indications": ["甲状腺功能减退", "甲状腺癌术后替代", "甲状腺肿"],
        "contraindications": ["未纠正的肾上腺皮质功能不全", "急性心肌梗死"],
        "adverse_reactions": ["过量症状：心悸、手抖、失眠、体重下降", "骨质疏松（长期过量）"],
        "dosage": "成人起始25-50μg/日，缓慢递增至75-200μg/日",
        "interactions": [
            {"drug": "钙剂/铁剂", "severity": "中度", "description": "降低左甲状腺素吸收", "recommendation": "间隔4小时以上服用"},
            {"drug": "华法林", "severity": "中度", "description": "增强抗凝效果", "recommendation": "甲状腺功能改变时重新评估INR"},
            {"drug": "奥美拉唑", "severity": "轻度", "description": "可能降低吸收", "recommendation": "空腹服用左甲状腺素"}
        ]
    },
    {
        "generic_name": "泼尼松",
        "trade_name": "泼尼松片",
        "aliases": ["Prednisone"],
        "category": "糖皮质激素",
        "indications": ["自身免疫性疾病", "过敏性疾病", "哮喘急性发作", "肾病综合征", "炎症性肠病"],
        "contraindications": ["全身性真菌感染", "活动性结核（未治疗）"],
        "adverse_reactions": ["高血糖", "骨质疏松", "库欣综合征", "免疫抑制", "消化性溃疡", "精神症状"],
        "dosage": "根据病情5-60mg/日，晨起顿服",
        "interactions": [
            {"drug": "NSAIDs", "severity": "中度", "description": "增加消化道溃疡风险", "recommendation": "联用时加PPI保护"},
            {"drug": "降糖药", "severity": "中度", "description": "升高血糖", "recommendation": "调整降糖药剂量"},
            {"drug": "利尿剂", "severity": "中度", "description": "增加低钾血症风险", "recommendation": "监测血钾"}
        ]
    },
    # ─── 泌尿系统 ───
    {
        "generic_name": "坦索罗辛",
        "trade_name": "哈乐",
        "aliases": ["Tamsulosin"],
        "category": "α1受体阻滞剂",
        "indications": ["良性前列腺增生排尿困难"],
        "contraindications": ["体位性低血压", "严重肝功能不全"],
        "adverse_reactions": ["头晕", "体位性低血压", "射精异常", "鼻塞"],
        "dosage": "0.2mg，每日1次，餐后服用",
        "interactions": [
            {"drug": "CYP3A4抑制剂", "severity": "中度", "description": "升高坦索罗辛浓度", "recommendation": "联用酮康唑时注意低血压"},
            {"drug": "降压药", "severity": "中度", "description": "增加低血压风险", "recommendation": "联用时监测血压"}
        ]
    },
    {
        "generic_name": "非那雄胺",
        "trade_name": "保列治",
        "aliases": ["Finasteride"],
        "category": "5α-还原酶抑制剂",
        "indications": ["良性前列腺增生", "男性型脱发"],
        "contraindications": ["妊娠（致畸）", "女性和儿童"],
        "adverse_reactions": ["性欲减退", "勃起功能障碍", "射精量减少", "乳房触痛"],
        "dosage": "前列腺增生：5mg/日；脱发：1mg/日",
        "interactions": [
            {"drug": "无显著临床相互作用", "severity": "轻度", "description": "非那雄胺药物相互作用少", "recommendation": "常规使用即可"}
        ]
    },
    # ─── 骨骼/关节 ───
    {
        "generic_name": "别嘌醇",
        "trade_name": "别嘌醇片",
        "aliases": ["Allopurinol"],
        "category": "黄嘌呤氧化酶抑制剂",
        "indications": ["高尿酸血症", "痛风", "尿酸性肾病"],
        "contraindications": ["HLA-B*5801阳性（需筛查）", "急性痛风发作期单独使用"],
        "adverse_reactions": ["皮疹", "肝功能异常", "Stevens-Johnson综合征（严重）", "骨髓抑制"],
        "dosage": "起始100mg/日，渐增至300mg/日",
        "interactions": [
            {"drug": "硫唑嘌呤", "severity": "严重", "description": "别嘌醇抑制硫唑嘌呤代谢致严重骨髓抑制", "recommendation": "联用时硫唑嘌呤减量75%"},
            {"drug": "华法林", "severity": "中度", "description": "可能增强抗凝效果", "recommendation": "监测INR"},
            {"drug": "ACEI", "severity": "中度", "description": "增加超敏反应风险", "recommendation": "注意过敏症状"}
        ]
    },
    {
        "generic_name": "秋水仙碱",
        "trade_name": "秋水仙碱片",
        "aliases": ["Colchicine"],
        "category": "抗痛风药",
        "indications": ["急性痛风发作", "痛风预防", "家族性地中海热"],
        "contraindications": ["严重肝肾功能不全", "骨髓抑制"],
        "adverse_reactions": ["腹泻", "恶心", "呕吐", "骨髓抑制（大剂量）"],
        "dosage": "急性发作：1mg起始后0.5mg/h×1次；预防：0.5mg/日",
        "interactions": [
            {"drug": "CYP3A4抑制剂", "severity": "严重", "description": "升高秋水仙碱浓度致严重毒性", "recommendation": "联用克拉霉素/酮康唑时禁用或大幅减量"},
            {"drug": "他汀类", "severity": "中度", "description": "增加肌病风险", "recommendation": "联用时注意肌痛症状"}
        ]
    },
    # ─── 补充几种常用药 ───
    {
        "generic_name": "碳酸钙D3",
        "trade_name": "钙尔奇D",
        "aliases": ["Calcium Carbonate + Vitamin D3"],
        "category": "钙+维生素D补充剂",
        "indications": ["骨质疏松预防", "钙缺乏", "妊娠和哺乳期补钙"],
        "contraindications": ["高钙血症", "严重肾功能不全", "维生素D过量"],
        "adverse_reactions": ["便秘", "腹胀", "高钙血症（过量）"],
        "dosage": "钙600mg + VD3 125IU，每日1-2次",
        "interactions": [
            {"drug": "左甲状腺素", "severity": "中度", "description": "降低甲状腺素吸收", "recommendation": "间隔4小时以上"},
            {"drug": "四环素", "severity": "中度", "description": "钙离子螯合降低吸收", "recommendation": "间隔2小时以上"},
            {"drug": "噻嗪类利尿剂", "severity": "中度", "description": "减少钙排泄增加高钙风险", "recommendation": "联用时监测血钙"}
        ]
    },
    {
        "generic_name": "氯化钾",
        "trade_name": "补达秀",
        "aliases": ["Potassium Chloride", "KCl"],
        "category": "电解质补充剂",
        "indications": ["低钾血症", "预防利尿剂引起的低钾"],
        "contraindications": ["高钾血症", "严重肾功能不全", "Addison病未治疗"],
        "adverse_reactions": ["胃肠刺激", "恶心", "高钾血症（过量）"],
        "dosage": "1-2g/次，每日2-3次，餐后服用",
        "interactions": [
            {"drug": "ACEI/ARB", "severity": "严重", "description": "增加高钾血症风险", "recommendation": "联用时密切监测血钾"},
            {"drug": "保钾利尿剂", "severity": "严重", "description": "增加高钾血症风险", "recommendation": "避免联用"}
        ]
    },
    {
        "generic_name": "维生素B12",
        "trade_name": "弥可保",
        "aliases": ["Mecobalamin", "甲钴胺"],
        "category": "维生素类",
        "indications": ["周围神经病变", "巨幼细胞性贫血", "糖尿病周围神经病变"],
        "contraindications": ["对本品过敏"],
        "adverse_reactions": ["胃肠不适（少见）", "皮疹（罕见）"],
        "dosage": "500μg，每日3次",
        "interactions": [
            {"drug": "无显著相互作用", "severity": "轻度", "description": "甲钴胺药物相互作用极少", "recommendation": "常规使用"}
        ]
    },
    {
        "generic_name": "呋塞米",
        "trade_name": "速尿",
        "aliases": ["Furosemide"],
        "category": "袢利尿剂",
        "indications": ["急性肺水肿", "心力衰竭水肿", "肾性水肿", "肝硬化腹水"],
        "contraindications": ["无尿", "严重电解质紊乱", "肝性脑病"],
        "adverse_reactions": ["低钾血症", "低钠血症", "脱水", "高尿酸血症", "耳毒性（大剂量）"],
        "dosage": "口服20-80mg/次，必要时增加；IV 20-40mg",
        "interactions": [
            {"drug": "氨基糖苷类", "severity": "严重", "description": "增加耳毒性和肾毒性", "recommendation": "避免联用或密切监测"},
            {"drug": "地高辛", "severity": "严重", "description": "低钾增加地高辛毒性", "recommendation": "监测血钾"},
            {"drug": "锂盐", "severity": "严重", "description": "减少锂排泄致中毒", "recommendation": "监测锂浓度"}
        ]
    },
    {
        "generic_name": "格列齐特",
        "trade_name": "达美康",
        "aliases": ["Gliclazide"],
        "category": "磺脲类促泌剂",
        "indications": ["2型糖尿病"],
        "contraindications": ["1型糖尿病", "糖尿病酮症酸中毒", "严重肝肾功能不全"],
        "adverse_reactions": ["低血糖", "体重增加", "胃肠不适"],
        "dosage": "缓释片30-120mg/日，早餐时服用",
        "interactions": [
            {"drug": "氟康唑", "severity": "严重", "description": "抑制代谢增加低血糖风险", "recommendation": "联用时密切监测血糖"},
            {"drug": "β受体阻滞剂", "severity": "中度", "description": "掩盖低血糖症状", "recommendation": "加强血糖监测"}
        ]
    },
    {
        "generic_name": "瑞舒伐他汀",
        "trade_name": "可定",
        "aliases": ["Rosuvastatin"],
        "category": "HMG-CoA还原酶抑制剂（他汀类）",
        "indications": ["高胆固醇血症", "动脉粥样硬化", "冠心病预防"],
        "contraindications": ["活动性肝病", "妊娠及哺乳期", "严重肾功能不全（高剂量）"],
        "adverse_reactions": ["肌痛", "转氨酶升高", "头痛", "横纹肌溶解（罕见）"],
        "dosage": "5-20mg，每日1次",
        "interactions": [
            {"drug": "环孢素", "severity": "严重", "description": "显著增加他汀暴露量", "recommendation": "避免联用"},
            {"drug": "吉非罗齐", "severity": "严重", "description": "增加横纹肌溶解风险", "recommendation": "避免联用"},
            {"drug": "华法林", "severity": "中度", "description": "起始时可能影响INR", "recommendation": "治疗初期监测INR"}
        ]
    },
]


def main():
    parser = argparse.ArgumentParser(description="扩充药品知识库")
    parser.add_argument("--output", type=str,
                        default="data/drug_kb/drug_database.json")
    args = parser.parse_args()

    output_path = Path(args.output)

    # 读取已有数据
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing_names = {d["generic_name"] for d in existing}
    print(f"已有药品: {len(existing)} 种")

    # 合并新药（去重）
    added = 0
    for drug in NEW_DRUGS:
        if drug["generic_name"] not in existing_names:
            existing.append(drug)
            existing_names.add(drug["generic_name"])
            added += 1

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"新增: {added} 种，总计: {len(existing)} 种")
    print(f"保存至: {output_path}")

    # 按类别统计
    cats = {}
    for d in existing:
        cat = d.get("category", "未分类")
        cats[cat] = cats.get(cat, 0) + 1
    print("\n按类别分布:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
