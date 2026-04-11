"""
医学同义词归一化模块：将常见医学同义表述统一为规范形式。

用于评测时消除 "2型糖尿病" vs "T2DM" 等同义表述导致的误判。
覆盖：疾病名称、症状、检查项目、药物名称、解剖部位等。
"""

import re
from typing import List, Set, Tuple

# ─────────────────────────────────────────────
# 同义词组定义：每组内任意两个词视为等价
# 组内第一个词为规范形式（canonical）
# ─────────────────────────────────────────────

SYNONYM_GROUPS: List[Set[str]] = [
    # ========== 内分泌 ==========
    {"2型糖尿病", "T2DM", "II型糖尿病", "二型糖尿病", "非胰岛素依赖型糖尿病", "NIDDM"},
    {"1型糖尿病", "T1DM", "I型糖尿病", "一型糖尿病", "胰岛素依赖型糖尿病", "IDDM"},
    {"糖尿病", "DM", "diabetes mellitus"},
    {"甲状腺功能亢进", "甲亢", "Graves病", "甲状腺毒症"},
    {"甲状腺功能减退", "甲减", "甲状腺功能低下"},
    {"糖化血红蛋白", "HbA1c", "糖化", "glycated hemoglobin"},
    {"空腹血糖", "FPG", "FBG", "空腹血糖值"},
    {"餐后血糖", "PPG", "餐后2小时血糖", "OGTT"},

    # ========== 心血管 ==========
    {"高血压", "HBP", "原发性高血压", "高血压病"},
    {"高血压1级", "一级高血压", "轻度高血压"},
    {"高血压2级", "二级高血压", "中度高血压"},
    {"高血压3级", "三级高血压", "重度高血压"},
    {"冠心病", "冠状动脉粥样硬化性心脏病", "CAD", "缺血性心脏病", "冠状动脉疾病"},
    {"心肌梗死", "心梗", "MI", "AMI", "急性心肌梗死", "急性心梗"},
    {"心房颤动", "房颤", "AF", "Afib", "心房纤颤"},
    {"心力衰竭", "心衰", "HF", "CHF", "充血性心力衰竭"},
    {"心绞痛", "angina", "心绞痛发作"},
    {"动脉粥样硬化", "AS", "atherosclerosis"},
    {"高脂血症", "高血脂", "血脂异常", "dyslipidemia"},
    {"低密度脂蛋白", "LDL", "LDL-C", "低密度脂蛋白胆固醇"},
    {"高密度脂蛋白", "HDL", "HDL-C", "高密度脂蛋白胆固醇"},

    # ========== 呼吸系统 ==========
    {"慢性阻塞性肺疾病", "COPD", "慢阻肺", "慢性阻塞性肺病"},
    {"支气管哮喘", "哮喘", "asthma"},
    {"肺炎", "pneumonia", "肺部感染"},
    {"上呼吸道感染", "上感", "URI", "感冒", "普通感冒"},
    {"肺结核", "TB", "tuberculosis", "结核"},
    {"肺栓塞", "PE", "pulmonary embolism"},

    # ========== 消化系统 ==========
    {"消化性溃疡", "胃溃疡", "PU", "peptic ulcer"},
    {"十二指肠溃疡", "DU", "duodenal ulcer"},
    {"胃食管反流病", "GERD", "反流性食管炎", "胃食管反流"},
    {"幽门螺杆菌", "HP", "Hp", "幽门螺旋杆菌", "H.pylori"},
    {"肝硬化", "liver cirrhosis", "肝硬变"},
    {"脂肪肝", "脂肪性肝病", "fatty liver", "NAFLD", "非酒精性脂肪肝"},
    {"炎症性肠病", "IBD", "inflammatory bowel disease"},
    {"溃疡性结肠炎", "UC", "ulcerative colitis"},
    {"克罗恩病", "CD", "Crohn病", "克隆病"},

    # ========== 神经系统 ==========
    {"脑梗死", "脑梗", "缺血性脑卒中", "cerebral infarction", "脑血栓"},
    {"脑出血", "cerebral hemorrhage", "出血性脑卒中", "脑溢血"},
    {"脑卒中", "中风", "stroke", "CVA"},
    {"帕金森病", "PD", "Parkinson病", "帕金森"},
    {"阿尔茨海默病", "AD", "Alzheimer病", "老年痴呆", "老年性痴呆"},
    {"癫痫", "epilepsy", "羊癫疯"},
    {"偏头痛", "migraine", "血管性头痛"},
    {"三叉神经痛", "trigeminal neuralgia"},

    # ========== 肾脏 ==========
    {"慢性肾脏病", "CKD", "慢性肾病", "慢性肾功能不全"},
    {"急性肾损伤", "AKI", "急性肾衰竭", "ARF"},
    {"肾小球肾炎", "GN", "glomerulonephritis"},
    {"尿路感染", "UTI", "泌尿系感染", "尿路感染症"},
    {"肾结石", "kidney stone", "泌尿系结石"},
    {"肌酐", "Cr", "creatinine", "血肌酐"},
    {"尿素氮", "BUN", "blood urea nitrogen"},
    {"肾小球滤过率", "GFR", "eGFR", "估算肾小球滤过率"},

    # ========== 骨科/风湿 ==========
    {"骨质疏松症", "骨质疏松", "osteoporosis", "OP"},
    {"类风湿关节炎", "RA", "类风湿", "rheumatoid arthritis"},
    {"骨关节炎", "OA", "osteoarthritis", "退行性关节炎"},
    {"系统性红斑狼疮", "SLE", "红斑狼疮", "lupus"},
    {"痛风", "gout", "高尿酸血症相关关节炎"},
    {"强直性脊柱炎", "AS", "ankylosing spondylitis"},

    # ========== 肿瘤 ==========
    {"肺癌", "lung cancer", "支气管肺癌"},
    {"肝癌", "HCC", "肝细胞癌", "hepatocellular carcinoma"},
    {"胃癌", "gastric cancer", "胃恶性肿瘤"},
    {"结直肠癌", "CRC", "大肠癌", "colorectal cancer"},
    {"乳腺癌", "breast cancer"},
    {"甲状腺癌", "thyroid cancer"},

    # ========== 感染 ==========
    {"乙型肝炎", "乙肝", "HBV感染", "慢性乙型肝炎"},
    {"丙型肝炎", "丙肝", "HCV感染"},
    {"艾滋病", "AIDS", "HIV感染", "获得性免疫缺陷综合征"},
    {"败血症", "sepsis", "脓毒症", "脓毒血症"},

    # ========== 精神科 ==========
    {"抑郁症", "MDD", "抑郁障碍", "重度抑郁", "depression"},
    {"焦虑症", "GAD", "广泛性焦虑障碍", "焦虑障碍"},
    {"双相情感障碍", "双相障碍", "bipolar disorder", "躁郁症"},
    {"精神分裂症", "schizophrenia"},

    # ========== 妇产科 ==========
    {"子宫肌瘤", "uterine fibroid", "子宫平滑肌瘤"},
    {"多囊卵巢综合征", "PCOS", "多囊卵巢"},
    {"子宫内膜异位症", "endometriosis", "内异症"},
    {"妊娠期糖尿病", "GDM", "孕期糖尿病"},
    {"妊娠期高血压", "PIH", "妊高征"},

    # ========== 儿科 ==========
    {"手足口病", "HFMD", "hand foot and mouth disease"},
    {"川崎病", "Kawasaki病", "皮肤黏膜淋巴结综合征"},

    # ========== 常见症状 ==========
    {"发热", "发烧", "体温升高", "fever"},
    {"咳嗽", "cough"},
    {"头痛", "头疼", "headache"},
    {"胸痛", "胸部疼痛", "chest pain"},
    {"腹痛", "肚子痛", "肚子疼", "腹部疼痛", "abdominal pain"},
    {"恶心", "nausea", "想吐"},
    {"呕吐", "vomiting"},
    {"腹泻", "拉肚子", "diarrhea"},
    {"便秘", "constipation"},
    {"呼吸困难", "气短", "气促", "dyspnea", "喘不上气"},
    {"水肿", "浮肿", "edema"},
    {"眩晕", "头晕", "dizziness", "vertigo"},
    {"心悸", "心慌", "palpitation"},
    {"失眠", "insomnia", "睡不着"},
    {"乏力", "疲劳", "fatigue", "无力"},

    # ========== 常见检查 ==========
    {"血常规", "CBC", "complete blood count", "全血细胞计数"},
    {"血红蛋白", "Hb", "HGB", "hemoglobin"},
    {"白细胞", "WBC", "白细胞计数"},
    {"血小板", "PLT", "platelet"},
    {"C反应蛋白", "CRP", "C-reactive protein"},
    {"降钙素原", "PCT", "procalcitonin"},
    {"丙氨酸氨基转移酶", "ALT", "GPT", "谷丙转氨酶"},
    {"天冬氨酸氨基转移酶", "AST", "GOT", "谷草转氨酶"},
    {"心电图", "ECG", "EKG", "electrocardiogram"},
    {"超声心动图", "心脏彩超", "echocardiography", "UCG"},
    {"CT扫描", "CT", "电子计算机断层扫描", "computed tomography"},
    {"磁共振成像", "MRI", "核磁共振", "核磁"},
    {"胸部X线", "胸片", "chest X-ray", "CXR"},

    # ========== 常见药物 ==========
    {"二甲双胍", "metformin", "格华止"},
    {"阿司匹林", "aspirin", "拜阿司匹林", "ASA"},
    {"硝苯地平", "nifedipine", "心痛定", "拜新同"},
    {"氨氯地平", "amlodipine", "络活喜"},
    {"阿托伐他汀", "atorvastatin", "立普妥"},
    {"瑞舒伐他汀", "rosuvastatin", "可定"},
    {"奥美拉唑", "omeprazole", "洛赛克"},
    {"布洛芬", "ibuprofen", "芬必得"},
    {"对乙酰氨基酚", "扑热息痛", "acetaminophen", "paracetamol"},
    {"头孢类抗生素", "头孢", "cephalosporin"},
    {"阿莫西林", "amoxicillin"},
    {"左氧氟沙星", "levofloxacin", "可乐必妥"},
    {"氯吡格雷", "clopidogrel", "波立维"},
    {"华法林", "warfarin"},
    {"胰岛素", "insulin"},
    {"地塞米松", "dexamethasone"},
    {"泼尼松", "prednisone", "强的松"},
]

# ─────────────────────────────────────────────
# 构建查找表
# ─────────────────────────────────────────────

# 词 → 规范形式（取组内按字典序排列的第一个中文词，若无中文则取最长词）
_CANONICAL = {}


def _pick_canonical(group: Set[str]) -> str:
    chinese = [t for t in group if re.search(r'[\u4e00-\u9fff]', t)]
    if chinese:
        return sorted(chinese, key=len, reverse=True)[0]
    return sorted(group, key=len, reverse=True)[0]


for _group in SYNONYM_GROUPS:
    _canonical = _pick_canonical(_group)
    for _term in _group:
        _CANONICAL[_term.lower()] = _canonical

# 按长度降序排列，确保长词优先匹配（避免 "2型糖尿病" 被 "糖尿病" 先匹配）
_SORTED_TERMS: List[Tuple[str, str]] = sorted(
    _CANONICAL.items(), key=lambda x: -len(x[0])
)


def normalize_medical_text(text: str) -> str:
    """
    将文本中的医学同义词统一为规范形式。

    策略：按长度降序匹配，避免短词误覆盖长词。
    例如 "2型糖尿病" 优先于 "糖尿病" 匹配。
    """
    if not text:
        return text
    result = text.lower()
    for term, canonical in _SORTED_TERMS:
        if term in result and term != canonical.lower():
            result = result.replace(term, canonical)
    return result


def get_synonym_group(term: str) -> Set[str]:
    """查找某个词所在的同义词组"""
    lower = term.lower()
    canonical = _CANONICAL.get(lower)
    if canonical is None:
        return set()
    return {t for t, c in _CANONICAL.items() if c == canonical}


def are_synonyms(term_a: str, term_b: str) -> bool:
    """判断两个词是否为同义词"""
    a_canonical = _CANONICAL.get(term_a.lower())
    b_canonical = _CANONICAL.get(term_b.lower())
    if a_canonical is None or b_canonical is None:
        return False
    return a_canonical == b_canonical
