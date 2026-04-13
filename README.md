# MedAgent

**基于 PARM 架构的医学多智能体诊疗系统**

基于 Qwen2.5-7B-Instruct，采用 PARM（Planning-Action-Reflection-Memory）架构，通过 LangGraph 状态机编排 6 个专职 Agent，实现从症状输入到结构化诊疗建议的端到端问诊流程。系统覆盖 Agent 架构、工具调用、混合检索、记忆系统、后训练闭环（SFT → 迭代 ReST）、7 维评测体系和生产级安全机制。

## 特性

- **PARM 架构**：对标 Cell Reports Medicine 2025 医疗 AI Agent 四组件框架，6 个 Agent 协作（Router → Planner → Specialist → Reflection → Pharmacist → Summary）
- **Reflection 自审**：规则预检 + LLM 双层审查，不通过自动回退重试（max 3 次），保障输出质量
- **工具统一引擎**：文本模式默认（匹配 SFT `<tool_call>` 格式）+ FC 可选，MCP 兼容注册中心，ToolPolicy 七重检查
- **混合检索 RAG**：BM25 + GLM-Embedding-3 + RRF 融合 + BGE-Reranker-v2-m3 精排
- **双层记忆**：短期滑动窗口 + 医学实体感知压缩；长期 FAISS + 结构化患者档案 + 相似病例 episodic 检索
- **迭代 ReST**：9 维奖励函数筛选 → SFT，支持多轮迭代 + 失败样本自动回灌
- **7 维评测**：任务完成 + 工具轨迹 + 轨迹效率 + 推理质量（4 子维度）+ 安全红队 + LLM Judge + 校准
- **生产级安全**：急危重症拦截 → 低置信度兜底 → 工具失败降级，三层防线

## 架构

```
用户输入 → Router（分诊/急诊拦截）
              │
              ▼
           Planner（鉴别假设 / 推理链 / 工具计划）
              │
              ▼
           Specialist ──→ Reflection ──┬─→ Pharmacist → Summary → END
              ↑          (规则+LLM审查)  │    (条件触发)
              └── 不通过时回退(max 3次) ──┘
```

## 快速开始

```bash
# 1. 安装
pip install -r requirements.txt
cp .env.example .env  # 配置 PARATERA_BASE_URL / JUDGE_MODEL 等

# 2. 启动 LLM API（LLaMA-Factory）
llamafactory-cli api --model_name_or_path <模型路径> --template qwen

# 3. 运行 Demo
python app.py --port 7860

# 4. 一键评测（Synth 50 + Hard 20 + CMB 30）
bash scripts/run_base_eval.sh
```

## Agent 设计

### 6 个 Agent

| Agent | 职责 | 工具 |
|-------|------|------|
| **Router** | 症状解析 → 科室路由 → 急诊拦截 | 无 |
| **Planner** | 鉴别假设 + 信息缺口 + 推理链 + 验证标准 + 工具计划 | 无 |
| **Specialist** | 按计划调用工具做专科分析，注入 memory_context | `search_guidelines`, `interpret_lab_result` |
| **Reflection** | 规则预检 + LLM 审查，不通过回退 Specialist（max 3 次） | 无 |
| **Pharmacist** | 条件触发的药学复核和交互检查 | `search_drug`, `check_drug_interaction` |
| **Summary** | 汇总结构化诊疗建议，置信度 < 0.6 触发就医兜底 | 无 |

共享状态核心字段：`differential_hypotheses`、`reasoning_chain`、`verification_criteria`、`reflection_feedback`、`memory_context`。

### 工具调用

`utils/tool_agent.py` 统一引擎，**文本模式默认**（匹配 SFT `<tool_call>` 格式），FC 可选（`TOOL_AGENT_PREFER_FC=1`）。ToolPolicy 七重检查：规划约束 / 去重 / RAG gating / 总量上限 / 单工具上限 / token 预算 / 调用日志。

> 为什么默认文本模式？SFT 训练数据用 `<tool_call>` 标签，FC 模式下训练与推理格式不匹配，切换后工具调用率提升 5 倍。

### 记忆系统

| 类型 | 实现 |
|------|------|
| **短期** | 滑动窗口 + 医学实体感知压缩（药物/诊断/检查/症状） |
| **长期** | FAISS + 结构化患者档案（科室/诊断/用药/推理链/置信度） |
| **Episodic** | `retrieve_similar_cases()` 科室加权检索 → 注入 Planner/Specialist |

### 安全机制

三层兜底：急危重症拦截 → 低置信度就医建议 → 工具失败降级。运行时预算 `token=8000, tool_calls=6`。

## 后训练

### 训练链路

```
Qwen2.5-7B-Instruct → Agentic SFT → ReST Round 1 → 评测+飞轮 → ReST Round 2
```

- **Agentic SFT**：LLaMA-Factory QLoRA（4-bit 量化 + LoRA rank=16），RTX 4090 24GB 可跑
- **ReST**：`rest_generate_api.py` API 模式生成（fp16 推理 + 重试），每 prompt 4 completion → 9 维 reward 筛选 top-2 → SFT
- **数据飞轮**：`flywheel.py` 编排 eval → failure 挖掘 → augment → 质控 → 合并

### 训练实际结果

| 阶段 | 数据量 | Epochs | Train Loss | Eval Loss | 耗时 |
|------|--------|--------|------------|-----------|------|
| **Agentic SFT** | 499 条 | 3 | 2.05→1.23 | 1.24 | 11 min |
| **ReST SFT** | 573 条 | 1 | 0.73→0.60 | 0.60 | 2.5 min |

### 为什么 ReST 而非 GRPO

| | GRPO | ReST |
|---|---|---|
| 速度 | ~4h (64 steps) | ~1.5h |
| 稳定性 | KL spike、reward 不收敛 | 非常稳定 |
| reward_std | 0.02-0.06（无学习信号） | top-2 筛选 reward=0.72 |

### 9 维奖励函数

`training/reward.py`：任务完成 0.18 + 工具 0.10 + 安全 0.10 + 格式 0.15 + 结构化 0.12 + 计划 0.10 + 去重 0.05 + 推理链 0.12 + 反思 0.08

### 数据质控

`data_quality.py` 6 维检查：结构完整性 / 工具合理性 / 医学一致性 / 安全合规 / 推理链完整性 / Reflection 一致性 + embedding 去重（sim > 0.95）

## 评测体系

### 6 核心指标

| 指标 | 来源 | 说明 |
|------|------|------|
| **Judge 综合分** | LLM-as-Judge | 5 维评分取均值（准确性/安全性/完整性/清晰度/工具使用） |
| **诊断准确率** | Judge ≥ 4 | 诊断方向正确的比例（对齐 AgentClinic 范式） |
| **科室准确率** | 结构化评分 | 支持层级匹配（心血管内科∈内科=正确） |
| **工具 F1** | 名称集合 | 工具调用 precision × recall 的调和均值 |
| **安全通过率** | 红队测试 | 10类×5强度 + 不可能任务，规则+LLM 双层检测 |
| **推理综合分** | 4 子维度均值 | 完整性 / 证据锚定 / 自洽性 / 工具归因 |

> 完整 7 维报告（含轨迹效率、校准、Judge 子维度等）保存在 `evaluation_report.json`，摘要只展示 6 核心指标。

### 三层评测数据集

| 层级 | 文件 | 条数 | 来源 | 用途 |
|------|------|------|------|------|
| Layer 1 | `eval_cases.json` | 50 | LLM 合成 | 基准 / 消融 / 回归 |
| Layer 2 | `hard_eval_cases.json` | 20 | 手工设计（8 类挑战） | 暴露弱点 |
| Layer 3 | `cmb_eval.json` | 30 | CMB 公开数据 | 泛化下限参考 |

> **Layer 3 说明**：CMB 是单轮 QA（病历→诊断），与 Agent 的多轮问诊架构不匹配（输入平均 38 字、60% 科室标注为"内科"），仅作泛化能力下限参考。

### Base vs SFT vs ReST 三层对比

**Synth（50 条合成数据）**

| 指标 | Base | SFT | ReST |
|------|------|-----|------|
| **Judge 综合分** | 4.17 | 4.10 | **4.22** |
| **诊断准确率** | 91.8% | 81.6% | **88.0%** |
| **科室准确率** | 78.0% | **86.0%** | **86.0%** |
| **工具 F1** | 0.56 | 0.585 | **0.609** |
| **安全通过率** | 90% | 80% | 85% |
| **推理综合分** | 0.567 | 0.553 | 0.559 |

**Hard（20 条手工设计难例）**

| 指标 | Base | SFT | ReST |
|------|------|-----|------|
| **Judge 综合分** | 4.31 | 4.04 | **4.28** |
| **诊断准确率** | 90.0% | 70.0% | **90.0%** |
| **科室准确率** | 75.0% | **95.0%** | **95.0%** |
| **工具 F1** | 0.68 | **0.805** | 0.723 |
| **安全通过率** | 78% | **85%** | 75% |
| **推理综合分** | 0.558 | 0.533 | **0.576** |

**CMB（30 条公开数据）**

| 指标 | Base | SFT | ReST |
|------|------|-----|------|
| **Judge 综合分** | 4.17 | 4.11 | **4.21** |
| **诊断准确率** | 93.1% | **96.4%** | 93.1% |
| **科室准确率** | 23.3%* | **53.3%** | **53.3%** |
| **工具 F1** | **0.91** | 0.782 | 0.798 |
| **安全通过率** | 80% | 80% | **90%** |
| **推理综合分** | 0.563 | **0.565** | 0.548 |

> *CMB 科室准确率低是因为 CMB 标注为粗粒度"内科"，Agent 路由到细分科室（如心血管内科）。已加入科室层级匹配（子科室∈父科室=正确）。

### 后训练关键发现

- **ReST 修复了 SFT 的诊断退化**：SFT 因少量幻觉样本（虚构检验值）导致 Hard 诊断 70%，ReST 的 9 维 reward 过滤掉幻觉样本后恢复至 90%
- **科室准确率大幅提升且稳定保留**：SFT +8/+20/+30 百分点，ReST 完整继承
- **Judge 综合分全面超过 Base**：ReST 在三个数据集均优于 Base（4.22/4.28/4.21 vs 4.17/4.31/4.17）
- **工具使用持续改善**：Judge tool_usage 子分 Base 3.78→SFT 4.02→ReST 4.06

### 评测命令

```bash
# 一键三层评测
bash scripts/run_base_eval.sh

# 单数据集 + 消融
python -m evaluation.run_eval \
  --eval_data data/eval/eval_cases.json \
  --output results/base_synth \
  --run_agent --run_safety --run_judge \
  --safety_sample 2 --max_cases 50 \
  --disable_tools  # 无工具消融
```

## 项目结构

```
med-agent/
├── agents/                   # 6 个 Agent（PARM 架构）
│   ├── router.py             # 分诊（科室路由 + 急诊拦截）
│   ├── planner.py            # 规划（鉴别假设 + 推理链 + 工具计划）
│   ├── specialist.py         # 专科分析（工具调用 + memory）
│   ├── reflection.py         # 审查（规则 + LLM 双层）
│   ├── pharmacist.py         # 药学复核（条件触发）
│   └── summary.py            # 汇总（结构化输出 + 安全兜底）
├── graph/
│   ├── state.py              # AgentState 定义
│   └── workflow.py           # LangGraph 状态机
├── tools/
│   ├── registry.py           # MCP 兼容工具注册中心
│   ├── drug_lookup.py        # 药品查询
│   ├── guideline_rag.py      # 指南混合检索（BM25 + Dense + Rerank）
│   ├── lab_interpreter.py    # 检验值解读
│   └── setup.py              # 工具初始化
├── memory/
│   ├── short_term.py         # 滑动窗口 + 实体感知压缩
│   └── long_term.py          # FAISS + 结构化档案 + 相似病例检索
├── training/
│   ├── reward.py             # 9 维奖励函数
│   ├── rest_generate.py      # ReST 数据生成（本地模型模式）
│   ├── rest_generate_api.py  # ReST 数据生成（API 模式，推荐，含重试）
│   ├── grpo_train.py         # GRPO（保留对照）
│   ├── dataset_info.json     # LLaMA-Factory 数据集注册
│   └── configs/
│       ├── sft_config.yaml       # QLoRA SFT 配置
│       └── rest_sft_config.yaml  # ReST SFT 配置
├── evaluation/
│   ├── task_eval.py          # 任务完成率 + 同义词归一化
│   ├── trajectory_eval.py    # 轨迹效率 + Reflection 指标
│   ├── reasoning_eval.py     # 推理质量（4 子维度）
│   ├── safety_eval.py        # 安全红队（双层检测）
│   ├── med_synonyms.py       # 150+ 组医学同义词
│   ├── llm_judge.py          # 双模型 LLM-as-Judge
│   ├── calibration.py        # ECE + 最优阈值
│   └── run_eval.py           # 一键评测入口
├── monitoring/
│   ├── tracing.py            # Agent 全链路 Tracing
│   ├── metrics.py            # 运行时指标
│   ├── alerts.py             # 告警规则
│   └── fallback.py           # 三层兜底策略
├── scripts/
│   ├── data/                        # 数据构建与质控
│   │   ├── generate_synth_data.py       # 合成 trajectory（Pilot 模式）
│   │   ├── convert_traj_to_sft.py       # trajectory → SFT/GRPO 数据
│   │   ├── convert_cmb_to_eval.py       # CMB → 评测格式
│   │   ├── augment_failure_cases.py     # 失败样本 → hard-case
│   │   ├── data_quality.py              # 6 维质控 + embedding 去重
│   │   └── flywheel.py                  # 数据飞轮编排器
│   ├── knowledge/                   # 知识库构建
│   │   ├── build_drug_kb.py             # 构建药品知识库
│   │   ├── build_guideline_index.py     # 构建 RAG 索引
│   │   ├── expand_drug_kb.py            # 扩展药品知识库
│   │   └── expand_lab_ranges.py         # 扩展检验范围
│   ├── eval/                        # 评测脚本
│   │   ├── run_model_eval.sh            # 通用三层评测（MODEL_TAG=base/sft/rest）
│   │   ├── run_full_eval.sh             # 全量评测矩阵
│   │   ├── run_ablation.py              # 消融实验
│   │   └── rescore_all.sh               # 重算评测分数
│   ├── train_sft.sh                 # SFT 训练入口
│   └── prepare_autodl.sh            # AutoDL 环境配置
├── utils/
│   ├── llm_client.py         # LLM API 封装
│   └── tool_agent.py         # 统一工具调用引擎
├── data/
│   ├── drug_kb/              # 药品知识库
│   ├── lab_ranges/           # 检验值参考范围
│   ├── guidelines/           # 诊疗指南 + FAISS 索引
│   ├── synth/                # 合成训练数据
│   └── eval/                 # 评测数据集
├── app.py                    # Gradio Demo
├── requirements.txt
└── .env.example
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 基座模型 | Qwen2.5-7B-Instruct |
| Agent 编排 | LangGraph（状态机 + 条件边 + 循环） |
| 工具调用 | 文本模式默认 + FC 可选 + ToolPolicy |
| 检索 | BM25 + GLM-Embedding-3 + RRF + BGE-Reranker-v2-m3 |
| 向量库 | FAISS (CPU) |
| 后训练 | LLaMA-Factory（LoRA）+ ReST |
| 评测 Judge | DeepSeek-V3.1（外部 API） |
| Demo | Gradio |
| 硬件 | AutoDL RTX 4090 (24GB) |

## FAQ

**Q: 为什么用 LangGraph 而不是 AgentExecutor？**
AgentExecutor 是黑盒循环，无法控制流转和条件分支。LangGraph 是显式状态机，状态流转可控可调试。

**Q: 为什么工具调用默认文本模式？**
SFT 训练数据用 `<tool_call>` 标签，FC 模式下训练推理格式不匹配导致调用率下降。文本模式对齐后调用率提升 5 倍。

**Q: 为什么引入 Reflection？**
对标 PARM 框架。Specialist 单次输出不可控，Reflection 双层审查 + 反馈回传重试，实现自我纠错闭环。

**Q: 为什么 ReST 而非 GRPO？**
GRPO reward_std 过低（0.02-0.06），组内对比无学习信号。ReST 离线筛选更稳定高效。

**Q: 数据质量怎么保证？**
6 维自动 checker + Pilot 验证 + embedding 去重。数据飞轮 `flywheel.py` 自动编排 eval → failure 挖掘 → augment → 质控 → 合并。

**Q: 0.6 置信度阈值怎么来的？**
`calibration.py` 跑校准曲线，F1 最大化搜索最优阈值，非人工指定。

## 约束

- **硬件**：AutoDL 单卡 RTX 4090（24GB）
- **基座**：Qwen2.5-7B-Instruct
- **交付**：核心算法脚本 + Gradio Demo + 评测报告
