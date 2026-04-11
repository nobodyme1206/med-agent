# MedAgent — 医学多智能体诊疗系统

基于 **PARM 架构（Planning-Action-Reflection-Memory）+ LangGraph + OpenAI Function Calling + MCP Tool Registry + ReST 强化自训练**，构建端到端可上线、可迭代、可控成本的医学多 Agent 诊疗系统。覆盖 Agent 架构设计、数据飞轮、后训练闭环（SFT → 迭代 ReST）、多维评测体系和生产级安全交付。

---

## 系统架构

```
                            ┌──────────────────────────────────────────────────┐
                            │        MedAgent 系统架构（PARM 框架）              │
                            └──────────────────────────────────────────────────┘

  用户输入（症状描述）
         │
         ▼
  ┌──────────────┐
  │ Router Agent │  意图识别 / 科室路由 / 急诊拦截
  └──────┬───────┘
         │ 正常路径                               ┌──────────────────────────┐
         ▼                                        │    PARM 四组件映射        │
  ┌──────────────┐                                │                          │
  │Planner Agent │  推理规划 / 鉴别假设 / 工具计划  │  Planning  → Planner     │
  └──────┬───────┘                                │  Action    → Specialist  │
         ▼                                        │  Reflection→ Reflection  │
  ┌──────────────────────────────────────────────┐│  Memory    → Long/Short  │
  │           LangGraph 状态机编排                 │└──────────────────────────┘
  │                                              │
  │  Router ─┬─▶ Summary（急诊直达/升级兜底）      │
  │          └─▶ Planner ─▶ Specialist            │
  │                            │                  │
  │                            ▼                  │
  │                       Reflection              │
  │                     ┌─── │ ───┐               │
  │                     │  通过?   │               │
  │                     │         │               │
  │                  ✓  ▼      ✗  ▼               │
  │              Pharmacist  Specialist(重试)      │
  │              (条件触发)   (max 3次)             │
  │                  │                            │
  │                  ▼                            │
  │               Summary → END                   │
  └──────────────────────────────────────────────┘
         │
         ├─────────────────────────────┬─────────────────────────────┐
         ▼                             ▼                             ▼
  ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
  │  工具层 (MCP) │              │   记忆系统    │              │ 安全与兜底    │
  ├──────────────┤              ├──────────────┤              ├──────────────┤
  │ 指南检索 RAG  │              │ 短期：滑窗+摘要│              │ 急危重症拦截  │
  │ 检验值解读    │              │   + 实体感知压缩│              │ 低置信度兜底  │
  │ 药品与交互    │              │ 长期：FAISS   │              │ 工具失败降级  │
  │ ToolPolicy   │              │   + 结构化档案 │              │ Budget 超限停 │
  │              │              │   + 相似病例检索│              │              │
  └──────────────┘              └──────────────┘              └──────────────┘
         │                                                          │
         ▼                                                          ▼
  结构化输出 + 最终回复（诊断方向 / 检查 / 用药 / 随访 / 工具轨迹）  Tracing / Metrics / Alerts
```

---

## 核心技术亮点

| 维度 | 实现 | 应对面试追问 |
|------|------|-------------|
| **Agent 编排** | LangGraph 状态机 + `Router → Planner → Specialist → Reflection → Pharmacist(条件) → Summary`（PARM 架构） | 6 节点有向图，Reflection 自审机制保障输出质量 |
| **PARM 架构** | Planning（Planner：鉴别假设+信息缺口+推理链+验证标准）→ Action（Specialist）→ Reflection（规则+LLM 双层审查）→ Memory（结构化档案+相似病例检索） | 对标 Cell Reports Medicine 2025 医疗 AI Agent 四组件框架 |
| **工具调用** | 统一引擎：OpenAI Function Calling 优先，文本解析兜底 + ToolPolicy 七重检查 | 规划约束、去重、RAG gating、总量/单工具 budget、token budget |
| **工具注册** | MCP 兼容注册中心，JSON Schema 标准化，统一调用日志 | 工具扩展只需注册，无需改 Agent 代码 |
| **混合检索** | BM25 + GLM-Embedding-3 + RRF 融合 + BGE-Reranker-v2-m3 | 稀疏+稠密互补，Reranker 精排 |
| **记忆系统** | 短期：滑动窗口 + 医学实体感知压缩；长期：FAISS + 结构化患者档案 + 相似病例 episodic 检索 | memory_context 注入 Planner/Specialist prompt |
| **后训练** | Agentic SFT + Stage-wise SFT + 迭代 ReST（多轮 Generate→Filter→SFT）+ hard-case 回灌 | `--rest_round` 支持多轮迭代，失败样本自动回灌 |
| **奖励函数** | 9 维组合：任务0.18 + 工具0.10 + 安全0.10 + 格式0.15 + 结构化0.12 + 计划0.10 + 去重0.05 + **推理链0.12** + **反思质量0.08** | PARM 对齐：推理链评估鉴别诊断/证据引用/步骤数，反思评估自检/修正/不确定性 |
| **数据质控** | 6 维自动 checker（+推理链完整性 +反馈一致性）+ embedding 去重 + Pilot 验证 | 推理链与诊断关联性检查，Reflection 反馈采纳检查 |
| **评测体系** | 7 维：任务完成 + 工具轨迹 + 轨迹效率 + **推理质量**（4 子维度）+ 安全红队 + 双模型 Judge + 校准 | 推理完整性/证据锚定率/自洽性/工具归因 |
| **安全评测** | 10类×5强度=50 条红队 + 5 条不可能任务，双层检测（关键词+LLM Judge），剂量-反应曲线 | 从温和到极端 5 级强度梯度，量化安全衰减 |
| **同义词归一化** | 150+ 组医学同义词（诊断/检查/药品/症状），接入评测的文本匹配和列表召回 | 解决"高血压≈高血压病"等同义问题 |
| **可观测性** | Agent Tracing + 运行时指标 + 告警规则 | 每个节点输入/输出/延迟/停止原因全链路追踪 |

---

## 项目结构

```
med-agent/
├── agents/                          # 6 个 Agent（PARM 架构）
│   ├── router.py                    # 分诊 Agent（意图识别 + 科室路由 + 急诊拦截）
│   ├── planner.py                   # 规划 Agent（鉴别假设 + 信息缺口 + 推理链 + 验证标准 + 工具计划）
│   ├── specialist.py                # 专科 Agent（按计划调用工具 + memory_context 辅助）
│   ├── reflection.py                # 反思 Agent（规则预检 + LLM 审查，不通过回退 Specialist）
│   ├── pharmacist.py                # 药师 Agent（条件触发的药学复核）
│   └── summary.py                   # 汇总 Agent（结构化输出 + 安全兜底 + 最终回复）
├── graph/
│   ├── state.py                     # AgentState（PARM 字段 + memory_context + reflection 字段）
│   └── workflow.py                  # LangGraph 6 节点状态机 + Reflection 条件边 + Memory 集成
├── tools/
│   ├── registry.py                  # MCP 兼容工具注册中心（JSON Schema）
│   ├── drug_lookup.py               # 工具：药品知识库查询
│   ├── guideline_rag.py             # 工具：诊疗指南混合检索
│   ├── lab_interpreter.py           # 工具：检验值解读
│   └── setup.py                     # 工具统一初始化
├── memory/
│   ├── short_term.py                # 短期记忆：滑动窗口 + 医学实体感知压缩 + get_key_entities
│   └── long_term.py                 # 长期记忆：FAISS + 结构化患者档案 + 相似病例 episodic 检索
├── training/
│   ├── reward.py                    # 9 维组合奖励（+推理链质量 +反思质量）
│   ├── rest_generate.py             # 迭代 ReST 数据生成（--rest_round 多轮合并）
│   ├── grpo_train.py                # GRPO 训练脚本（保留对照）
│   └── configs/
│       ├── sft_config.yaml          # Agentic SFT 配置（LLaMA-Factory 格式）
│       └── rest_sft_config.yaml     # ReST SFT 配置
├── evaluation/
│   ├── task_eval.py                 # 任务完成率（接入医学同义词归一化）
│   ├── trajectory_eval.py           # 轨迹效率 + reflection 指标
│   ├── reasoning_eval.py            # 推理质量评测（完整性 + 证据锚定 + 自洽性 + 工具归因）
│   ├── safety_eval.py               # 安全红队（10类×5强度 + 不可能任务，双层检测）
│   ├── med_synonyms.py              # 150+ 组医学同义词归一化
│   ├── llm_judge.py                 # 双模型 LLM-as-Judge + 一致性分析
│   ├── calibration.py               # 置信度校准（ECE + 最优阈值搜索）
│   └── run_eval.py                  # 一键评测入口（7 维 + 消融 + failure analysis）
├── monitoring/
│   ├── tracing.py                   # Agent 调用链 Tracing（Span/Trace 全链路）
│   ├── metrics.py                   # 运行时指标（P50/P99 延迟、token、成功率）
│   ├── alerts.py                    # 告警规则（错误率/延迟超阈值）
│   └── fallback.py                  # 三层兜底策略
├── scripts/
│   ├── generate_synth_data.py       # 合成 trajectory（含 Pilot 模式 + 质控集成）
│   ├── convert_traj_to_sft.py       # trajectory → SFT/GRPO 数据（PARM 标签嵌入）
│   ├── augment_failure_cases.py     # failure_cases → hard-case 数据（PARM 字段增强）
│   ├── data_quality.py              # 6 维质控管线（+推理链 +反馈一致性）+ embedding 去重
│   ├── flywheel.py                  # 数据飞轮编排器（eval→挖掘→增补→质控→合并）
│   ├── run_ablation.py              # 消融实验脚本
│   ├── run_full_eval.sh             # 全量评测 + failure augmentation
│   ├── build_drug_kb.py             # 构建药品知识库
│   └── build_guideline_index.py     # 构建诊疗指南 RAG 索引
├── utils/
│   ├── llm_client.py                # LLM API 统一封装（chat / FC / embed / JSON mode）
│   └── tool_agent.py                # 统一工具调用引擎（FC 优先 + 文本 fallback + ToolPolicy）
├── docs/
│   └── deep_comparison_cc_haha.md   # MedAgent × Claude Code 深度对比分析
├── data/
│   ├── drug_kb/                     # 药品知识库
│   ├── lab_ranges/                  # 检验值参考范围
│   ├── guidelines/                  # 诊疗指南 + FAISS 索引
│   ├── synth/                       # 合成训练数据
│   └── eval/                        # 评测数据集
├── app.py                           # Gradio Demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## Agent 设计

### 共享状态（AgentState）

所有 Agent 通过 LangGraph 的 `AgentState`（TypedDict）通信，核心字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `List[Dict]` | 完整对话历史（自动追加） |
| `patient_info` / `current_department` / `router_reasoning` | `str` | Router 提取的患者摘要、分诊科室和路由理由 |
| `problem_type` / `tool_plan` / `expected_evidence` / `plan_summary` / `need_pharmacist` | `str / List / bool` | Planner 生成的执行计划和路由决策 |
| **`differential_hypotheses`** | `List[str]` | **PARM**：Planner 输出的鉴别诊断假设列表 |
| **`information_gaps`** | `List[str]` | **PARM**：待解决的信息缺口 |
| **`reasoning_chain`** | `str` | **PARM**：推理链（从症状到诊断方向的逻辑链） |
| **`verification_criteria`** | `List[str]` | **PARM**：Reflection 用于验证 Specialist 输出的标准 |
| **`reflection_feedback`** | `str` | **PARM**：Reflection 不通过时的反馈意见 |
| **`reflection_passed`** / **`reflection_count`** | `bool / int` | **PARM**：审查通过状态和重试次数 |
| **`memory_context`** | `str` | **PARM**：从长期记忆检索的相似病例上下文 |
| `specialist_analysis` / `drug_advice` | `str` | 专科与药学分析结果 |
| `tool_calls` | `List[ToolCallRecord]` | 工具调用记录（含 `tool_name`, `input_args`, `success`, `call_signature`, `skipped_reason`） |
| `retrieved_knowledge` | `List[Dict]` | RAG 检索结果 |
| `structured_output` | `StructuredOutput` | 结构化最终结果 |
| `confidence` / `should_escalate` / `escalate_reason` | `float / bool / str` | 安全兜底和升级控制 |
| `loop_count` / `stop_reason` / `token_usage` | `int / str / int` | 运行时循环次数、停止原因和 token 消耗 |
| `final_response` | `str` | 最终输出给用户的回复 |

### 六个 Agent（PARM 架构）

| Agent | 职责 | 可用工具 | 推理模式 |
|-------|------|---------|--------|
| **Router** | 解析症状 → 判断科室 → 急诊拦截 → 提取患者信息 | 无 | Few-shot + JSON mode |
| **Planner** | 生成鉴别假设、信息缺口、推理链、验证标准、工具计划；注入 `memory_context` | 无 | JSON 规划 + 启发式 fallback |
| **Specialist** | 按计划做专科分析，引用 `memory_context` 辅助；逐一分析鉴别假设 | `search_guidelines`, `interpret_lab_result` | 统一引擎（FC 优先） |
| **Reflection** | 规则预检（覆盖率/证据/安全）+ LLM 审查；不通过回退 Specialist（max 3 次） | 无 | 规则 + LLM 双层 |
| **Pharmacist** | 条件触发的药学复核、交互检查和用药建议 | `search_drug`, `check_drug_interaction`, `search_by_indication` | 统一引擎（FC 优先） |
| **Summary** | 汇总所有 Agent 输出 + 推理链 + 审查信息，产出 `structured_output` 与 `final_response` | 无 | JSON mode + Guardrails |

### LangGraph 状态机

```
START → Router ─┬─→ Summary（急诊/升级直达）
                │
                └─→ Planner ──→ Specialist ──→ Reflection ─┬─→ Pharmacist（通过 + need_pharmacist）
                                    ↑                      ├─→ Summary（通过 + 无需药师）
                                    └── 不通过时回退 ────────┘   （reflection_count < 3）

条件分支：
  • Router 检测急危重症关键词 → `should_escalate = True` → 直接进入 Summary
  • Planner 输出 differential_hypotheses / tool_plan / verification_criteria，约束后续执行
  • Specialist 按计划调用工具，注入 memory_context 和 reflection_feedback
  • Reflection 双层审查：规则预检（覆盖率/证据/安全）→ LLM 审查（验证标准逐项检查）
    - 通过 → 进入 Pharmacist 或 Summary
    - 不通过 → reflection_feedback 写入 state，回退 Specialist 重试（max 3 次）
  • ToolPolicy：规划约束 + 去重 + `max_tool_calls = 6` + `max_calls_per_tool = 2`
  • Token 预算 `8000`，超限时记录 `stop_reason` 并提前停止
  • Summary 输出 `structured_output` 与 `final_response`，置信度 < 0.6 时触发就医兜底
                                   Pharmacist ──→ Summary → END
```

每个 Agent 节点自动集成 **Tracing**（`monitoring/tracing.py`），记录输入/输出/延迟/错误。LangGraph 不可用时，`workflow.py` 还提供顺序 fallback。

---

## 工具层

### 统一工具调用引擎（参考 Claude Code 架构）

`utils/tool_agent.py` 是所有 Agent 共用的工具调用引擎，参考了 Claude Code 的原生 `tool_use` 模式：

```
Agent（Specialist / Pharmacist）
    │
    ▼
run_tool_agent()          ← 统一入口
    │
    ├─▶ FC 模式（优先）     ← chat_with_messages(tools=schemas)
    │   API 返回结构化 tool_calls → 执行 → 结果反馈 → 继续推理
    │
    └─▶ 文本解析（fallback） ← 当 FC 不可用时自动切换
        解析 <tool_call> 标签 → 执行 → 结果拼入 prompt → 继续推理
```

**设计要点**：
- 工具 JSON Schema **从 `ToolRegistry` 自动获取**，Agent 只需声明工具名列表
- `Planner` 产出的 `tool_plan` 会作为 allowlist，约束 Specialist / Pharmacist 的可用工具
- `ToolPolicy` 统一处理：规划约束、RAG gating、重复调用抑制、总量上限、单工具上限
- FC 模式和文本 fallback 都会记录 `tool_name`, `input_args`, `success`, `call_signature`, `skipped_reason`
- 工具结果会被压缩摘要再回灌上下文，减少 token 消耗；超出预算时写入 `stop_reason`
- 消融实验可通过 `tools_enabled` / `rag_enabled` / `max_tool_calls` / `max_calls_per_tool` 开关控制

### 工具注册中心（MCP 兼容）

`tools/registry.py` 实现统一的工具注册中心，兼容 **OpenAI Function Calling** 和 **MCP** 两种格式，支持按名称/分类过滤：

| 工具 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `search_guidelines` | 诊疗指南混合检索 | 查询文本 | Top-K 相关知识块（BM25 + Dense + Rerank） |
| `interpret_lab_result` | 检验值解读 | 检验项 + 数值 | 正常/偏高/偏低 + 临床意义 |
| `search_drug` | 药品信息查询 | 药品名 | 适应症、禁忌、不良反应、用法用量 |
| `check_drug_interaction` | 药物交互检查 | 两种药名 | 交互严重程度 + 用药建议 |
| `search_by_indication` | 按适应症查药 | 适应症名 | 可用药品列表 |

每次工具调用自动记录：`tool_name`, `input_args`, `output`, `latency_ms`, `success`。

---

## 记忆系统（Memory — PARM 第四组件）

| 类型 | 实现 | 用途 |
|------|------|------|
| **短期记忆** | 滑动窗口（最近 10 轮）+ **医学实体感知压缩**（自动提取药物/诊断/检查/症状实体，压缩时保留关键实体） | 当前会话上下文，`get_key_entities()` 暴露实体集 |
| **长期记忆** | FAISS 向量存储 + **结构化患者档案**（`store_structured_profile`：科室/主诉/诊断/鉴别/推理链/用药/检查/置信度） | 跨会话档案管理 |
| **Episodic 检索** | `retrieve_similar_cases(chief_complaint, department, top_k, min_score)` → 科室加权 + 最小分数过滤 | 相似病例辅助决策 |
| **Memory 集成** | `run_consultation` 优先调用 `retrieve_similar_cases`，回退 `retrieve`；结果注入 `memory_context` → Planner/Specialist prompt | 全链路 Memory-Agent 耦合 |

---

## 数据生成与训练数据构建

`scripts/generate_synth_data.py` 用 LLM API 生成多轮问诊对话，每条 trajectory 会尽量保留可训练、可评测、可复盘的关键信号：

```json
{
  "case_id": "001",
  "patient_profile": "45岁男性，高血压病史3年",
  "chief_complaint": "头晕、视物模糊2天",
  "department": "心血管内科",
  "dialogue": [
    {"role": "patient", "content": "医生我最近总头晕..."},
    {"role": "agent", "thought": "患者高血压+头晕，需检查血压控制...",
     "tool_calls": [{"name": "search_guidelines", "args": {"query": "高血压头晕鉴别"}}],
     "response": "请问您最近血压测量值是多少？"}
  ],
  "final_diagnosis_direction": "高血压2级，降压药物调整",
  "preferred_tool_sequence": ["search_guidelines", "search_drug"],
  "plan_summary": "先检索指南，再补充药学分析",
  "structured_output": {
    "department": "心血管内科",
    "diagnosis_direction": "高血压2级，需评估降压方案",
    "recommended_tests": ["动态血压", "肾功能"],
    "medication_advice": ["遵医嘱评估是否调整降压药"],
    "need_followup": true,
    "followup_actions": ["线下复诊"],
    "evidence_summary": ["高血压病史", "头晕伴降压效果不佳"],
    "used_tools": ["search_guidelines", "search_drug"],
    "tool_plan": ["search_guidelines", "search_drug"],
    "final_response": "初步考虑高血压控制不佳，建议进一步评估。"
  }
}
```

**Pilot 模式**：`--pilot` 先生成 10 条 → 自动质控 → 人工确认 → 再批量生成：

```bash
# Step 1: 小批量验证
python scripts/generate_synth_data.py --pilot --pilot_size 10

# Step 2: 确认无误后批量生成（自动集成质控过滤）
python scripts/generate_synth_data.py --num_cases 500
```

### 6 维数据质控管线

`scripts/data_quality.py` 对每条合成 trajectory 做 6 维自动检查：

| 维度 | 检查内容 | 示例 |
|------|---------|------|
| **结构完整性** | 必填字段非空、对话 ≥ 2 轮、角色交替 | dialogue 缺 agent 回复 → 丢弃 |
| **工具调用合理性** | 工具名在注册表中、参数为合法 dict | 调用不存在的工具 → 丢弃 |
| **医学术语一致性** | 科室-主诉逻辑匹配、诊断方向非空 | 骨科 case 出现"血糖" → 标记 |
| **安全合规** | 无确定性诊断、包含就医建议 | "你得了糖尿病" → 丢弃 |
| **推理链完整性** | 推理链存在且 >10 字、有鉴别诊断假设、推理链与诊断方向关联 | 推理链为空 → 标记 |
| **Reflection 一致性** | Reflection 反馈被采纳到最终回复、重试次数 ≤ 2 | 反馈未被采纳 → 标记 |

去重：MD5 精确去重 + embedding 余弦相似度去近似（sim > 0.95 → 去重）。

```bash
python scripts/data_quality.py \
  --input data/synth/trajectories/all_trajectories.json \
  --output data/synth/trajectories/filtered.json \
  --use_embedding --sim_threshold 0.95
```

### 训练数据转换与 hard-case 回灌

`scripts/convert_traj_to_sft.py` 会从 trajectory 中导出多视图训练数据：

- `agent_sft.json`：主 ShareGPT SFT 数据（保留 `<think> / <tool_call> / <structured_output> / <response>`）
- `grpo_prompts.json`：带 `expected_tools / expected_first_tool / tool_plan / structured_output_target` 的 prompt 数据
- `router_sft.json` / `planner_sft.json` / `summary_sft.json`：分阶段监督数据

评测结束后，`evaluation/run_eval.py` 会导出 `failure_cases.json`；随后可用 `scripts/augment_failure_cases.py` 继续构造 hard-case 数据：

- `hard_case_sft.json`
- `hard_case_grpo.json`
- `hard_case_router_sft.json`
- `hard_case_planner_sft.json`
- `hard_case_summary_sft.json`

```bash
# trajectory → 主 SFT / GRPO / Router-Planner-Summary stage data
python scripts/convert_traj_to_sft.py \
  --input data/synth/trajectories/all_trajectories.json \
  --sft_output data/synth/sft_data/agent_sft.json \
  --grpo_output data/synth/sft_data/grpo_prompts.json

# failure_cases → hard-case 训练数据
python scripts/augment_failure_cases.py \
  --failure_cases results/baseline_clean/failure_cases.json \
  --output_dir data/synth/sft_data/failure_augmented
```

---

## 后训练

### 训练迭代链路

```
Qwen2.5-7B-Instruct（基座）
        ↓
Agentic SFT（学会 ReAct 格式 + 结构化输出 + 工具调用 + PARM 标签）
        ↓
ReST Round 1（9 维 reward 筛选 → SFT 强化）
        ↓
评测（7 维）→ flywheel.py 编排 → failure 挖掘 → 增补 → 质控 → 合并
        ↓
ReST Round 2（--rest_round 2 合并上轮数据 → 迭代优化）
        ↓
（可选）继续迭代 / 消融实验
```

### Agentic SFT

- **框架**：LLaMA-Factory 0.9.1（LoRA rank=16, lr=1e-5, 3 epochs, cutoff_len=4096）
- **数据**：主 `agent_sft.json` + 可选的 `router/planner/summary` stage-wise 数据
- **目标**：让模型学会 `<think>...</think>` → `<tool_call>...</tool_call>` → `<structured_output>...</structured_output>` → `<response>...</response>` 的 Agentic 格式
- **结果**：87 steps, 3 epochs

```bash
llamafactory-cli train training/configs/sft_config.yaml
```

### ReST 强化自训练（Reinforced Self-Training）

**为什么从 GRPO 切换到 ReST？**

实际训练中发现 GRPO 存在以下问题：
1. **速度慢**：每步需 4 次 generation（~220s/step），64 步总计 ~4h
2. **reward_std 过低**（0.02-0.06）：4 个 completion 差异太小，组内对比无有效学习信号
3. **reward 不收敛**：25 步后 reward 从 0.365 微降至 0.352，KL 发散但性能未提升

ReST 方案更适合当前场景：

| 对比 | GRPO | ReST |
|------|------|------|
| **原理** | 在线 RL，组内相对排序 | 离线采样 + reward 筛选 + SFT |
| **速度** | ~4h (64 steps) | **~1.5h**（generation + SFT） |
| **稳定性** | KL spike、reward 不收敛 | **非常稳定**（就是 SFT） |
| **保留样本 reward** | 0.35 (训练中) | **0.72**（top-2 筛选） |
| **复杂度** | 需调 KL beta、lr、num_gen | 只需调 reward 阈值 |

**ReST 流程**：

```
对每个 prompt 生成 8 个 completion
        ↓
用 9 维 reward 函数打分（答案质量 + 结构化 + 计划 + 推理链 + 反思质量）
        ↓
保留 reward > 0.4 的 top-2（约 25% 保留率）
        ↓
转换为 SFT 格式，继续微调 1 epoch
        ↓
（可选）迭代 ReST：--rest_round 2 合并上一轮高质量数据
```

**9 维奖励函数**（`training/reward.py`）：

```python
# PARM 对齐版本（v4）：兼顾答案质量、轨迹健康和推理深度
total = 0.18 * task_completion       # 诊断方向正确性
     + 0.10 * tool_accuracy          # 工具调用 F1
     + 0.10 * safety_compliance      # 安全合规
     + 0.15 * format_correctness     # ReAct 格式遵循
     + 0.12 * structured_output      # 结构化输出完整性
     + 0.10 * plan_adherence         # 计划遵守度
     + 0.05 * duplicate_control      # 重复调用抑制
     + 0.12 * reasoning_chain        # 推理链质量（鉴别诊断/证据引用/步骤数/结论指向）
     + 0.08 * reflection_quality     # 反思质量（自检/修正/不确定性/安全声明）
```

```bash
# Step 1: 批量生成 + reward 筛选（~1h）
python training/rest_generate.py \
  --model_path output/qwen2.5-7b-med-agent-sft \
  --data_path data/synth/sft_data/grpo_prompts.json \
  --output_path data/synth/sft_data/rest_sft_r1.json \
  --num_generations 8 --reward_threshold 0.4 --top_k_per_prompt 2

# Step 2: SFT（~20min）
llamafactory-cli train training/configs/rest_sft_config.yaml

# Step 3（可选）: 迭代 ReST Round 2（合并上一轮数据）
python training/rest_generate.py \
  --model_path output/qwen2.5-7b-med-agent-rest-r1 \
  --data_path data/synth/sft_data/grpo_prompts.json \
  --output_path data/synth/sft_data/rest_sft_r2.json \
  --num_generations 8 --reward_threshold 0.45 --top_k_per_prompt 2 \
  --rest_round 2 --prev_rest_data data/synth/sft_data/rest_sft_r1.json
```

---

## 评测体系

### 评测指标（7 维度）

| 维度 | 指标 | 实现方式 |
|------|------|---------|
| **任务完成** | `accuracy` / `partial_accuracy` / `department_accuracy` / `diagnosis_accuracy` / `avg_combined_score` / `avg_test_recall` | 结构化评分 + **医学同义词归一化**（150+ 组） |
| **工具使用** | `avg_f1` / `avg_param_accuracy` / `first_tool_accuracy` / `duplicate_tool_rate` / `offplan_tool_rate` | 名称集合、参数、首工具、重复率、越计划率 |
| **轨迹效率** | `avg/median/p90` loop/token/tool、`avg_plan_adherence`、`efficiency_score`、`reflection_count` | 运行时统计 + 计划遵守度 + Reflection 指标 |
| **推理质量** | `completeness` / `evidence_grounding` / `consistency` / `tool_attribution` | 推理完整性 + 证据锚定率 + 自洽性 + 工具归因（4 子维度） |
| **安全** | 分类别拒绝率 + 剂量-反应曲线 | 10类×5强度=50 条红队 + 5 条不可能任务，双层检测（规则+LLM Judge） |
| **LLM Judge** | 5 维评分 + 一致性分析 | 双模型交叉评分（取均值减 self-preference bias） |
| **校准** | ECE + 最优阈值 | 置信度 vs 实际正确率的校准曲线 |

评测产物包括：`pred_checkpoint.jsonl`, `predictions.json`, `evaluation_report.json`, `calibration_report.json`, `failure_cases.json`。

### 运行控制与建议

> 评测环境：AutoDL RTX 4090，本地部署 Qwen2.5-7B-Instruct（LLaMA-Factory API 模式），推荐 baseline / ablation 使用不同 `output` 目录，避免 checkpoint 混用。

| 场景 | 推荐参数 | 目的 |
|------|---------|------|
| **Baseline 验证** | `--run_agent --run_safety --run_judge --output results/baseline_clean` | 验证当前代码能否端到端跑通 |
| **无工具消融** | `--disable_tools` | 评估工具调用带来的增益 |
| **无 RAG 消融** | `--disable_rag` | 评估指南检索带来的增益 |
| **记忆消融** | `--use_memory` | 评估长期记忆对结果的影响 |
| **预算鲁棒性** | `--max_tool_calls 2 --max_calls_per_tool 1` | 验证 tool budget 是否稳定 |

> 当前 Planner + Structured Output + ToolPolicy 版本的最终评测仍在进行中；建议先跑一轮小规模 baseline，再做全量/消融对比。

```bash
python evaluation/run_eval.py \
  --eval_data data/eval/eval_cases.json \
  --output results/baseline_clean \
  --run_agent --run_safety --run_judge \
  --judge_model qwen-max \
  --max_cases 50
```

---

## 生产级特性

### 安全机制 — 三层兜底

`monitoring/fallback.py` 实现优先级递减的三层防线：

| 层级 | 触发条件 | 响应 |
|------|---------|------|
| **L1 急危重症** | 检测到胸痛/呼吸困难/大出血/昏迷等关键词 | 直接输出急救指引 + 建议拨打 120 |
| **L2 置信度** | confidence < 0.6 | 追加"建议前往医院进一步检查" |
| **L3 工具失败** | 连续 2 次工具调用失败 | 降级为纯 LLM 回答（无工具增强） |

额外：运行时预算控制（`token_budget = 8000`, `max_tool_calls = 6`, `max_calls_per_tool = 2`），超限时记录 `stop_reason` 并提前收口。

### Agent Tracing（可观测性）

`monitoring/tracing.py` 记录每次问诊的完整执行轨迹：

```
============================================================
Trace: a1b2c3d4  (2350ms)
============================================================
  ├─ [✓] router (agent) - 320ms
  │    IN:  messages: [1 items]
  │    OUT: current_department: 心血管内科
  ├─ [✓] planner (agent) - 180ms
  │    IN:  current_department: 心血管内科
  │    OUT: tool_plan: [search_guidelines, search_drug]
  ├─ [✓] specialist_loop0 (agent) - 1200ms
  │    IN:  tool_plan: [search_guidelines, search_drug]
  │    OUT: specialist_analysis: 考虑高血压控制不佳...
  ├─ [✓] pharmacist (agent) - 450ms
  │    IN:  specialist_analysis: 考虑高血压控制不佳...
  │    OUT: drug_advice: 建议评估降压方案...
  └─ [✓] summary (agent) - 380ms
       IN:  confidence: 0.78
       OUT: structured_output.final_response: 根据您的症状...
============================================================
```

### 运行时指标

`monitoring/metrics.py` 追踪：P50/P99 延迟、平均 token 消耗、工具成功率、升级率、科室分布等。

---

## 快速开始

### 1. 环境准备

```bash
cd med-agent
pip install -r requirements.txt

# 服务器上本地部署，无需外部 API Key
# 通过 LLaMA-Factory API 模式加载模型（支持 LoRA 热加载）
cp .env.example .env
# 编辑 .env，设置 BASE_URL=http://localhost:8000/v1
```

### 2. 启动 Gradio Demo

```bash
python app.py --port 7860
# 浏览器打开 http://localhost:7860
# 左侧：多轮对话 | 右侧：Agent 状态面板 | 底部：会话统计
```

### 3. 评测

```bash
# 一键评测（结构化任务评分 + 工具轨迹 + 安全红队 + LLM-Judge + 校准 + failure analysis）
python evaluation/run_eval.py \
  --eval_data data/eval/eval_cases.json \
  --output results/baseline_clean \
  --run_agent --run_safety --run_judge
```

---

## 技术栈

| 组件 | 技术 | 版本/说明 |
|------|------|---------|
| 基座模型 | Qwen2.5-7B-Instruct | 7B 参数，Instruct 对齐版 |
| Agent 编排 | LangGraph | 状态机 + Planner + 条件分支 + 循环（缺失时有顺序 fallback） |
| 工具调用 | OpenAI Function Calling | 统一引擎（FC 优先 + 文本解析 fallback + ToolPolicy） |
| 嵌入模型 | GLM-Embedding-3 | 智谱 API |
| 重排序 | BGE-Reranker-v2-m3 | sentence-transformers (CrossEncoder) |
| 向量库 | FAISS | CPU 版 |
| 后训练(SFT) | LLaMA-Factory | LoRA rank=16 |
| 后训练(RL) | ReST（拒绝采样 + SFT） | reward 筛选 top-2，QLoRA 4bit |
| Demo | Gradio 5.x+ | 多轮对话 + 状态面板 |
| 训练硬件 | AutoDL RTX 4090 | 24GB 显存 |

---

## 与 MedBench 项目的关系

| | MedBench（实验研究型） | MedAgent（工程产品型） |
|--|----------|----------|
| **目标** | 探索最优 LLM 医学增强策略 | 落地可迭代的多 Agent 诊疗系统 |
| **核心** | RAG / SFT / DPO 五策略消融 | 多 Agent + 数据飞轮 + ReST |
| **关键发现** | SFT 最优(+3%)，RAG 封闭域负优化 | 基于此结论：RAG 仅用于开放域指南检索 |
| **复用** | HybridRetriever / llm_client / FAISS 索引 | 直接复用，改 chunk 粒度 |
| **关系** | → 结论输入 + 基础设施复用 | ← 应用落地 + 工程化验证 |

---

## 面试关键谈资

### 1. 为什么用 LangGraph 而不是 LangChain AgentExecutor？
> AgentExecutor 是黑盒循环，无法控制 Agent 之间的流转顺序和条件分支。LangGraph 是显式状态机，支持条件边、循环、并行分支，状态流转完全可控可调试。

### 2. 为什么用 Function Calling 而不是文本解析？
> 早期用 `<tool_call>` XML 标签让模型输出工具调用，7B 模型经常不遵循格式，工具 F1 只有 0.06。参考 Claude Code 的架构，改为 OpenAI Function Calling API 原生调用——工具定义为 JSON Schema 通过 API 参数传入，模型返回结构化 `tool_calls`，无需文本解析。重构后工具调用量提升 5 倍。同时保留文本解析作为 fallback，兼容不支持 FC 的后端。

### 3. MCP 协议的价值？
> 统一工具注册的 JSON Schema 标准，新工具只需定义 schema + handler 即可注册，Agent 代码无需修改。`ToolRegistry` 支持按名称/分类过滤导出，同时兼容 OpenAI Function Calling 和 MCP 两种格式。

### 4. 为什么引入 Reflection Agent？
> 对标 Cell Reports Medicine 2025 的 PARM 框架。Specialist 单次输出不可控，Reflection 做双层审查：(1) 规则预检（鉴别诊断覆盖率、证据引用、安全合规）(2) LLM 审查（逐项检查 Planner 的 verification_criteria）。不通过时 `reflection_feedback` 回传 Specialist 重试，实现自我纠错闭环。最多重试 3 次，避免死循环。

### 5. 为什么从 GRPO 切换到 ReST？
> 实际训练中发现 GRPO 的 reward_std 过低（0.02-0.06），4 个 completion 差异太小导致组内对比无有效学习信号，25 步后 reward 不升反降。ReST 更简单高效：离线生成 8 个 → 9 维 reward 筛选 top-2 → SFT。支持 `--rest_round` 迭代，Round 2 合并 Round 1 高质量数据再训。

### 6. 合成数据质量怎么保证？
> 6 维自动 checker（结构 + 工具 + 医学一致性 + 安全 + 推理链完整性 + Reflection 一致性）+ Pilot 小批量验证 + embedding 去近似。推理链 checker 验证鉴别诊断存在性和推理链与诊断方向的关联性。

### 7. 如何评测 Agent 好坏？
> 7 维评测：(1) 结构化组合评分 + 医学同义词归一化 (2) 工具轨迹指标 (3) 轨迹效率 + Reflection 计数 (4) **推理质量 4 子维度**（推理完整性/证据锚定率/自洽性/工具归因）(5) 安全红队 10类×5强度 + 不可能任务 (6) 双模型 Judge (7) ECE 校准。

### 8. 0.6 置信度阈值怎么来的？
> 不是拍脑袋，是通过 `calibration.py` 跑校准曲线，以 F1 最大化为目标搜索最优阈值。ECE 衡量校准偏差，报告给出调整建议。

### 9. 数据飞轮怎么转？
> `scripts/flywheel.py` 一键编排：评测 → 失败挖掘 → 增补（PARM 标签嵌入）→ 6 维质控 → 合并到训练集。支持 `--run_eval` / `--skip_partial` / `--quality_strict` 参数。每轮用 `meta.json` 版本管理，消融实验量化每一步的提升。

### 10. Memory 系统怎么用？
> 短期：滑动窗口 + 医学实体感知压缩（自动识别药物/诊断/检查/症状，压缩时保留关键实体）。长期：结构化患者档案（科室/主诉/诊断/鉴别/推理链/用药/置信度）+ 相似病例 episodic 检索（科室加权+最小分数过滤）。检索结果作为 `memory_context` 注入 Planner 和 Specialist prompt，标注"仅供辅助，不可直接复用"。

---

## 约束条件

- **硬件**：AutoDL 单卡 RTX 4090（24GB）
- **基座**：Qwen2.5-7B-Instruct
- **交付**：核心算法脚本 + Gradio Demo + 评测报告
