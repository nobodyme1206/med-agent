# MedAgent — 医学多智能体诊疗系统

基于 **LangGraph + OpenAI Function Calling + MCP Tool Registry + ReST 强化自训练**，构建端到端可上线、可迭代、可控成本的医学多 Agent 诊疗系统。覆盖 Agent 架构设计、数据飞轮、后训练闭环（SFT → ReST）、多维评测体系和生产级安全交付。

---

## 系统架构

```
                            ┌──────────────────────────────────────────────────┐
                            │              MedAgent 系统架构                     │
                            └──────────────────────────────────────────────────┘

  用户输入（症状描述）
         │
         ▼
  ┌──────────────┐     ┌─────────────────────────────────────────────────────┐
  │ Router Agent │────▶│            LangGraph 状态机编排                       │
  │  意图识别     │     │                                                     │
  │  科室路由     │     │   Router ─┬─▶ Specialist ─▶ Pharmacist ─▶ Summary  │
  │  急诊拦截     │     │           │       ↑    │                     │      │
  └──────────────┘     │           │       └────┘ (max 3 轮)          │      │
                       │           └─▶ Summary（急诊直达）              │      │
                       └─────────────────────────────────────────────────────┘
                                          │
         ┌────────────────────────────────┼─────────────────────────────┐
         ▼                                ▼                             ▼
  ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
  │  工具层 (MCP) │              │   记忆系统    │              │  安全兜底     │
  ├──────────────┤              ├──────────────┤              ├──────────────┤
  │ 药品知识库    │              │ 短期：滑窗+摘要│              │ 急危重症拦截  │
  │ 诊疗指南 RAG  │              │ 长期：FAISS   │              │ 置信度兜底    │
  │ 检验值解读    │              │              │              │ 工具失败降级  │
  └──────────────┘              └──────────────┘              └──────────────┘
         │                                                          │
         ▼                                                          ▼
  最终回复（含溯源 + 安全声明 + 就医建议）              Tracing / Metrics / Alerts
```

---

## 核心技术亮点

| 维度 | 实现 | 应对面试追问 |
|------|------|-------------|
| **Agent 编排** | LangGraph 状态机，支持条件分支 + 循环 + 急诊直达 | 为什么不用 AgentExecutor → 状态可控、可调试 |
| **推理模式** | ReAct（Reasoning + Acting），结构化思考链 | FC 优先 + 文本解析 fallback，双模式保障 |
| **工具调用** | 统一引擎：OpenAI Function Calling 优先，文本解析兜底 | 参考 Claude Code 原生 tool_use 架构，工具 schema 从 Registry 自动获取 |
| **工具注册** | MCP 兼容注册中心，JSON Schema 标准化，统一调用日志 | 工具扩展只需注册，无需改 Agent 代码 |
| **混合检索** | BM25 + GLM-Embedding-3 + RRF 融合 + BGE-Reranker-v2-m3 | 稀疏+稠密互补，Reranker 精排 |
| **记忆系统** | 短期滑动窗口（LLM 摘要压缩）+ 长期 FAISS 向量存储 | 支持跨会话患者档案检索 |
| **后训练** | Agentic SFT → ReST 强化自训练（拒绝采样 + SFT，QLoRA 4bit） | 比 GRPO 快 3-5x，单卡 4090 可训练 |
| **奖励函数** | 4 维组合：任务(0.30) + 格式(0.30) + 工具(0.20) + 安全(0.20) | 子奖励独立可调，支持消融 |
| **数据质控** | 5 维自动 checker + embedding 去重 + Pilot 小批量验证 | 面试关键：合成数据质量如何保证 |
| **评测体系** | 语义匹配 + 工具参数级 F1 + 双模型 Judge + 置信度校准 | 面试关键：如何评测 Agent 好坏 |
| **安全机制** | 三层兜底 + 安全红队评测（10 类攻击） | 置信度阈值来自 ECE 校准分析 |
| **可观测性** | Agent Tracing + 运行时指标 + 告警规则 | 每个节点输入/输出/延迟全链路追踪 |

---

## 项目结构

```
med-agent/
├── agents/                          # 4 个 Agent
│   ├── router.py                    # 分诊 Agent（意图识别 + 科室路由）
│   ├── specialist.py                # 专科 Agent（ReAct + RAG + 检验解读）
│   ├── pharmacist.py                # 药师 Agent（药品查询 + 交互检查）
│   └── summary.py                   # 汇总 Agent（整合 + 安全兜底 + 置信度）
├── graph/
│   ├── state.py                     # AgentState 共享状态（TypedDict）
│   └── workflow.py                  # LangGraph 状态机编排 + Tracing 集成
├── tools/
│   ├── registry.py                  # MCP 兼容工具注册中心（JSON Schema）
│   ├── drug_lookup.py               # 工具：药品知识库查询
│   ├── guideline_rag.py             # 工具：诊疗指南混合检索
│   ├── lab_interpreter.py           # 工具：检验值解读
│   └── setup.py                     # 工具统一初始化
├── memory/
│   ├── short_term.py                # 短期记忆：滑动窗口 + LLM 摘要压缩
│   └── long_term.py                 # 长期记忆：FAISS 向量存储患者历史
├── training/
│   ├── reward.py                    # 多维度组合奖励函数（ReST 筛选 + GRPO 共用）
│   ├── rest_generate.py             # ReST 数据生成（批量采样 + reward 筛选）
│   ├── grpo_train.py                # GRPO 训练脚本（探索参考，已切换到 ReST）
│   └── configs/
│       ├── sft_config.yaml          # Agentic SFT 配置（LLaMA-Factory 格式）
│       └── rest_sft_config.yaml     # ReST SFT 配置（筛选数据上继续微调）
├── evaluation/
│   ├── task_eval.py                 # 任务完成率（语义匹配）+ 工具 F1（含参数级）
│   ├── trajectory_eval.py           # 轨迹效率评测
│   ├── safety_eval.py               # 安全红队评测（10 类攻击场景）
│   ├── llm_judge.py                 # 双模型 LLM-as-Judge + Cohen's Kappa 标定
│   ├── calibration.py               # 置信度校准（ECE + 最优阈值搜索）
│   └── run_eval.py                  # 一键评测入口（含校准分析）
├── monitoring/
│   ├── tracing.py                   # Agent 调用链 Tracing（Span/Trace 全链路）
│   ├── metrics.py                   # 运行时指标（P50/P99 延迟、token、成功率）
│   ├── alerts.py                    # 告警规则（错误率/延迟超阈值）
│   └── fallback.py                  # 三层兜底策略
├── scripts/
│   ├── generate_synth_data.py       # 合成 trajectory（含 Pilot 模式 + 质控集成）
│   ├── convert_traj_to_sft.py       # trajectory → SFT/ReST 训练数据
│   ├── data_quality.py              # 5 维数据质控管线 + embedding 去重
│   ├── run_ablation.py              # 消融实验脚本（8 组配置对比）
│   ├── build_drug_kb.py             # 构建药品知识库
│   └── build_guideline_index.py     # 构建诊疗指南 RAG 索引
├── utils/
│   ├── llm_client.py                # LLM API 统一封装（chat / FC / embed / JSON mode）
│   └── tool_agent.py                # 统一工具调用引擎（FC 优先 + 文本解析 fallback）
├── data/
│   ├── drug_kb/                     # 药品知识库（结构化 JSON）
│   ├── lab_ranges/                  # 检验值参考范围
│   ├── guidelines/                  # 诊疗指南文本 + FAISS 索引
│   ├── synth/                       # 合成训练数据 + 版本管理
│   └── eval/                        # 评测数据集
├── app.py                           # Gradio Demo（对话 + 状态面板 + 统计）
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
| `current_department` | `str` | Router 分配的科室 |
| `specialist_analysis` | `str` | 专科分析结果 |
| `drug_advice` | `str` | 药师用药建议 |
| `tool_calls` | `List[ToolCallRecord]` | 工具调用记录（自动追加） |
| `retrieved_knowledge` | `List[Dict]` | RAG 检索结果 |
| `confidence` | `float` | 当前置信度（0-1，< 0.6 触发兜底） |
| `should_escalate` | `bool` | 是否需要人工接管 |
| `final_response` | `str` | 最终输出给用户的回复 |

### 四个 Agent

| Agent | 职责 | 可用工具 | 推理模式 |
|-------|------|---------|--------|
| **Router** | 解析症状 → 判断科室 → 急诊拦截 → 路由 | 无 | Few-shot + JSON mode（保证输出格式） |
| **Specialist** | 多轮问诊 + 检验解读 + 初步诊断建议 | `search_guidelines`, `interpret_lab_result` | 统一引擎（FC 优先） |
| **Pharmacist** | 药物查询 + 交互检查 + 用药建议 | `search_drug`, `check_drug_interaction`, `search_by_indication` | 统一引擎（FC 优先） |
| **Summary** | 汇总所有 Agent 输出 + 安全检查 + 置信度评估 | 无 | 单次生成 + Guardrails |

### LangGraph 状态机

```
START → Router ─┬─→ Specialist ──→ Pharmacist ──→ Summary → END
                │       ↑     │
                │       └─────┘ (需要补充信息，max 3 轮)
                │
                └─→ Summary（急诊/升级 → 直接兜底）

条件分支：
  • Router 检测急危重症关键词 → should_escalate = True → 跳过专科直达 Summary
  • Specialist 分析为空 → 循环重试（最多 3 轮）
  • Summary 置信度 < 0.6 → 追加就医建议
```

每个 Agent 节点自动集成 **Tracing**（`monitoring/tracing.py`），记录输入/输出/延迟/错误。

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
- 兼容 LLaMA-Factory vLLM 后端（`user/assistant` 交替消息格式）
- 消融实验可通过 `tools_enabled` / `rag_enabled` 开关控制

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

## 记忆系统

| 类型 | 实现 | 用途 |
|------|------|------|
| **短期记忆** | 滑动窗口（最近 10 轮），超限时 LLM 摘要压缩 | 当前会话上下文 |
| **长期记忆** | FAISS 向量存储，会话结束自动摘要存入 | 跨会话患者档案检索 |

---

`scripts/generate_synth_data.py` 用 LLM API 生成多轮问诊对话，每条 trajectory 包含：

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
  "tools_used": ["search_guidelines", "search_drug"]
}
```

**Pilot 模式**：`--pilot` 先生成 10 条 → 自动质控 → 人工确认 → 再批量生成：

```bash
# Step 1: 小批量验证
python scripts/generate_synth_data.py --pilot --pilot_size 10

# Step 2: 确认无误后批量生成（自动集成质控过滤）
python scripts/generate_synth_data.py --num_cases 500
```

### 5 维数据质控管线

`scripts/data_quality.py` 对每条合成 trajectory 做 5 维自动检查：

| 维度 | 检查内容 | 示例 |
|------|---------|------|
| **结构完整性** | 必填字段非空、对话 ≥ 2 轮、角色交替 | dialogue 缺 agent 回复 → 丢弃 |
| **工具调用合理性** | 工具名在注册表中、参数为合法 dict | 调用不存在的工具 → 丢弃 |
| **医学术语一致性** | 科室-主诉逻辑匹配、诊断方向非空 | 骨科 case 出现"血糖" → 标记 |
| **安全合规** | 无确定性诊断、包含就医建议 | "你得了糖尿病" → 丢弃 |
| **去重** | MD5 精确去重 / embedding 余弦相似度去近似 | 相似度 > 0.95 → 去重 |

```bash
python scripts/data_quality.py \
  --input data/synth/trajectories/all_trajectories.json \
  --output data/synth/trajectories/filtered.json \
  --use_embedding --sim_threshold 0.95
```

---

## 后训练

### 训练迭代链路

```
Qwen2.5-7B-Instruct（基座）
        ↓
Agentic SFT（学会 ReAct 格式 + 工具调用模式）
        ↓
ReST Round 1（拒绝采样 → reward 筛选 → SFT 强化）
        ↓
评测 → 分析 bad case → 补充合成数据
        ↓
ReST Round 2（迭代优化）
```

### Agentic SFT

- **框架**：LLaMA-Factory 0.9.1（LoRA rank=16, lr=1e-5, 3 epochs, cutoff_len=4096）
- **数据**：499 条 trajectory 转换的 ShareGPT 格式数据（保留 thought-action-observation 结构）
- **目标**：让模型学会 `<think>...</think>` → `<tool_call>...</tool_call>` → `<response>...</response>` 的 ReAct 格式
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
用 4 维 reward 函数打分
        ↓
保留 reward > 0.4 的 top-2（约 25% 保留率）
        ↓
转换为 SFT 格式，继续微调 1 epoch
```

**多维度奖励函数**（`training/reward.py`）：

```python
# 权重可调的组合奖励（v2 平衡版）
total = 0.30 * task_completion   # 中文 n-gram 关键词匹配
     + 0.30 * format_correctness  # ReAct 格式正确性
     + 0.20 * tool_accuracy       # 工具调用格式 + 名称匹配
     + 0.20 * safety_compliance   # 安全合规（无确诊、无处方、有就医建议）
```

```bash
# Step 1: 批量生成 + reward 筛选（~1h）
python training/rest_generate.py \
  --model_path output/qwen2.5-7b-med-agent-sft \
  --data_path data/grpo_prompts.json \
  --output_path data/rest_sft.json \
  --num_generations 8 --reward_threshold 0.4 --top_k_per_prompt 2

# Step 2: 在筛选数据上继续 SFT（~20min）
llamafactory-cli train training/configs/rest_sft_config.yaml
```

---

## 评测体系

### 评测指标（6 维度）

| 维度 | 指标 | 实现方式 |
|------|------|---------|
| **任务完成** | 诊断准确率 + 部分准确率 | embedding 语义相似度（cosine > 0.7 为正确），fallback 关键词匹配 |
| **工具使用** | 工具名 F1 + 参数准确率 | 名称集合 F1 + 逐参数精确匹配 |
| **轨迹效率** | 平均步数 + Token 消耗 | Agent 循环次数 + input/output tokens |
| **安全** | 红队攻击拒绝率 | 10 类攻击场景（确诊诱导、索取处方、有害信息、隐私泄露…） |
| **推理质量** | LLM-as-Judge 5 维评分 | 双模型交叉评分（取均值减 self-preference bias） |
| **校准** | ECE + 最优阈值 | 置信度 vs 实际正确率的校准曲线 |

### 消融对比

> 评测环境：AutoDL RTX 4090，本地部署 Qwen2.5-7B-Instruct（LLaMA-Factory API 模式），50 条评测集。

| 模型 | 任务完成率 | 语义相似度 | 工具 F1 | 工具调用/case | Judge 总分 | Judge 安全 |
|------|-----------|-----------|--------|-------------|----------|----------|
| **Base (Qwen2.5-7B)** | 2.0% | 0.213 | 0.023 | — | 4.06 | 4.70 |
| + Agentic SFT | 4.0% | 0.191 | 0.062 | 0.18 | 4.06 | 4.74 |
| + ReST R1 | 4.0% | 0.197 | 0.085 | 0.24 | 4.09 | 4.76 |
| + **Function Calling 重构** | — | — | 🔄评测中 | ~1.0+ | — | — |

> 注：Function Calling 重构后工具调用从 ~0.2 次/case 提升到 ~1.0 次/case（5 倍提升），评测进行中。

```bash
python evaluation/run_eval.py \
  --eval_data data/eval/cases_500.json \
  --output results/ \
  --run_agent --run_safety --run_judge \
  --judge_model qwen-max
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

额外：Token 预算控制（单次会话上限 8000 tokens），超出时截断历史 + 摘要压缩。

### Agent Tracing（可观测性）

`monitoring/tracing.py` 记录每次问诊的完整执行轨迹：

```
============================================================
Trace: a1b2c3d4  (2350ms)
============================================================
  ├─ [✓] router (agent) - 320ms
  │    IN:  messages: [1 items]
  │    OUT: current_department: 心血管内科
  ├─ [✓] specialist_loop0 (agent) - 1200ms
  │    IN:  current_department: 心血管内科
  │    OUT: specialist_analysis: 考虑高血压2级...
  ├─ [✓] pharmacist (agent) - 450ms
  │    IN:  specialist_analysis: 考虑高血压2级...
  │    OUT: drug_advice: 建议调整降压方案...
  └─ [✓] summary (agent) - 380ms
       IN:  confidence: 0.78
       OUT: final_response: 根据您的症状...
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
# 一键评测（任务完成率 + 工具F1 + 安全红队 + LLM-Judge + 校准分析）
python evaluation/run_eval.py \
  --eval_data data/eval/cases_500.json \
  --output results/ \
  --run_agent --run_safety --run_judge
```

---

## 技术栈

| 组件 | 技术 | 版本/说明 |
|------|------|---------|
| 基座模型 | Qwen2.5-7B-Instruct | 7B 参数，Instruct 对齐版 |
| Agent 编排 | LangGraph | 状态机 + 条件分支 + 循环 |
| 工具调用 | OpenAI Function Calling | 统一引擎（FC 优先 + 文本解析 fallback） |
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

### 4. 为什么从 GRPO 切换到 ReST？
> 实际训练中发现 GRPO 的 reward_std 过低（0.02-0.06），4 个 completion 差异太小导致组内对比无有效学习信号，25 步后 reward 不升反降。ReST（拒绝采样 + SFT）更简单高效：离线生成 8 个 completion → reward 筛选 top-2（avg reward 0.72 vs GRPO 的 0.35）→ 在高质量数据上 SFT。速度快 3-5x，训练完全稳定。

### 5. 合成数据质量怎么保证？
> 5 维自动 checker（结构、工具、医学一致性、安全、去重）+ Pilot 小批量验证 + embedding 去近似。通过率低于 50% 时自动阻断并提示检查 prompt。

### 6. 如何评测 Agent 好坏？
> 6 维评测：(1) embedding 语义匹配替代关键词 (2) 工具参数级 F1（不只看名称）(3) 双模型 Judge 交叉评分减 bias (4) Cohen's Kappa 人工标定 (5) 置信度 ECE 校准 (6) 安全红队 10 类攻击。

### 7. 0.6 置信度阈值怎么来的？
> 不是拍脑袋，是通过 `calibration.py` 跑校准曲线，以 F1 最大化为目标搜索最优阈值。ECE 衡量校准偏差，报告给出调整建议。

### 8. 数据飞轮怎么转？
> 合成 → 质控 → SFT → ReST（采样+筛选+SFT）→ 评测 → 分析 bad case → 补充合成 → 迭代。每轮用 `meta.json` 版本管理，消融实验量化每一步的提升。

### 9. 可观测性怎么做？
> 全链路 Tracing：每个 Agent 节点的输入/输出/延迟/错误自动记录，支持 JSON 持久化和控制台可视化。运行时 metrics 追踪 P50/P99 延迟、工具成功率。

---

## 约束条件

- **硬件**：AutoDL 单卡 RTX 4090（24GB）
- **基座**：Qwen2.5-7B-Instruct
- **交付**：核心算法脚本 + Gradio Demo + 评测报告
