# MedAgent × Claude Code (cc-haha) 深度对比分析报告

> 基于两个项目的源码深度分析 + 学术文献联网调研，从 Agent 架构设计、工具系统、评测体系三个维度展开。

---

## 一、架构总览对比

| 维度 | MedAgent | Claude Code (cc-haha) |
|------|----------|----------------------|
| **语言/运行时** | Python (LangGraph + OpenAI SDK) | TypeScript (Bun + Ink TUI) |
| **Agent 编排** | LangGraph 显式状态机（5 节点有向图） | Coordinator-Worker 异步多 Agent（无固定图） |
| **Agent 数量** | 固定 5 个专业 Agent | 动态 N 个 Worker + 1 个 Coordinator |
| **状态管理** | TypedDict 全局共享状态（AgentState） | 每个 Worker 独立上下文 + 消息传递 |
| **工具调用** | 统一引擎（FC 优先 + 文本 fallback）+ ToolPolicy | 原生 Anthropic tool_use + 权限系统 |
| **工具注册** | MCP 兼容 ToolRegistry（JSON Schema） | 静态工具池 + MCP Server 动态扩展 |
| **记忆系统** | 短期滑窗 + 长期 FAISS | 跨会话持久化记忆（memdir） |
| **可观测性** | 自建 Tracing（Span/Trace） | OpenTelemetry 全链路 |
| **后训练** | SFT → ReST 数据飞轮 | 无（依赖基座模型能力） |
| **评测** | 6 维评测体系 + 消融实验 | 无内建评测框架 |

---

## 二、Agent 编排架构深度对比

### 2.1 MedAgent：显式状态机编排

MedAgent 采用 **LangGraph StateGraph** 构建确定性的有向图：

```
START → Router → Planner → Specialist ──┬── Pharmacist → Summary → END
                              ↑ loop    │
                              └─────────┘
```

**核心设计特点**：
- **固定拓扑**：5 个角色节点 + 条件边，流程完全可预测
- **条件路由**：`after_router()` / `after_specialist()` 等函数根据状态字段（`should_escalate`、`loop_count`、`need_pharmacist`）做确定性路由
- **循环控制**：Specialist 分析为空时可循环重试（max 3），有明确的 `max_loops` 上限
- **顺序 fallback**：`_SequentialWorkflow` 在 LangGraph 不可用时提供 Python 原生的顺序执行
- **全局共享状态**：所有 Agent 读写同一个 `AgentState` TypedDict，字段间有明确的上下游依赖

**优势**：可控、可调试、可消融（每个节点可单独开关），适合医学场景的安全合规要求。

**局限**：无法动态扩展新角色；并行度受限于图结构。

### 2.2 Claude Code：Coordinator-Worker 异步编排

cc-haha 采用 **Coordinator 模式**，Coordinator 是一个特殊的主 Agent，通过 `AgentTool` 动态派生 Worker：

```
User → Coordinator ──┬── Worker A (research)     ← 并行
                     ├── Worker B (research)     ← 并行
                     ├── [Coordinator synthesizes]
                     ├── Worker C (implementation)← 串行
                     └── Worker D (verification)  ← 串行
```

**核心设计特点**：
- **动态拓扑**：没有预定义的图结构，Coordinator 根据任务自主决定派出几个 Worker、什么顺序
- **异步并行**：Worker 通过 `<task-notification>` XML 异步回报结果，支持真正的并行执行
- **上下文隔离**：每个 Worker "starts with zero context"，需要 Coordinator 在 prompt 中完整传递信息
- **续传机制**：`SendMessageTool` 可以继续已完成 Worker 的上下文（复用 prompt cache）
- **Fork 语义**：`forkSubagent` 允许 fork 自身上下文给子 Agent，共享 prompt cache 降低成本
- **四阶段工作流**：Research → Synthesis → Implementation → Verification，但这是建议而非强制

**优势**：极高灵活性，适合开放性编程任务；并行能力强。

**局限**：不可预测、难以做确定性评测；Worker 结果质量依赖 Coordinator 的 prompt 质量。

### 2.3 学术视角评价

根据 PMC 系统综述（*AI Agents in Clinical Medicine*, 2025）的研究发现：

> **"Our analysis identified highest performance for teams of 4–5 agents, beyond which performance plateaus or even declines."**

MedAgent 的 5 Agent 设计恰好处于学术研究发现的最优团队规模范围内。该综述还指出：

> **"Sequential processing (45.5% of studies) is optimal for stepwise workflows like diagnostic workups."**

MedAgent 的顺序编排（Router → Planner → Specialist → Pharmacist → Summary）与临床诊疗工作流的阶段性特征天然匹配。而 cc-haha 的 Coordinator-Worker 模式更接近 **Supervisor coordination (36.4%)**，适合需要灵活编排的编程任务。

该综述的三层框架建议：
- **Tier 1（临床试点）**：工具增强 LLM → 对应 MedAgent 的 Specialist 单独使用工具
- **Tier 2（监督研究）**：单 Agent 完整工作流 → 对应 MedAgent 的端到端问诊流程
- **Tier 3（纯研究）**：多 Agent 系统 → MedAgent 属于此层，但通过固定拓扑+安全兜底降低了风险

> **关键发现**：Multi-agent 系统相比 single-agent 仅有 +17% 的中位数提升，且方差极大。MedAgent 的价值不在于 multi-agent 本身带来的提升，而在于 **Planner 的工具规划约束 + ToolPolicy 的执行控制 + Summary 的安全收口**，这些是单纯堆叠 Agent 数量无法实现的。

---

## 三、工具系统深度对比

### 3.1 工具注册与发现

| 特性 | MedAgent (`tools/registry.py`) | cc-haha (`src/tools.ts`) |
|------|------|------|
| **注册方式** | `ToolRegistry.register(ToolDefinition)` 运行时注册 | 静态导入 + 条件编译（`feature()` 开关） |
| **Schema 格式** | JSON Schema（兼容 OpenAI FC + MCP） | TypeScript 类型定义（运行时转 JSON Schema） |
| **工具数量** | 5 个领域工具 | 40+ 通用工具 |
| **分类过滤** | `filter_names` / `filter_category` | `filterToolsByDenyRules` 权限过滤 |
| **MCP 支持** | `get_mcp_tools()` 导出 MCP 格式 | 原生 MCP Server 集成 |
| **调用日志** | `ToolCallRecord`（自动记录延迟/成功率） | OpenTelemetry spans |

MedAgent 的 `ToolRegistry` 是一个轻量级但完备的注册中心，其 `get_openai_tools()` 和 `get_mcp_tools()` 双格式导出设计，参考了 Claude Code 的工具标准化思路。

cc-haha 的工具系统更复杂：`assembleToolPool()` 将内建工具和 MCP 工具合并去重，`filterToolsByDenyRules()` 提供权限级别的工具过滤，甚至有 `ToolSearchTool` 让 Agent 动态搜索可用工具。

### 3.2 工具调用引擎

**MedAgent 的 `run_tool_agent()`** 是核心创新之一：

```python
# 优先 Function Calling
fc_result = _run_fc_mode(...)   # OpenAI FC API 原生调用
if fc_result is not None:
    return fc_result

# Fallback：文本解析
return _run_text_mode(...)       # <tool_call> 标签解析
```

核心机制：
- **ToolPolicy 七重检查**：tools_enabled → 规划约束（`allowed_tool_names`）→ RAG gating → 总量上限 → 单工具上限 → 重复调用抑制 → token 预算
- **结果压缩**：`_summarize_tool_result()` 将工具返回压缩至 600 字符再回灌上下文
- **call_signature 去重**：对参数做归一化后生成签名，完全相同的调用被自动跳过

**cc-haha 的工具调用**：
- 原生 Anthropic `tool_use` 模式，模型直接输出 `tool_use` content block
- 权限系统（`ToolPermissionContext`）在调用前做安全检查
- Worker 的工具集由 `ASYNC_AGENT_ALLOWED_TOOLS` 白名单控制
- Token budget 通过 `checkTokenBudget()` 做 diminishing returns 检测

**MedAgent 从 cc-haha 借鉴的关键设计**：
1. FC 优先 + fallback 双模式（README 明确标注 "参考 cc-haha 的原生 tool_use 模式"）
2. ToolPolicy 的概念类似 cc-haha 的 `filterToolsByDenyRules` + token budget
3. 工具 schema 从 Registry 自动获取，Agent 只需声明工具名

### 3.3 学术评价

MedAgentBench（Stanford, NEJM AI 2025）的评测框架提供了参考：
- **300 个临床任务 + FHIR 兼容环境 + 标准化 API** → MedAgent 可考虑对接 FHIR 标准提升互操作性
- **最佳模型 Claude 3.5 Sonnet 仅 69.67% 成功率** → 说明医学 Agent 的工具调用仍有巨大提升空间
- 该 benchmark 强调 **tool calling 决策能力**：GPT-4 工具调用决策准确率 87.5%，而 Llama-3-70B 仅 39.1%，Mixtral-8B 仅 7.8%

这与 MedAgent 的经验一致：7B 模型用文本解析模式工具 F1 极低（README 提到仅 0.06），切换到 FC 模式后 "工具调用量提升 5 倍"。

---

## 四、评测体系深度分析

### 4.1 MedAgent 的 6 维评测

MedAgent 建立了目前开源医学 Agent 项目中较为完备的评测框架：

| 维度 | 指标 | 对标学术标准 |
|------|------|------------|
| **任务完成** | combined_score（0.4×科室+0.6×诊断方向语义相似度）+ test_recall | 类似 MedAgentBench 的 task success rate |
| **工具使用** | F1 / 参数准确率 / first_tool / 重复率 / 越计划率 | 对标 AgentClinic 的 tool trajectory 评估 |
| **轨迹效率** | loop/token/tool 统计 + plan_adherence + efficiency_score | 学术界缺少此维度，MedAgent 的创新点 |
| **安全红队** | 10 类攻击拒绝率 | 对标 SafetyBench / red-teaming 范式 |
| **LLM Judge** | 双模型 5 维评分 + 一致性分析 | 对标 MT-Bench / AlpacaEval 的 Judge 范式 |
| **校准** | ECE + 最优阈值搜索 | 对标 calibration literature |

### 4.2 与 cc-haha 评测能力的对比

cc-haha **没有内建评测框架**。它的质量保证依赖于：
- 用户交互中的实时反馈
- Coordinator 的 Verification 阶段（派 Worker 做测试）
- OpenTelemetry 指标监控

这是两个项目定位差异的体现：cc-haha 是通用编程工具，其 "评测" 是每次交互的实时验证；MedAgent 是医学领域系统，需要离线、可重复、多维度的评测来证明安全性和有效性。

### 4.3 学术评价与改进建议

根据最新研究，MedAgent 的评测体系在以下方面值得关注：

**✅ 做得好的**：
1. **多维度评测**：同时看任务正确率、工具轨迹、安全性、校准，这比绝大多数医学 Agent 项目更全面
2. **failure analysis + hard-case 回灌**：形成了评测→训练的闭环，对标 data flywheel 最佳实践
3. **消融实验支持**：`--disable_tools` / `--disable_rag` / `--use_memory` 开关设计精良
4. **断点续跑**：checkpoint.jsonl 设计在实际评测中非常实用
5. **双模型 Judge**：减少 self-preference bias，这是 LLM-as-Judge 的已知问题

**⚠️ 可改进的**：

1. **缺少真实临床数据验证**
   - PMC 综述明确指出："Multi-agent systems showed high variability, modest improvements over single agents, and in one randomized trial, performed worse than physicians."
   - 建议：即使无法获取真实 EHR 数据，也应尝试对接 MedAgentBench 的 FHIR 虚拟环境，用标准化的临床任务验证

2. **诊断评测过于依赖文本匹配**
   - 当前 `_soft_text_match` 使用 ROUGE-L + 关键词 F1，对医学同义词和等价表述覆盖不足
   - 建议：集成医学 NER（如 CMeEE）或 UMLS 语义匹配

3. **工具评测缺少因果分析**
   - 当前只看 "调了什么工具"，没有评估 "工具结果是否被正确整合到最终回复中"
   - 建议：增加 tool-output-to-response attribution 评测

4. **缺少 Agent 间通信质量评测**
   - Router → Planner → Specialist 的信息传递质量没有单独评测
   - 建议：参考 cc-haha Coordinator 的 "synthesize findings" 要求，评测 Planner 的 tool_plan 质量

5. **安全评测可对标更高标准**
   - 10 类攻击场景是好的起点，但可参考 AgentClinic 的 24 种认知偏差扰动和 MAQuE 的行为层评测
   - 建议：增加剂量-反应分析（同一攻击的不同强度）

---

## 五、MedAgent 可从 cc-haha 借鉴的改进方向

### 5.1 并行 Agent 执行

cc-haha 的 Coordinator 最强大的能力是 **并行派发 Worker**。MedAgent 当前是严格串行的：

```
Router → Planner → Specialist → Pharmacist → Summary
```

可考虑：当 Planner 判断同时需要 Specialist 和 Pharmacist 时，并行执行两者，再在 Summary 汇总。LangGraph 支持并行分支（`add_node` + 并行边），实现成本不高。

### 5.2 Worker 续传与上下文复用

cc-haha 的 `SendMessageTool` 允许继续已完成 Worker 的上下文。MedAgent 中 Specialist 循环重试时每次都重建完整 prompt，可以借鉴续传思路，将前一轮的工具结果作为增量上下文传入，减少 token 浪费。

### 5.3 动态工具搜索

cc-haha 有 `ToolSearchTool`，当工具池很大时，Agent 可以先搜索相关工具再调用。MedAgent 当前工具数量较少（5 个），但如果未来扩展到更多科室/更多知识库，这个设计值得借鉴。

### 5.4 权限与安全系统

cc-haha 的 `filterToolsByDenyRules` + `ToolPermissionContext` 提供了多层权限控制。MedAgent 的 ToolPolicy 已经做了规划约束和预算控制，但缺少 **动态权限调整**（例如：对不同风险等级的患者调整工具权限）。

### 5.5 Prompt Cache 优化

cc-haha 对 prompt cache 的优化非常极致：
- 工具列表排序保持稳定（`assembleToolPool` 中的 `byName` 排序）
- Agent 列表从 tool description 移到 attachment message（减少 10.2% cache 创建 token）
- Fork subagent 共享 parent 的 prompt cache

MedAgent 在多轮对话场景中也可以做类似优化：将不变的系统 prompt + 工具 schema 缓存，只传递增量上下文。

---

## 六、总结

### MedAgent 的架构竞争力

| 维度 | 评分 | 说明 |
|------|------|------|
| **Agent 编排** | ★★★★☆ | LangGraph 状态机 + 条件路由 + 循环控制，可控性强；缺少并行能力 |
| **工具系统** | ★★★★☆ | FC + fallback 双模式 + ToolPolicy 七重检查，参考了 cc-haha 的最佳实践；MCP 兼容 |
| **评测体系** | ★★★★★ | 6 维评测 + 消融 + failure 回灌，在同类项目中属上乘 |
| **训练闭环** | ★★★★★ | SFT → ReST + 数据飞轮 + hard-case 回灌，cc-haha 无此能力 |
| **安全机制** | ★★★★☆ | 三层兜底 + 预算控制 + 红队测试；可加强动态权限 |
| **可观测性** | ★★★☆☆ | 自建 Tracing 满足需求，但不如 cc-haha 的 OpenTelemetry 标准化 |

### 核心结论

1. **MedAgent 的固定拓扑 vs cc-haha 的动态编排**：医学场景需要确定性和可追溯性，MedAgent 的选择是正确的。学术研究也支持"sequential processing 最适合诊疗工作流"的结论。

2. **MedAgent 的评测体系是其最大差异化优势**：cc-haha 和绝大多数 Agent 项目都没有系统化评测。6 维评测 + 消融实验 + failure 回灌的闭环，对标了 MedAgentBench 和 AgentClinic 等学术标准。

3. **工具调用引擎是两个项目的最强交集**：MedAgent 的 `run_tool_agent()` 明确借鉴了 cc-haha 的 tool_use 模式，FC + fallback + ToolPolicy 的设计在 7B 模型上取得了显著效果。

4. **后训练能力是 MedAgent 独有的护城河**：cc-haha 完全依赖基座模型能力，而 MedAgent 的 SFT → ReST → failure 回灌形成了持续改进的数据飞轮。

---

## 参考文献

1. Gorenshtein et al. *AI Agents in Clinical Medicine: A Systematic Review*. PMC, 2025. — 20 项研究的系统综述，提出三层实施框架。
2. Jiang et al. *MedAgentBench: A Virtual EHR Environment to Benchmark Medical LLM Agents*. NEJM AI, 2025. — Stanford 标准化医学 Agent 评测框架。
3. LangChain. *Benchmarking Multi-Agent Architectures*. Blog, 2025. — Supervisor vs hierarchical vs network 架构对比。
4. Schmidgall et al. *AgentClinic*. 2024. — 24 种认知偏差扰动的医学 Agent 评测。
5. cc-haha (Claude Code 泄露源码修复版). NanmiCoder/cc-haha, GitHub, 2026.
