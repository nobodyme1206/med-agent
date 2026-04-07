"""
药师 Agent：基于专科 Agent 的诊断结果，提供用药建议和药物交互检查。
使用 Function Calling 模式调用药品知识库工具。
"""

import json
import logging
from typing import Dict

from graph.state import AgentState
from tools.registry import registry
from utils.llm_client import chat

logger = logging.getLogger(__name__)

PHARMACIST_SYSTEM_PROMPT = """你是一位临床药师，负责审核用药方案的安全性和合理性。

你的职责：
1. 根据诊断结果，建议可能的治疗药物
2. 检查患者当前用药是否存在药物交互
3. 提供用药注意事项和剂量建议
4. 对特殊人群（老人、儿童、孕妇、肝肾功能不全）给出调整建议

可用工具：
- search_drug(drug_name): 查询药品详细信息
- check_drug_interaction(drug_a, drug_b): 检查两种药物的交互作用
- search_by_indication(indication): 按适应症查询可用药品

请按以下格式输出：
<think>你的分析过程</think>
<tool_call>{{"name": "工具名", "args": {{"参数名": "参数值"}}}}</tool_call>
或者（最终建议）：
<think>你的分析过程</think>
<response>你的用药建议</response>

重要提醒：
- 所有用药建议都需注明"以下建议仅供参考，具体用药请遵医嘱"
- 对于处方药，必须强调需要医生开具处方"""


def _parse_output(output: str) -> Dict:
    """解析药师输出（容错多种格式）"""
    import re
    result = {"thought": "", "tool_call": None, "response": ""}

    if "<think>" in output and "</think>" in output:
        start = output.index("<think>") + len("<think>")
        end = output.index("</think>")
        result["thought"] = output[start:end].strip()

    # 提取 tool_call（多种格式容错）
    tool_json = None
    # 格式1: <tool_call>...</tool_call>
    if "<tool_call>" in output and "</tool_call>" in output:
        start = output.index("<tool_call>") + len("<tool_call>")
        end = output.index("</tool_call>")
        tool_json = output[start:end].strip()
    # 格式2: ```json ... ``` 中包含 name+args
    if tool_json is None:
        m = re.search(r'```(?:json)?\s*(\{[^`]*"name"[^`]*"args"[^`]*\})\s*```', output, re.DOTALL)
        if m:
            tool_json = m.group(1).strip()
    # 格式3: 裸 JSON {"name": ..., "args": ...}
    if tool_json is None:
        m = re.search(r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\})', output)
        if m:
            tool_json = m.group(1).strip()

    if tool_json:
        try:
            result["tool_call"] = json.loads(tool_json)
        except json.JSONDecodeError:
            logger.warning(f"Pharmacist: tool_call JSON 解析失败: {tool_json[:100]}")

    if "<response>" in output and "</response>" in output:
        start = output.index("<response>") + len("<response>")
        end = output.index("</response>")
        result["response"] = output[start:end].strip()

    if not result["response"] and not result["tool_call"]:
        result["response"] = output.strip()

    return result


def _execute_tool(tool_call: Dict) -> str:
    """执行工具调用"""
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    try:
        result = registry.call(name, **args)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def pharmacist_review(state: AgentState) -> AgentState:
    """
    药师 Agent 节点：审核用药方案，提供药物建议。

    流程：
    1. 综合患者信息和专科分析
    2. 查询相关药物信息
    3. 检查药物交互
    4. 输出用药建议
    """
    patient_info = state.get("patient_info", "")
    specialist_analysis = state.get("specialist_analysis", "")
    department = state.get("current_department", "")
    new_tool_calls = []

    # 从对话中提取已知用药信息
    all_text = patient_info + " " + " ".join(
        m["content"] for m in state.get("messages", [])
    )

    prompt = (
        f"【患者信息】{patient_info}\n"
        f"【就诊科室】{department}\n"
        f"【专科医生分析】{specialist_analysis}\n\n"
        f"请基于以上信息，提供用药评估和建议。"
        f"如果提到了具体药品，请先查询药品信息。"
        f"如果患者正在服用多种药物，请检查药物交互。"
    )

    system = PHARMACIST_SYSTEM_PROMPT
    response = chat(prompt, system=system, temperature=0.2, max_tokens=1024)

    if not response:
        logger.error("Pharmacist: LLM 调用失败")
        return {
            "drug_advice": "用药评估暂时不可用，请咨询线下药师。",
        }

    parsed = _parse_output(response)

    # 运行时配置（消融实验支持）
    from graph.workflow import _runtime_config
    tools_enabled = _runtime_config.get("use_tools", True)

    # 工具调用循环（最多 3 轮）
    for round_idx in range(3):
        if not parsed["tool_call"]:
            break

        tool_call = parsed["tool_call"]

        # 消融：工具关闭时跳过
        if not tools_enabled:
            logger.info(f"Pharmacist: 工具已禁用，跳过 {tool_call.get('name')}")
            break

        logger.info(f"Pharmacist: 工具调用 [{round_idx+1}] {tool_call.get('name')}")

        tool_result = _execute_tool(tool_call)
        # 检查 JSON 顶层 error key 判断成功与否（避免内容含 "error" 字样的误判）
        _tool_success = True
        try:
            _parsed_result = json.loads(tool_result)
            if isinstance(_parsed_result, dict) and "error" in _parsed_result:
                _tool_success = False
        except (json.JSONDecodeError, TypeError):
            pass
        new_tool_calls.append({
            "tool_name": tool_call.get("name", ""),
            "input_args": tool_call.get("args", {}),
            "output": tool_result,
            "success": _tool_success,
        })

        followup = (
            f"{prompt}\n\n"
            f"【你之前的思考】{parsed['thought']}\n"
            f"【工具调用】{json.dumps(tool_call, ensure_ascii=False)}\n"
            f"【工具返回】{tool_result}\n\n"
            f"请根据工具结果继续分析，如需查询更多药物信息可继续调用工具，否则给出最终建议。"
        )
        response = chat(followup, system=system, temperature=0.2, max_tokens=1024)
        if not response:
            break

        parsed = _parse_output(response)

    drug_advice = parsed["response"]
    logger.info(f"Pharmacist: 用药建议生成完成")

    return {
        "drug_advice": drug_advice,
        "tool_calls": new_tool_calls,
    }
