"""
专科 Agent：基于分诊结果进行问诊推理，调用 RAG 和检验解读工具辅助诊断。
使用 ReAct（Reasoning + Acting）模式。
"""

import json
import logging
from typing import Dict, List

from graph.state import AgentState
from tools.registry import registry
from utils.llm_client import chat

logger = logging.getLogger(__name__)

SPECIALIST_SYSTEM_PROMPT = """你是一位{department}的资深主治医师，正在进行门诊问诊。

你的工作流程（ReAct模式）：
1. **思考(Thought)**：分析患者当前信息，判断需要什么额外信息或检查
2. **行动(Action)**：如果需要查询医学知识，调用工具；如果信息充足，给出诊断建议
3. **观察(Observation)**：分析工具返回的结果
4. **回复(Response)**：基于所有信息给出专业、有温度的回复

可用工具：
- search_guidelines(query): 检索诊疗指南和医学知识
- interpret_lab_result(test_name, value, unit): 解读检验结果

规则：
- 如果患者提供了检验数值，务必调用 interpret_lab_result 进行解读
- 对不确定的诊断，主动建议进一步检查
- 回复要专业但通俗易懂
- 不要直接开处方，用药建议交给药师Agent

请按以下格式输出：
<think>你的思考过程</think>
<tool_call>{{"name": "工具名", "args": {{"参数名": "参数值"}}}}</tool_call>
或者（如果不需要工具）：
<think>你的思考过程</think>
<response>你的回复内容</response>"""


def _parse_specialist_output(output: str) -> Dict:
    """解析专科 Agent 的结构化输出（容错多种格式）"""
    import re
    result = {"thought": "", "tool_call": None, "response": ""}

    # 提取 think
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
            logger.warning(f"Specialist: tool_call JSON 解析失败: {tool_json[:100]}")

    # 提取 response
    if "<response>" in output and "</response>" in output:
        start = output.index("<response>") + len("<response>")
        end = output.index("</response>")
        result["response"] = output[start:end].strip()

    # 如果没有结构化输出，把整个输出当作 response
    if not result["response"] and not result["tool_call"]:
        result["response"] = output.strip()

    return result


def _execute_tool(tool_call: Dict) -> str:
    """执行工具调用并返回结果字符串"""
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    try:
        result = registry.call(name, **args)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def specialist_analyze(state: AgentState) -> AgentState:
    """
    专科 Agent 节点：基于科室进行专业问诊推理。

    流程：
    1. 构建包含患者信息和对话历史的 prompt
    2. LLM 推理（可能触发工具调用）
    3. 如果有工具调用，执行工具并将结果反馈给 LLM
    4. 输出分析结论
    """
    department = state.get("current_department", "内科")
    patient_info = state.get("patient_info", "")
    messages = state.get("messages", [])
    new_tool_calls = []
    retrieved_knowledge = list(state.get("retrieved_knowledge", []))

    # 构建对话上下文
    conversation = ""
    for msg in messages:
        role = "患者" if msg.get("role") == "user" else "医生"
        conversation += f"{role}：{msg['content']}\n"

    system = SPECIALIST_SYSTEM_PROMPT.format(department=department)
    prompt = (
        f"【患者信息】{patient_info}\n\n"
        f"【对话记录】\n{conversation}\n"
        f"请进行分析并给出回复。"
    )

    # 第一轮 LLM 推理
    response = chat(prompt, system=system, temperature=0.3, max_tokens=1024)
    if not response:
        logger.error("Specialist: LLM 调用失败")
        return {
            "specialist_analysis": "抱歉，系统暂时无法分析，请稍后重试。",
        }

    parsed = _parse_specialist_output(response)
    all_thoughts = [parsed["thought"]] if parsed["thought"] else []

    # 运行时配置（消融实验支持）
    from graph.workflow import _runtime_config
    tools_enabled = _runtime_config.get("use_tools", True)
    rag_enabled = _runtime_config.get("use_rag", True)

    # 工具调用循环（最多 3 轮）
    max_tool_rounds = 3
    for round_idx in range(max_tool_rounds):
        if not parsed["tool_call"]:
            break

        tool_call = parsed["tool_call"]

        # 消融：工具关闭时跳过所有工具调用
        if not tools_enabled:
            logger.info(f"Specialist: 工具已禁用，跳过 {tool_call.get('name')}")
            break
        # 消融：RAG 关闭时跳过 guideline 检索
        if not rag_enabled and tool_call.get("name") == "search_guidelines":
            logger.info("Specialist: RAG 已禁用，跳过 search_guidelines")
            parsed = {"thought": parsed["thought"], "tool_call": None, "response": ""}
            continue

        logger.info(f"Specialist: 工具调用 [{round_idx+1}] {tool_call.get('name')}")

        # 执行工具
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

        # 如果是 RAG 检索，保存知识块
        if tool_call.get("name") == "search_guidelines":
            try:
                rag_result = json.loads(tool_result)
                if rag_result.get("found"):
                    retrieved_knowledge.extend(rag_result.get("chunks", []))
            except json.JSONDecodeError:
                pass

        # 将工具结果反馈给 LLM
        followup_prompt = (
            f"{prompt}\n\n"
            f"【你之前的思考】{parsed['thought']}\n"
            f"【工具调用】{json.dumps(tool_call, ensure_ascii=False)}\n"
            f"【工具返回】{tool_result}\n\n"
            f"请根据工具返回的结果，继续分析并给出回复。"
        )
        response = chat(followup_prompt, system=system, temperature=0.3, max_tokens=1024)
        if not response:
            break

        parsed = _parse_specialist_output(response)
        if parsed["thought"]:
            all_thoughts.append(parsed["thought"])

    # 最终分析结论
    analysis = parsed["response"]
    thought_chain = " → ".join(all_thoughts) if all_thoughts else ""

    logger.info(f"Specialist: 分析完成 (思考链长度={len(all_thoughts)}, 工具调用={len(new_tool_calls)})")

    return {
        "specialist_analysis": analysis,
        "tool_calls": new_tool_calls,
        "retrieved_knowledge": retrieved_knowledge,
        "messages": [{"role": "assistant", "content": f"[{department}医生思考] {thought_chain}"}],
    }
