"""
统一工具调用引擎（参考 cc-haha 的原生 tool_use 模式）。

核心思想：
  - 优先使用 OpenAI Function Calling API，由 API 层结构化返回工具调用
  - 当 API 不支持 function calling 时，自动回退到文本解析模式
  - 所有 Agent（Specialist、Pharmacist）共用此引擎，消除重复代码
  - 工具 schema 从 ToolRegistry 自动获取，不需要 Agent 手写

用法：
    from utils.tool_agent import run_tool_agent
    result = run_tool_agent(
        system_prompt="你是一位...",
        user_prompt="患者信息...",
        tool_names=["search_guidelines", "interpret_lab_result"],
        tools_enabled=True,
        rag_enabled=True,
    )
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from graph.state import TOKEN_BUDGET
from tools.registry import registry
from utils.llm_client import chat, chat_with_messages, get_token_usage

logger = logging.getLogger(__name__)

# 默认使用文本模式（与 SFT 训练数据的 <tool_call> 格式一致）
# 设置环境变量 TOOL_AGENT_PREFER_FC=1 可切换为 FC 模式
_PREFER_FC = os.getenv("TOOL_AGENT_PREFER_FC", "0").strip() in ("1", "true", "yes")

# ─── 文本模式 fallback 提示词片段 ───
_TEXT_TOOL_INSTRUCTION = (
    "请使用 <think>...</think> 标签展示思考过程，"
    "使用 <tool_call>...</tool_call> 调用工具，"
    "使用 <response>...</response> 给出最终回复。"
)


def _build_text_tool_description(tool_names: List[str]) -> str:
    """从 Registry 生成文本模式的工具描述（告诉模型有哪些工具可用）"""
    schemas = registry.get_openai_tools(filter_names=tool_names)
    if not schemas:
        return ""
    lines = ["\n可用工具："]
    for s in schemas:
        func = s["function"]
        params = func.get("parameters", {}).get("properties", {})
        param_desc = ", ".join(f"{k}: {v.get('description', '')}" for k, v in params.items())
        lines.append(f"- {func['name']}({param_desc}): {func['description']}")
    lines.append("")
    lines.append('调用格式：<tool_call>{{"name": "工具名", "args": {{"参数名": "参数值"}}}}</tool_call>')
    return "\n".join(lines)


def _execute_tool(name: str, args: Dict) -> str:
    """通过 Registry 执行工具调用，返回 JSON 字符串"""
    try:
        result = registry.call(name, **args)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _check_tool_success(tool_result_str: str) -> bool:
    """检查工具结果是否成功（顶层 error key）"""
    try:
        parsed = json.loads(tool_result_str)
        if isinstance(parsed, dict) and "error" in parsed:
            return False
    except (json.JSONDecodeError, TypeError):
        pass
    return True


def _normalize_args(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_args(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize_args(v) for v in value]
    return value


def _make_call_signature(name: str, args: Dict) -> str:
    normalized = _normalize_args(args or {})
    return f"{name}:{json.dumps(normalized, ensure_ascii=False, sort_keys=True)}"


def _summarize_tool_result(tool_result_str: str, max_length: int = 600) -> str:
    try:
        parsed = json.loads(tool_result_str)
    except (json.JSONDecodeError, TypeError):
        return tool_result_str[:max_length]

    if isinstance(parsed, dict):
        compact = {}
        for key in ("summary", "result", "interpretation", "recommendation", "risk_level", "found"):
            if key in parsed:
                compact[key] = parsed[key]
        if "chunks" in parsed and isinstance(parsed["chunks"], list):
            compact["chunks"] = parsed["chunks"][:2]
        if "drug" in parsed:
            compact["drug"] = parsed["drug"]
        if not compact:
            compact = parsed
        return json.dumps(compact, ensure_ascii=False)[:max_length]

    if isinstance(parsed, list):
        return json.dumps(parsed[:3], ensure_ascii=False)[:max_length]

    return str(parsed)[:max_length]


def _check_runtime_budget() -> Optional[str]:
    usage = get_token_usage()
    if usage > TOKEN_BUDGET:
        return f"token_budget_exceeded:{usage}>{TOKEN_BUDGET}"
    return None


def _tool_policy(
    tool_name: str,
    args: Dict,
    allowed_tool_names: List[str],
    executed_calls: List[Dict],
    tools_enabled: bool,
    rag_enabled: bool,
    max_total_tool_calls: Optional[int],
    max_calls_per_tool: Optional[int],
) -> Tuple[bool, str, str]:
    signature = _make_call_signature(tool_name, args)

    if not tools_enabled:
        return False, "工具已禁用", signature
    if tool_name not in allowed_tool_names:
        return False, "工具不在规划列表内", signature
    if not rag_enabled and tool_name == "search_guidelines":
        return False, "RAG 已禁用", signature
    if max_total_tool_calls is not None and len(executed_calls) >= max_total_tool_calls:
        return False, "达到总工具调用上限", signature

    same_tool_count = sum(1 for tc in executed_calls if tc.get("tool_name") == tool_name)
    if max_calls_per_tool is not None and same_tool_count >= max_calls_per_tool:
        return False, f"工具 {tool_name} 达到单工具调用上限", signature

    if any(tc.get("call_signature") == signature for tc in executed_calls):
        return False, f"重复调用 {tool_name} 被抑制", signature

    return True, "", signature


def _parse_text_output(output: str) -> Dict:
    """解析文本模式的结构化输出（兼容训练数据的多种格式）"""
    import re
    result = {"thought": "", "tool_call": None, "response": ""}

    # 提取 think（兼容双层嵌套 <think><think>...</think></think>）
    if "<think>" in output and "</think>" in output:
        start = output.index("<think>") + len("<think>")
        end = output.rindex("</think>")
        thought = output[start:end].strip()
        thought = thought.replace("<think>", "").replace("</think>", "").strip()
        result["thought"] = thought

    # 提取 tool_call（多种格式容错）
    tool_json = None
    if "<tool_call>" in output and "</tool_call>" in output:
        start = output.index("<tool_call>") + len("<tool_call>")
        end = output.index("</tool_call>")
        tool_json = output[start:end].strip()
    if tool_json is None:
        m = re.search(r'```(?:json)?\s*(\{[^`]*"name"[^`]*"args"[^`]*\})\s*```', output, re.DOTALL)
        if m:
            tool_json = m.group(1).strip()
    if tool_json is None:
        m = re.search(r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\})', output)
        if m:
            tool_json = m.group(1).strip()

    if tool_json:
        try:
            result["tool_call"] = json.loads(tool_json)
        except json.JSONDecodeError:
            logger.warning(f"ToolAgent: tool_call JSON 解析失败: {tool_json[:100]}")

    if "<response>" in output and "</response>" in output:
        start = output.index("<response>") + len("<response>")
        end = output.index("</response>")
        result["response"] = output[start:end].strip()

    if not result["response"] and not result["tool_call"]:
        result["response"] = output.strip()

    return result


# ─── Function Calling 模式 ───

def _run_fc_mode(
    system_prompt: str,
    user_prompt: str,
    tool_schemas: List[Dict],
    allowed_tool_names: List[str],
    tools_enabled: bool = True,
    rag_enabled: bool = True,
    agent_name: str = "agent",
    max_rounds: int = 3,
    max_total_tool_calls: Optional[int] = None,
    max_calls_per_tool: Optional[int] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[Dict]:
    """
    用 OpenAI Function Calling API 执行。
    成功返回 {"response", "tool_calls", "knowledge"}；
    API 不支持时返回 None。
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    new_tool_calls = []
    retrieved_knowledge = []
    stop_reason = ""

    for round_idx in range(max_rounds):
        budget_reason = _check_runtime_budget()
        if budget_reason:
            stop_reason = budget_reason
            break

        # 第一轮传工具 schema 让模型决定是否调用；后续轮次已有工具结果，不再传工具
        use_tools = tool_schemas if (round_idx == 0 and tool_schemas) else None
        try:
            result = chat_with_messages(
                msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=use_tools,
            )
        except Exception as e:
            logger.warning(f"ToolAgent FC: API 调用失败: {e}")
            return None

        if result is None:
            return None

        content = result.get("content", "")
        api_tool_calls = result.get("tool_calls")

        # 没有工具调用 → 最终回复
        if not api_tool_calls:
            return {
                "response": content,
                "tool_calls": new_tool_calls,
                "knowledge": retrieved_knowledge,
                "stop_reason": stop_reason,
            }

        tool_results_text_parts = []

        for tc in api_tool_calls:
            func = tc.get("function", {})
            fname = func.get("name", "")
            try:
                fargs = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                fargs = {}

            allow, deny_reason, signature = _tool_policy(
                tool_name=fname,
                args=fargs,
                allowed_tool_names=allowed_tool_names,
                executed_calls=new_tool_calls,
                tools_enabled=tools_enabled,
                rag_enabled=rag_enabled,
                max_total_tool_calls=max_total_tool_calls,
                max_calls_per_tool=max_calls_per_tool,
            )
            if not allow:
                logger.info(f"ToolAgent FC: 跳过 {fname}({fargs}) - {deny_reason}")
                tool_results_text_parts.append(f"【工具策略】已跳过 {fname}：{deny_reason}")
                continue

            logger.info(f"ToolAgent FC: 工具调用 [{round_idx+1}] {fname}({fargs})")

            tool_result = _execute_tool(fname, fargs)
            _success = _check_tool_success(tool_result)

            new_tool_calls.append({
                "tool_name": fname,
                "input_args": fargs,
                "output": tool_result,
                "success": _success,
                "agent_name": agent_name,
                "round_id": round_idx + 1,
                "call_signature": signature,
                "skipped_reason": "",
            })

            if fname == "search_guidelines":
                try:
                    rag_result = json.loads(tool_result)
                    if rag_result.get("found"):
                        retrieved_knowledge.extend(rag_result.get("chunks", []))
                except json.JSONDecodeError:
                    pass

            tool_results_text_parts.append(
                f"【工具 {fname} 返回】\n{_summarize_tool_result(tool_result)}"
            )

        if not tool_results_text_parts:
            stop_reason = stop_reason or "all_tool_calls_blocked"
            break

        msgs.append({"role": "assistant", "content": content or f"（调用了工具：{', '.join(tc.get('function',{}).get('name','') for tc in api_tool_calls)}）"})
        msgs.append({"role": "user", "content": "\n\n".join(tool_results_text_parts) + "\n\n请根据工具返回的结果，继续分析并给出回复。"})

    if stop_reason.startswith("token_budget_exceeded"):
        return {
            "response": "",
            "tool_calls": new_tool_calls,
            "knowledge": retrieved_knowledge,
            "stop_reason": stop_reason,
        }

    try:
        final = chat_with_messages(msgs, temperature=temperature, max_tokens=max_tokens)
        return {
            "response": final.get("content", "") if final else "",
            "tool_calls": new_tool_calls,
            "knowledge": retrieved_knowledge,
            "stop_reason": stop_reason or ("max_rounds_reached" if max_rounds > 0 else ""),
        }
    except Exception:
        return {
            "response": "",
            "tool_calls": new_tool_calls,
            "knowledge": retrieved_knowledge,
            "stop_reason": stop_reason,
        }


# ─── 文本解析模式（fallback）───

def _run_text_mode(
    system_prompt: str,
    user_prompt: str,
    allowed_tool_names: List[str],
    tools_enabled: bool = True,
    rag_enabled: bool = True,
    agent_name: str = "agent",
    max_rounds: int = 3,
    max_total_tool_calls: Optional[int] = None,
    max_calls_per_tool: Optional[int] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Dict:
    """文本模式：用 <tool_call> 标签解析工具调用"""
    response = chat(user_prompt, system=system_prompt, temperature=temperature, max_tokens=max_tokens)
    if not response:
        return {"response": "", "tool_calls": [], "knowledge": [], "stop_reason": ""}

    parsed = _parse_text_output(response)
    all_thoughts = [parsed["thought"]] if parsed["thought"] else []
    new_tool_calls = []
    retrieved_knowledge = []
    stop_reason = ""

    for round_idx in range(max_rounds):
        budget_reason = _check_runtime_budget()
        if budget_reason:
            stop_reason = budget_reason
            break

        if not parsed["tool_call"]:
            break

        tool_call = parsed["tool_call"]
        tname = tool_call.get("name", "")

        allow, deny_reason, signature = _tool_policy(
            tool_name=tname,
            args=tool_call.get("args", {}),
            allowed_tool_names=allowed_tool_names,
            executed_calls=new_tool_calls,
            tools_enabled=tools_enabled,
            rag_enabled=rag_enabled,
            max_total_tool_calls=max_total_tool_calls,
            max_calls_per_tool=max_calls_per_tool,
        )
        if not allow:
            logger.info(f"ToolAgent text: 跳过 {tname} - {deny_reason}")
            followup = (
                f"{user_prompt}\n\n"
                f"【你之前的思考】{parsed['thought']}\n"
                f"【工具策略】{deny_reason}\n\n"
                f"请在当前限制下继续分析并给出回复。"
            )
            response = chat(followup, system=system_prompt, temperature=temperature, max_tokens=max_tokens)
            if not response:
                stop_reason = stop_reason or "tool_call_blocked"
                break
            parsed = _parse_text_output(response)
            continue

        logger.info(f"ToolAgent text: 工具调用 [{round_idx+1}] {tname}")

        tool_result = _execute_tool(tname, tool_call.get("args", {}))
        _success = _check_tool_success(tool_result)

        new_tool_calls.append({
            "tool_name": tname,
            "input_args": tool_call.get("args", {}),
            "output": tool_result,
            "success": _success,
            "agent_name": agent_name,
            "round_id": round_idx + 1,
            "call_signature": signature,
            "skipped_reason": "",
        })

        if tname == "search_guidelines":
            try:
                rag_result = json.loads(tool_result)
                if rag_result.get("found"):
                    retrieved_knowledge.extend(rag_result.get("chunks", []))
            except json.JSONDecodeError:
                pass

        followup = (
            f"{user_prompt}\n\n"
            f"【你之前的思考】{parsed['thought']}\n"
            f"【工具调用】{json.dumps(tool_call, ensure_ascii=False)}\n"
            f"【工具返回】{_summarize_tool_result(tool_result)}\n\n"
            f"请根据工具返回的结果，继续分析并给出回复。"
        )
        response = chat(followup, system=system_prompt, temperature=temperature, max_tokens=max_tokens)
        if not response:
            break

        parsed = _parse_text_output(response)
        if parsed["thought"]:
            all_thoughts.append(parsed["thought"])

    return {
        "response": parsed["response"],
        "tool_calls": new_tool_calls,
        "knowledge": retrieved_knowledge,
        "stop_reason": stop_reason,
    }


# ─── 统一入口 ───

def run_tool_agent(
    system_prompt: str,
    user_prompt: str,
    tool_names: List[str],
    tools_enabled: bool = True,
    rag_enabled: bool = True,
    agent_name: str = "agent",
    max_rounds: int = 3,
    max_total_tool_calls: Optional[int] = None,
    max_calls_per_tool: Optional[int] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    prefer_fc: Optional[bool] = None,
) -> Dict:
    """
    统一工具调用引擎入口。

    Args:
        system_prompt: Agent 的系统提示词
        user_prompt: 用户 prompt（含患者信息、对话记录等）
        tool_names: 该 Agent 可使用的工具名称列表
        tools_enabled: 是否启用工具（消融实验用）
        rag_enabled: 是否启用 RAG（消融实验用）
        max_rounds: 最大工具调用轮数
        temperature: 采样温度
        max_tokens: 最大生成长度
        prefer_fc: 是否优先使用 FC 模式，None 时读取环境变量/默认 text 模式

    Returns:
        {"response": str, "tool_calls": list, "knowledge": list}
    """
    use_fc = prefer_fc if prefer_fc is not None else _PREFER_FC

    # 构建工具 schema（从 Registry 自动获取）
    active_tool_names = []
    if tools_enabled:
        for name in tool_names:
            if not rag_enabled and name == "search_guidelines":
                continue
            active_tool_names.append(name)

    tool_schemas = registry.get_openai_tools(filter_names=active_tool_names) if active_tool_names else []

    if use_fc and tool_schemas:
        # FC 模式：依赖 API 层结构化 tool_calls
        fc_result = _run_fc_mode(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tool_schemas=tool_schemas,
            allowed_tool_names=active_tool_names,
            tools_enabled=tools_enabled,
            rag_enabled=rag_enabled,
            agent_name=agent_name,
            max_rounds=max_rounds,
            max_total_tool_calls=max_total_tool_calls,
            max_calls_per_tool=max_calls_per_tool,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if fc_result is not None:
            logger.info(f"ToolAgent: FC 模式完成 (工具调用={len(fc_result['tool_calls'])})")
            return fc_result
        logger.info("ToolAgent: FC 模式失败，回退到文本模式")

    # 文本解析模式（默认）：与 SFT 训练数据的 <tool_call> 格式一致
    logger.info(f"ToolAgent: 使用文本模式 (可用工具={active_tool_names})")
    tool_desc = _build_text_tool_description(active_tool_names) if active_tool_names else ""
    text_system = system_prompt + tool_desc + "\n" + _TEXT_TOOL_INSTRUCTION
    return _run_text_mode(
        system_prompt=text_system,
        user_prompt=user_prompt,
        allowed_tool_names=active_tool_names,
        tools_enabled=tools_enabled,
        rag_enabled=rag_enabled,
        agent_name=agent_name,
        max_rounds=max_rounds,
        max_total_tool_calls=max_total_tool_calls,
        max_calls_per_tool=max_calls_per_tool,
        temperature=temperature,
        max_tokens=max_tokens,
    )
