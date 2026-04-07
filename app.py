"""
MedAgent Gradio Demo：医学多智能体问诊系统交互界面。

左侧：多轮对话（患者 ↔ Agent）
右侧：Agent 内部状态实时展示（科室、思考链、工具调用、置信度）
底部：会话统计

用法:
  python app.py [--port 7860] [--share]
"""

import sys
import json
import time
import uuid
import logging
import argparse
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
from tools.setup import setup_tools
from graph.workflow import run_consultation
from memory.short_term import ShortTermMemory
from monitoring.metrics import tracker
from monitoring.alerts import alert_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 初始化工具
setup_tools()

# ─────────────────────────────────────────────
# 核心逻辑
# ─────────────────────────────────────────────

memory = ShortTermMemory(window_size=10)


def process_message(user_message: str, chat_history: list):
    """处理用户消息并返回 Agent 回复 + 状态信息"""
    if not user_message.strip():
        return chat_history, "", "", "", ""

    # 追踪指标
    request_id = str(uuid.uuid4())[:8]
    tracker.start_request(request_id)

    # 构建历史：优先使用 ShortTermMemory（含滑动窗口 + LLM 摘要压缩）
    history_messages = memory.get_messages()

    # 运行 Agent
    start = time.time()
    try:
        result = run_consultation(user_message, history=history_messages)
    except Exception as e:
        logger.error(f"Agent 运行失败: {e}")
        result = {
            "final_response": f"抱歉，系统遇到错误：{str(e)}。请稍后重试。",
            "current_department": "未知",
            "tool_calls": [],
            "confidence": 0.0,
            "should_escalate": True,
        }
    latency = (time.time() - start) * 1000

    # 结束追踪
    tracker.end_request(result)

    # 告警检查
    stats = tracker.get_aggregate_stats()
    alerts = alert_manager.check(stats)
    for alert in alerts:
        logger.warning(f"Alert: {alert}")

    # 提取信息
    final_response = result.get("final_response", "系统暂时无法回复")
    department = result.get("current_department", "")
    confidence = result.get("confidence", 0.0)
    tool_calls = result.get("tool_calls", [])
    should_escalate = result.get("should_escalate", False)
    specialist_analysis = result.get("specialist_analysis", "")

    # 更新记忆
    memory.add_message("user", user_message)
    memory.add_message("assistant", final_response)

    # 更新对话（Gradio 6.x message dict 格式）
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": final_response})

    # 状态面板信息
    dept_info = f"**科室**: {department}"
    if should_escalate:
        dept_info += "\n\n⚠️ **已标记需要人工介入**"
    dept_info += f"\n\n**置信度**: {confidence:.0%}"
    dept_info += f"\n\n**延迟**: {latency:.0f}ms"

    # 思考链
    thought_text = ""
    if specialist_analysis:
        thought_text = specialist_analysis[:500]

    # 工具调用
    tool_text = ""
    if tool_calls:
        for i, tc in enumerate(tool_calls, 1):
            name = tc.get("tool_name", "")
            args = tc.get("input_args", {})
            success = "✅" if tc.get("success") else "❌"
            tool_text += f"{i}. {success} **{name}**\n"
            tool_text += f"   参数: `{json.dumps(args, ensure_ascii=False)}`\n\n"
    else:
        tool_text = "本轮未调用工具"

    # 统计
    stats = tracker.get_aggregate_stats()
    stats_text = (
        f"**总请求**: {stats.get('total_requests', 0)} | "
        f"**平均延迟**: {stats.get('avg_latency_ms', 0):.0f}ms | "
        f"**工具成功率**: {stats.get('tool_success_rate', 0):.0%} | "
        f"**升级率**: {stats.get('escalation_rate', 0):.0%}"
    )

    return chat_history, dept_info, thought_text, tool_text, stats_text


def clear_all():
    """清空对话和记忆"""
    memory.clear()
    tracker.reset_history()
    return [], "", "", "", ""


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

EXAMPLES = [
    "我最近总是头晕，血压偏高，吃了硝苯地平但没效果",
    "我孩子3岁，发烧39度两天了，还有点咳嗽",
    "空腹血糖7.8mmol/L，糖化血红蛋白7.2%，吃二甲双胍胃不舒服",
    "突然胸口剧烈疼痛，出了一身冷汗",
    "反复上腹痛一个月，餐后加重，有反酸",
    "腰痛两周，弯腰时加重，左腿有时麻",
]


_CUSTOM_CSS = """
.status-panel {padding: 12px; border-radius: 8px; background: #f8f9fa;}
.stats-bar {padding: 8px 16px; background: #e9ecef; border-radius: 6px; margin-top: 8px;}
"""

def build_demo():
    with gr.Blocks(
        title="MedAgent - 医学多智能体问诊系统",
    ) as demo:
        gr.Markdown(
            "# 🏥 MedAgent — 医学多智能体问诊系统\n"
            "基于 LangGraph + ReAct + 多 Agent 协作的端到端医学问诊 Demo。"
            "输入症状描述开始问诊。\n\n"
            "⚠️ *本系统仅供演示，不构成医疗建议。如有不适请及时就医。*"
        )

        with gr.Row():
            # 左侧：对话区
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="问诊对话",
                    height=480,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="请描述您的症状...",
                        label="输入",
                        scale=5,
                        show_label=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空", scale=1)

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=msg_input,
                    label="示例问题",
                )

            # 右侧：状态面板
            with gr.Column(scale=2):
                dept_display = gr.Markdown(
                    value="等待问诊...",
                    label="分诊信息",
                    elem_classes=["status-panel"],
                )
                thought_display = gr.Textbox(
                    label="专科分析",
                    lines=6,
                    interactive=False,
                )
                tool_display = gr.Markdown(
                    value="等待问诊...",
                    label="工具调用记录",
                )

        # 底部统计
        stats_display = gr.Markdown(
            value="暂无统计数据",
            elem_classes=["stats-bar"],
        )

        # 事件绑定
        send_btn.click(
            process_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, dept_display, thought_display, tool_display, stats_display],
        ).then(lambda: "", outputs=msg_input)

        msg_input.submit(
            process_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, dept_display, thought_display, tool_display, stats_display],
        ).then(lambda: "", outputs=msg_input)

        clear_btn.click(
            clear_all,
            outputs=[chatbot, dept_display, thought_display, tool_display, stats_display],
        )

    return demo


# ─────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=_CUSTOM_CSS,
    )
