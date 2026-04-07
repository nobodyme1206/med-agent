"""
工具层单元测试：验证药品查询、检验值解读、工具注册中心。
不依赖 API，仅测试本地数据和逻辑。

用法: python -m pytest tests/test_tools.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.drug_lookup import search_drug, check_drug_interaction, search_by_indication, _load_db
from tools.lab_interpreter import interpret_lab_result, interpret_lab_panel, _load_ranges
from tools.registry import ToolRegistry, ToolDefinition


class TestDrugLookup:

    def test_search_by_generic_name(self):
        result = search_drug("二甲双胍")
        assert result["found"] is True
        assert result["generic_name"] == "二甲双胍"

    def test_search_not_found(self):
        result = search_drug("不存在的药品xyz")
        assert result["found"] is False

    def test_search_returns_required_fields(self):
        result = search_drug("阿莫西林")
        assert result["found"] is True
        for field in ["generic_name", "category", "indications",
                      "contraindications", "adverse_reactions", "dosage"]:
            assert field in result

    def test_drug_interaction_missing_drug(self):
        result = check_drug_interaction("不存在的药", "阿莫西林")
        assert result["has_interaction"] is False

    def test_search_by_indication(self):
        results = search_by_indication("高血压")
        assert len(results) >= 3

    def test_search_by_indication_not_found(self):
        results = search_by_indication("不存在的病xyz")
        assert len(results) == 1
        assert "message" in results[0]

    def test_expanded_drug_count(self):
        db = _load_db()
        assert len(db) >= 50

    def test_new_drug_categories(self):
        for name in ["缬沙坦", "达格列净", "舍曲林", "利伐沙班"]:
            result = search_drug(name)
            assert result["found"] is True, f"{name} not found"


class TestLabInterpreter:

    def test_normal_value(self):
        result = interpret_lab_result("血红蛋白", 140)
        assert result["found"] is True
        assert result["status"] == "正常"

    def test_high_value(self):
        result = interpret_lab_result("空腹血糖", 8.5)
        assert result["found"] is True
        assert result["status"] == "偏高"
        assert len(result["possible_causes"]) > 0

    def test_low_value(self):
        result = interpret_lab_result("血红蛋白", 90)
        assert result["found"] is True
        assert result["status"] == "偏低"

    def test_alias_matching(self):
        result = interpret_lab_result("WBC", 12)
        assert result["found"] is True
        assert result["status"] == "偏高"

    def test_not_found(self):
        result = interpret_lab_result("不存在的指标", 100)
        assert result["found"] is False

    def test_returns_required_fields(self):
        result = interpret_lab_result("ALT", 60)
        assert result["found"] is True
        for field in ["test_name", "value", "unit", "reference_range",
                      "status", "possible_causes", "clinical_significance"]:
            assert field in result

    def test_panel_interpretation(self):
        panel = [
            {"test_name": "血红蛋白", "value": 85},
            {"test_name": "空腹血糖", "value": 5.2},
            {"test_name": "ALT", "value": 80},
        ]
        results = interpret_lab_panel(panel)
        assert len(results) == 3
        assert results[0]["status"] == "偏低"
        assert results[1]["status"] == "正常"
        assert results[2]["status"] == "偏高"

    def test_expanded_lab_count(self):
        ranges = _load_ranges()
        assert len(ranges) >= 40


class TestToolRegistry:

    def test_register_and_list(self):
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool", description="测试",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "ok", category="test",
        )
        reg.register(tool)
        assert "test_tool" in reg.list_tools()

    def test_call_success(self):
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="echo", description="echo",
            parameters={"type": "object", "properties": {}},
            handler=lambda msg="": f"echo: {msg}",
        )
        reg.register(tool)
        assert reg.call("echo", msg="hello") == "echo: hello"

    def test_call_not_registered(self):
        reg = ToolRegistry()
        try:
            reg.call("not_registered")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_call_failure_returns_error(self):
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="fail", description="fail",
            parameters={"type": "object", "properties": {}},
            handler=lambda: (_ for _ in ()).throw(RuntimeError("test")),
        )
        reg.register(tool)
        result = reg.call("fail")
        assert "error" in result

    def test_call_stats(self):
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="s", description="s",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "ok",
        )
        reg.register(tool)
        reg.call("s")
        stats = reg.get_call_stats()
        assert stats["total_calls"] == 1
        assert stats["success_rate"] == 1.0

    def test_openai_format(self):
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="my_tool", description="desc",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            handler=lambda: None,
        )
        reg.register(tool)
        tools = reg.get_openai_tools()
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "my_tool"

    def test_mcp_format(self):
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="mcp", description="d",
            parameters={"type": "object", "properties": {}},
            handler=lambda: None,
        )
        reg.register(tool)
        tools = reg.get_mcp_tools()
        assert "inputSchema" in tools[0]

    def test_reset_history(self):
        reg = ToolRegistry()
        tool = ToolDefinition(
            name="r", description="r",
            parameters={"type": "object", "properties": {}},
            handler=lambda: None,
        )
        reg.register(tool)
        reg.call("r")
        reg.reset_history()
        assert len(reg.get_call_history()) == 0
