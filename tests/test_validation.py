"""Tests for the validation gate."""

from edgepilot.executors.base import ToolCall
from edgepilot.executors.get_time import GetTimeExecutor
from edgepilot.executors.registry import ExecutorRegistry
from edgepilot.executors.system_query import SystemQueryExecutor
from edgepilot.validation.gate import ValidationGate


class TestValidationGate:
    def setup_method(self) -> None:
        self.registry = ExecutorRegistry()
        self.registry.register(GetTimeExecutor())
        self.registry.register(SystemQueryExecutor())
        self.gate = ValidationGate(self.registry)

    def test_valid_call_passes(self) -> None:
        call = ToolCall(tool_name="get_time", arguments={})
        result = self.gate.process(call)
        assert result.success is True

    def test_invalid_args_rejected(self) -> None:
        call = ToolCall(tool_name="system_query", arguments={"query": 123})
        result = self.gate.process(call)
        # query expects str — Pydantic v2 strict-ish mode rejects int
        assert result.success is False
        assert "Validation failed" in (result.error or "")

    def test_unknown_tool_rejected(self) -> None:
        call = ToolCall(tool_name="does_not_exist", arguments={})
        result = self.gate.process(call)
        assert result.success is False
        assert "Unknown tool" in (result.error or "")

    def test_missing_required_arg_rejected(self) -> None:
        # system_query requires 'query'
        call = ToolCall(tool_name="system_query", arguments={})
        result = self.gate.process(call)
        assert result.success is False
        assert "Validation failed" in (result.error or "")
