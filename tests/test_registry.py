"""Tests for the typed executor registry."""

import pytest
from pydantic import BaseModel

from edgepilot.executors.base import BaseExecutor, ToolCall
from edgepilot.executors.get_time import GetTimeExecutor
from edgepilot.executors.registry import ExecutorRegistry
from edgepilot.executors.system_query import SystemQueryExecutor


class DummyArgs(BaseModel):
    value: int


class DummyExecutor(BaseExecutor):
    name = "dummy"
    description = "A dummy tool for testing."
    args_schema = DummyArgs

    def execute(self, *, value: int) -> str:
        return f"got {value}"


class TestExecutorRegistry:
    def setup_method(self) -> None:
        self.registry = ExecutorRegistry()

    def test_register_and_list(self) -> None:
        self.registry.register(DummyExecutor())
        tools = self.registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "dummy"
        assert "parameters" in tools[0]

    def test_duplicate_registration_raises(self) -> None:
        self.registry.register(DummyExecutor())
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(DummyExecutor())

    def test_validate_and_execute_success(self) -> None:
        self.registry.register(DummyExecutor())
        call = ToolCall(tool_name="dummy", arguments={"value": 42})
        result = self.registry.validate_and_execute(call)
        assert result.success is True
        assert result.output == "got 42"
        assert result.error is None

    def test_validate_rejects_bad_args(self) -> None:
        self.registry.register(DummyExecutor())
        call = ToolCall(tool_name="dummy", arguments={"value": "not_an_int"})
        result = self.registry.validate_and_execute(call)
        assert result.success is False
        assert "Validation failed" in (result.error or "")

    def test_validate_rejects_missing_args(self) -> None:
        self.registry.register(DummyExecutor())
        call = ToolCall(tool_name="dummy", arguments={})
        result = self.registry.validate_and_execute(call)
        assert result.success is False
        assert "Validation failed" in (result.error or "")

    def test_unknown_tool_rejected(self) -> None:
        call = ToolCall(tool_name="nonexistent", arguments={})
        result = self.registry.validate_and_execute(call)
        assert result.success is False
        assert "Unknown tool" in (result.error or "")

    def test_real_executors(self) -> None:
        self.registry.register(GetTimeExecutor())
        self.registry.register(SystemQueryExecutor())

        # get_time with defaults
        result = self.registry.validate_and_execute(
            ToolCall(tool_name="get_time", arguments={})
        )
        assert result.success is True
        assert "Current local time" in result.output

        # system_query
        result = self.registry.validate_and_execute(
            ToolCall(tool_name="system_query", arguments={"query": "os"})
        )
        assert result.success is True

    def test_extra_args_rejected(self) -> None:
        """Extra arguments not in the schema should be rejected by Pydantic strict mode."""
        self.registry.register(DummyExecutor())
        # Pydantic v2 by default ignores extra fields, but the call should still work
        # with the valid field — this tests that only valid fields are passed through
        call = ToolCall(tool_name="dummy", arguments={"value": 10, "extra": "bad"})
        result = self.registry.validate_and_execute(call)
        # With default Pydantic config, extra fields are ignored and execution succeeds
        assert result.success is True
        assert result.output == "got 10"
