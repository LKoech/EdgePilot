"""Tests for planner — unit tests using a mock planner (no Ollama needed)."""

from edgepilot.executors.base import ToolCall
from edgepilot.planner.base import BasePlanner, PlannerResult
from edgepilot.planner.ollama_planner import OllamaPlanner


class MockPlanner(BasePlanner):
    """A mock planner that returns pre-configured responses."""

    def __init__(self, responses: list[PlannerResult]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def plan(
        self, user_input: str, tool_descriptions: list[dict], context: str = ""
    ) -> PlannerResult:
        if self._call_count < len(self._responses):
            result = self._responses[self._call_count]
        else:
            result = self._responses[-1]
        self._call_count += 1
        return result

    @property
    def call_count(self) -> int:
        return self._call_count


class TestMockPlanner:
    def test_returns_tool_call(self) -> None:
        planner = MockPlanner([
            PlannerResult(
                tool_call=ToolCall(tool_name="get_time", arguments={}),
                raw_output='{"tool_name": "get_time", "arguments": {}}',
            )
        ])
        result = planner.plan("what time is it?", [])
        assert result.is_tool_call
        assert result.tool_call is not None
        assert result.tool_call.tool_name == "get_time"

    def test_returns_text(self) -> None:
        planner = MockPlanner([
            PlannerResult(text_response="Hello! How can I help?")
        ])
        result = planner.plan("hi", [])
        assert not result.is_tool_call
        assert result.text_response == "Hello! How can I help?"


class TestOllamaPlannerParsing:
    """Test the JSON parsing logic without calling Ollama."""

    def test_parse_clean_json(self) -> None:
        raw = '{"tool_name": "get_time", "arguments": {}}'
        result = OllamaPlanner._try_parse_tool_call(raw)
        assert result is not None
        assert result.tool_name == "get_time"

    def test_parse_json_in_code_fence(self) -> None:
        raw = '```json\n{"tool_name": "get_time", "arguments": {}}\n```'
        result = OllamaPlanner._try_parse_tool_call(raw)
        assert result is not None
        assert result.tool_name == "get_time"

    def test_parse_json_with_surrounding_text(self) -> None:
        raw = (
            'I will call the tool:\n'
            '{"tool_name": "system_query", "arguments": {"query": "os"}}\nDone.'
        )
        result = OllamaPlanner._try_parse_tool_call(raw)
        assert result is not None
        assert result.tool_name == "system_query"

    def test_parse_plain_text_returns_none(self) -> None:
        raw = "Hello! I can help you with that."
        result = OllamaPlanner._try_parse_tool_call(raw)
        assert result is None

    def test_parse_invalid_json_returns_none(self) -> None:
        raw = '{"tool_name": "get_time", "arguments": {invalid}}'
        result = OllamaPlanner._try_parse_tool_call(raw)
        assert result is None

    def test_parse_json_missing_tool_name_returns_none(self) -> None:
        raw = '{"arguments": {}}'
        result = OllamaPlanner._try_parse_tool_call(raw)
        assert result is None
