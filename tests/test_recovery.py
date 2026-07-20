"""Tests for the adaptive recovery controller."""

from edgepilot.executors.base import ToolCall
from edgepilot.executors.flaky_executor import FlakyExecutor
from edgepilot.executors.get_time import GetTimeExecutor
from edgepilot.executors.registry import ExecutorRegistry
from edgepilot.planner.base import BasePlanner, PlannerResult
from edgepilot.recovery.controller import FailureType, RecoveryController


class SequencePlanner(BasePlanner):
    """Planner that returns a sequence of pre-configured results."""

    def __init__(self, responses: list[PlannerResult]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.contexts: list[str] = []

    def plan(
        self, user_input: str, tool_descriptions: list[dict], context: str = ""
    ) -> PlannerResult:
        self.contexts.append(context)
        idx = min(self._call_count, len(self._responses) - 1)
        result = self._responses[idx]
        self._call_count += 1
        return result


class TestRecoveryController:
    def setup_method(self) -> None:
        self.registry = ExecutorRegistry()
        self.registry.register(GetTimeExecutor())

    def test_success_on_first_try(self) -> None:
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(tool_name="get_time", arguments={}),
            )
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        result = ctrl.handle_request("what time is it?")

        assert result.success is True
        assert "Current local time" in result.response
        assert result.total_attempts == 1
        assert len(result.recovery_events) == 0

    def test_text_response_no_tool(self) -> None:
        planner = SequencePlanner([
            PlannerResult(text_response="Hello there!")
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        result = ctrl.handle_request("hi")

        assert result.success is True
        assert result.response == "Hello there!"
        assert result.tool_name is None

    def test_recovery_on_unknown_tool(self) -> None:
        """First attempt calls unknown tool, second attempt calls valid tool."""
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(tool_name="nonexistent", arguments={}),
            ),
            PlannerResult(
                tool_call=ToolCall(tool_name="get_time", arguments={}),
            ),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        result = ctrl.handle_request("what time is it?")

        assert result.success is True
        assert result.total_attempts == 2
        assert len(result.recovery_events) == 1
        assert "nonexistent" in planner.contexts[1]

    def test_recovery_on_validation_failure_then_success(self) -> None:
        """First call has bad args, recovery re-plans with valid args."""
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="get_time",
                    arguments={"timezone": 12345},
                ),
            ),
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="get_time",
                    arguments={"timezone": "local"},
                ),
            ),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=2)
        result = ctrl.handle_request("time?")
        assert result.success is True
        assert result.total_attempts == 2
        assert len(result.recovery_events) == 1

    def test_budget_exhaustion(self) -> None:
        """All attempts fail — budget exhausted."""
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(tool_name="nonexistent", arguments={}),
            ),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=2)
        result = ctrl.handle_request("do something")

        assert result.success is False
        assert result.total_attempts == 3  # 1 initial + 2 retries
        assert len(result.recovery_events) == 3
        assert "attempts" in result.response.lower()

    def test_recovery_context_passed_to_planner(self) -> None:
        """Verify the planner receives error context on re-plan."""
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(tool_name="bad_tool", arguments={}),
            ),
            PlannerResult(text_response="I'll just respond directly."),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        ctrl.handle_request("test")

        assert planner.contexts[0] == ""
        assert "bad_tool" in planner.contexts[1]
        assert "Error" in planner.contexts[1]


class TestRecoveryFailureTypes:
    """Tests for failure type classification and structured recovery events."""

    def setup_method(self) -> None:
        self.registry = ExecutorRegistry()
        self.registry.register(GetTimeExecutor())
        self.registry.register(FlakyExecutor())

    def test_unknown_tool_classified(self) -> None:
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(tool_name="no_such_tool", arguments={}),
            ),
            PlannerResult(text_response="OK"),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        result = ctrl.handle_request("test")

        assert len(result.recovery_events) == 1
        assert result.recovery_events[0].failure_type == FailureType.UNKNOWN_TOOL
        assert result.recovery_events[0].tool_name == "no_such_tool"

    def test_validation_failure_classified(self) -> None:
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="get_time",
                    arguments={"timezone": 999},
                ),
            ),
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="get_time", arguments={}
                ),
            ),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        result = ctrl.handle_request("time?")

        assert result.success is True
        assert len(result.recovery_events) == 1
        assert result.recovery_events[0].failure_type == FailureType.VALIDATION

    def test_execution_failure_classified(self) -> None:
        """Flaky executor with fail_rate=1.0 triggers execution failure."""
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="flaky_lookup",
                    arguments={"query": "test", "fail_rate": 1.0},
                ),
            ),
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="get_time", arguments={}
                ),
            ),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        result = ctrl.handle_request("look up data")

        assert result.success is True
        assert result.total_attempts == 2
        assert len(result.recovery_events) == 1
        assert result.recovery_events[0].failure_type == FailureType.EXECUTION

    def test_mixed_failure_types(self) -> None:
        """Unknown tool -> exec failure -> success."""
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(tool_name="fake", arguments={}),
            ),
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="flaky_lookup",
                    arguments={"query": "x", "fail_rate": 1.0},
                ),
            ),
            PlannerResult(
                tool_call=ToolCall(tool_name="get_time", arguments={}),
            ),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        result = ctrl.handle_request("test")

        assert result.success is True
        assert result.total_attempts == 3
        assert len(result.recovery_events) == 2
        assert result.recovery_events[0].failure_type == FailureType.UNKNOWN_TOOL
        assert result.recovery_events[1].failure_type == FailureType.EXECUTION

    def test_failure_type_in_context(self) -> None:
        """Verify failure type is included in recovery context."""
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(tool_name="fake", arguments={}),
            ),
            PlannerResult(text_response="ok"),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        ctrl.handle_request("test")

        assert "unknown_tool" in planner.contexts[1]

    def test_recovery_event_has_tool_name(self) -> None:
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="flaky_lookup",
                    arguments={"query": "x", "fail_rate": 1.0},
                ),
            ),
            PlannerResult(text_response="giving up"),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=1)
        result = ctrl.handle_request("test")

        assert result.recovery_events[0].tool_name == "flaky_lookup"

    def test_flaky_executor_success_path(self) -> None:
        """Flaky executor with fail_rate=0 always succeeds."""
        planner = SequencePlanner([
            PlannerResult(
                tool_call=ToolCall(
                    tool_name="flaky_lookup",
                    arguments={"query": "data", "fail_rate": 0.0},
                ),
            ),
        ])
        ctrl = RecoveryController(planner, self.registry, max_retries=3)
        result = ctrl.handle_request("look up data")

        assert result.success is True
        assert result.total_attempts == 1
        assert len(result.recovery_events) == 0
        assert "42 records" in result.response
