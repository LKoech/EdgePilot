"""Adaptive recovery controller — re-plans on failure with bounded retries."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from edgepilot.executors.base import ToolResult
from edgepilot.executors.registry import ExecutorRegistry
from edgepilot.metrics.collector import metrics
from edgepilot.planner.base import BasePlanner
from edgepilot.validation.gate import ValidationGate

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    VALIDATION = "validation"
    EXECUTION = "execution"
    UNKNOWN_TOOL = "unknown_tool"


@dataclass
class RecoveryEvent:
    """A single recovery attempt record."""

    attempt: int
    error: str
    failure_type: FailureType
    tool_name: str
    replanned: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineResult:
    """Full result of processing a user request through the pipeline."""

    user_input: str
    response: str
    success: bool
    tool_name: str | None = None
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    total_attempts: int = 1
    elapsed_sec: float = 0.0


def _classify_failure(result: ToolResult) -> FailureType:
    """Classify a tool failure into a FailureType for structured logging."""
    error = result.error or ""
    if "Unknown tool" in error:
        return FailureType.UNKNOWN_TOOL
    if "Validation failed" in error:
        return FailureType.VALIDATION
    return FailureType.EXECUTION


class RecoveryController:
    """Orchestrates the plan -> validate -> execute loop with adaptive recovery.

    On failure (validation rejection or execution error), feeds the error
    back to the planner as context and retries up to max_retries times.
    Every recovery event is logged as a structured JSON record.
    """

    def __init__(
        self,
        planner: BasePlanner,
        registry: ExecutorRegistry,
        max_retries: int = 3,
    ) -> None:
        self.planner = planner
        self.gate = ValidationGate(registry)
        self.registry = registry
        self.max_retries = max_retries

    def handle_request(self, user_input: str) -> PipelineResult:
        """Process a user request end-to-end with adaptive recovery."""
        start = time.perf_counter()
        tool_descriptions = self.registry.list_tools()
        recovery_events: list[RecoveryEvent] = []
        context = ""
        last_response = ""

        for attempt in range(1, self.max_retries + 2):
            plan_result = self.planner.plan(user_input, tool_descriptions, context)

            # Direct text response — no tool call needed
            if not plan_result.is_tool_call:
                elapsed = time.perf_counter() - start
                metrics.end_to_end_latency.observe(elapsed)
                self._log_completion(
                    user_input,
                    True,
                    None,
                    attempt,
                    len(recovery_events),
                    elapsed,
                    plan_result.tokens_per_sec,
                )
                return PipelineResult(
                    user_input=user_input,
                    response=plan_result.text_response or "",
                    success=True,
                    recovery_events=recovery_events,
                    total_attempts=attempt,
                    elapsed_sec=elapsed,
                )

            # Tool call — validate and execute
            assert plan_result.tool_call is not None
            tool_result: ToolResult = self.gate.process(plan_result.tool_call)

            if tool_result.success:
                elapsed = time.perf_counter() - start
                metrics.end_to_end_latency.observe(elapsed)
                if recovery_events:
                    metrics.recovery_successes.inc()
                self._log_completion(
                    user_input,
                    True,
                    tool_result.tool_name,
                    attempt,
                    len(recovery_events),
                    elapsed,
                    plan_result.tokens_per_sec,
                )
                return PipelineResult(
                    user_input=user_input,
                    response=tool_result.output,
                    success=True,
                    tool_name=tool_result.tool_name,
                    recovery_events=recovery_events,
                    total_attempts=attempt,
                    elapsed_sec=elapsed,
                )

            # Failure — classify, record, and build context for re-plan
            last_response = tool_result.error or "Unknown error"
            failure_type = _classify_failure(tool_result)
            can_retry = attempt <= self.max_retries

            event = RecoveryEvent(
                attempt=attempt,
                error=last_response,
                failure_type=failure_type,
                tool_name=plan_result.tool_call.tool_name,
                replanned=can_retry,
            )
            recovery_events.append(event)
            metrics.recovery_attempts.inc()

            self._log_recovery_event(user_input, event)

            if can_retry:
                context = (
                    f"Previous attempt {attempt} failed.\n"
                    f"Tool: {plan_result.tool_call.tool_name}\n"
                    f"Args: {plan_result.tool_call.arguments}\n"
                    f"Error: {last_response}\n"
                    f"Failure type: {failure_type.value}\n"
                    f"Please try a different approach or different arguments."
                )
            else:
                metrics.recovery_exhausted.inc()

        elapsed = time.perf_counter() - start
        metrics.end_to_end_latency.observe(elapsed)
        self._log_completion(
            user_input,
            False,
            None,
            self.max_retries + 1,
            len(recovery_events),
            elapsed,
            0.0,
        )
        return PipelineResult(
            user_input=user_input,
            response=(
                f"I wasn't able to complete that after "
                f"{self.max_retries + 1} attempts. "
                f"Last error: {last_response}"
            ),
            success=False,
            recovery_events=recovery_events,
            total_attempts=self.max_retries + 1,
            elapsed_sec=elapsed,
        )

    @staticmethod
    def _log_recovery_event(user_input: str, event: RecoveryEvent) -> None:
        """Emit a structured recovery event log."""
        logger.warning(
            "Recovery event: attempt=%d tool=%s failure=%s",
            event.attempt,
            event.tool_name,
            event.failure_type.value,
            extra={
                "event_type": "recovery",
                "attempt": event.attempt,
                "tool_name": event.tool_name,
                "failure_type": event.failure_type.value,
                "error": event.error,
                "user_input": user_input,
            },
        )

    @staticmethod
    def _log_completion(
        user_input: str,
        success: bool,
        tool_name: str | None,
        attempts: int,
        recovery_count: int,
        elapsed_sec: float,
        tokens_per_sec: float,
    ) -> None:
        """Emit a structured pipeline completion log."""
        level = logging.INFO if success else logging.ERROR
        logger.log(
            level,
            "Pipeline %s: tool=%s attempts=%d recoveries=%d elapsed=%.3fs",
            "succeeded" if success else "failed",
            tool_name or "none",
            attempts,
            recovery_count,
            elapsed_sec,
            extra={
                "event_type": "pipeline_complete",
                "success": success,
                "tool_name": tool_name,
                "attempt": attempts,
                "recovery_count": recovery_count,
                "elapsed_sec": round(elapsed_sec, 4),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "user_input": user_input,
            },
        )
