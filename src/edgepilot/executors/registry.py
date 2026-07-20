"""Typed executor registry — validates tool calls against Pydantic schemas before dispatch."""

import logging
from typing import Any

from pydantic import ValidationError

from edgepilot.executors.base import BaseExecutor, ToolCall, ToolResult
from edgepilot.metrics.collector import metrics

logger = logging.getLogger(__name__)


class ExecutorRegistry:
    """Registry of typed executors. Validates and dispatches tool calls."""

    def __init__(self) -> None:
        self._executors: dict[str, BaseExecutor] = {}

    def register(self, executor: BaseExecutor) -> None:
        if executor.name in self._executors:
            raise ValueError(f"Executor already registered: {executor.name}")
        self._executors[executor.name] = executor
        logger.info("Registered executor: %s", executor.name)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool descriptions for the planner's system prompt."""
        tools = []
        for ex in self._executors.values():
            schema = ex.args_schema.model_json_schema()
            tools.append(
                {
                    "name": ex.name,
                    "description": ex.description,
                    "parameters": schema,
                }
            )
        return tools

    def validate_and_execute(self, tool_call: ToolCall) -> ToolResult:
        """Validate a tool call against its schema, then execute.

        Returns a ToolResult. On validation failure, returns an error result
        without executing — malformed calls never reach the executor.
        """
        executor = self._executors.get(tool_call.tool_name)
        if executor is None:
            metrics.validation_rejections.inc()
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_call.tool_name}",
            )

        # Schema validation gate — reject before execution
        try:
            validated = executor.args_schema.model_validate(tool_call.arguments)
        except ValidationError as e:
            metrics.validation_rejections.inc()
            logger.warning(
                "Validation rejected tool call to %s: %s",
                tool_call.tool_name,
                e.errors(),
            )
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output="",
                error=f"Validation failed: {e}",
            )

        metrics.validation_passes.inc()

        # Execute with validated arguments
        try:
            output = executor.execute(**validated.model_dump())
            metrics.tool_executions.labels(tool=tool_call.tool_name, status="success").inc()
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                output=output,
            )
        except Exception as e:
            metrics.tool_executions.labels(tool=tool_call.tool_name, status="error").inc()
            logger.exception("Executor %s raised an exception", tool_call.tool_name)
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output="",
                error=f"Execution error: {e}",
            )

    @property
    def tool_names(self) -> list[str]:
        return list(self._executors.keys())
