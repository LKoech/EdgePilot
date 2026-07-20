"""Schema validation gate — sits between planner and executor dispatch."""

import logging

from edgepilot.executors.base import ToolCall, ToolResult
from edgepilot.executors.registry import ExecutorRegistry

logger = logging.getLogger(__name__)


class ValidationGate:
    """Validates and dispatches tool calls through the executor registry.

    This is the single entry point: planner output goes through here,
    gets validated, and either executes or returns an error result.
    """

    def __init__(self, registry: ExecutorRegistry) -> None:
        self.registry = registry

    def process(self, tool_call: ToolCall) -> ToolResult:
        """Validate the tool call against the registry schema and execute if valid."""
        logger.info(
            "Validation gate processing: tool=%s args=%s",
            tool_call.tool_name,
            tool_call.arguments,
        )
        result = self.registry.validate_and_execute(tool_call)

        if result.success:
            logger.info("Tool %s executed successfully", tool_call.tool_name)
        else:
            logger.warning("Tool %s failed: %s", tool_call.tool_name, result.error)

        return result
