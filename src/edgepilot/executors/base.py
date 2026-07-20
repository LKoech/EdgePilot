"""Base executor interface and tool-call schema."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """The structured JSON the planner emits to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result from executing a tool."""

    tool_name: str
    success: bool
    output: str
    error: str | None = None


class BaseExecutor(ABC):
    """Abstract base for all typed executors.

    Subclasses must define:
      - name: the tool name the planner references
      - description: natural-language description for the planner's system prompt
      - args_schema: a Pydantic model class defining the expected arguments
      - execute(**kwargs): the actual tool logic
    """

    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Run the tool with validated arguments. Returns a string result."""
        ...
