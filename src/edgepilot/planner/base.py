"""Abstract planner interface — swappable SLM backend."""

from abc import ABC, abstractmethod

from edgepilot.executors.base import ToolCall


class PlannerResult:
    """Result from the planner: either a tool call or a direct text response."""

    def __init__(
        self,
        tool_call: ToolCall | None = None,
        text_response: str | None = None,
        raw_output: str = "",
        tokens_generated: int = 0,
        generation_time_sec: float = 0.0,
    ) -> None:
        self.tool_call = tool_call
        self.text_response = text_response
        self.raw_output = raw_output
        self.tokens_generated = tokens_generated
        self.generation_time_sec = generation_time_sec

    @property
    def is_tool_call(self) -> bool:
        return self.tool_call is not None

    @property
    def tokens_per_sec(self) -> float:
        if self.generation_time_sec > 0:
            return self.tokens_generated / self.generation_time_sec
        return 0.0


class BasePlanner(ABC):
    """Abstract planner. Subclasses implement plan() with a specific LLM backend."""

    @abstractmethod
    def plan(
        self,
        user_input: str,
        tool_descriptions: list[dict],
        context: str = "",
    ) -> PlannerResult:
        """Given user input and available tools, produce a plan (tool call or text)."""
        ...
