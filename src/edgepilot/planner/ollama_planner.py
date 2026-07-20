"""Ollama-backed SLM planner — runs fully local via the Ollama HTTP API."""

import json
import logging
import time
from typing import Any

import httpx
from pydantic import ValidationError

from edgepilot.executors.base import ToolCall
from edgepilot.metrics.collector import metrics
from edgepilot.planner.base import BasePlanner, PlannerResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are EdgePilot, a local voice assistant. You help users by calling tools.

Available tools:
{tools_json}

When you need to use a tool, respond with EXACTLY this JSON format and nothing else:
{{"tool_name": "<name>", "arguments": {{...}}}}

When no tool is needed, respond with plain text.

Rules:
- Only use tools from the list above.
- Match argument names and types exactly to the tool's parameter schema.
- Never fabricate tool names or arguments not in the schema.
- If a previous tool call failed, try a different approach or respond with text.
"""


class OllamaPlanner(BasePlanner):
    """Planner that calls a local Ollama instance."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout

    def plan(
        self,
        user_input: str,
        tool_descriptions: list[dict[str, Any]],
        context: str = "",
    ) -> PlannerResult:
        tools_json = json.dumps(tool_descriptions, indent=2)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_json)

        messages = [{"role": "system", "content": system_prompt}]
        if context:
            content = f"Context from previous attempt:\n{context}"
            messages.append({"role": "system", "content": content})
        messages.append({"role": "user", "content": user_input})

        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Ollama request failed: %s", e)
            return PlannerResult(
                text_response=(
                    f"Planner error: could not reach Ollama at "
                    f"{self.base_url}. Is it running?"
                ),
                raw_output="",
            )

        elapsed = time.perf_counter() - start
        data = response.json()

        raw_output = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)

        metrics.planner_latency.observe(elapsed)
        if eval_count > 0:
            metrics.planner_tokens_per_sec.observe(eval_count / elapsed)

        logger.debug("Planner raw output: %s", raw_output)

        # Try to parse as a tool call
        tool_call = self._try_parse_tool_call(raw_output)
        if tool_call is not None:
            return PlannerResult(
                tool_call=tool_call,
                raw_output=raw_output,
                tokens_generated=eval_count,
                generation_time_sec=elapsed,
            )

        return PlannerResult(
            text_response=raw_output,
            raw_output=raw_output,
            tokens_generated=eval_count,
            generation_time_sec=elapsed,
        )

    @staticmethod
    def _try_parse_tool_call(raw: str) -> ToolCall | None:
        """Attempt to extract a ToolCall from the raw LLM output."""
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool_name" in data:
                return ToolCall.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict) and "tool_name" in data:
                    return ToolCall.model_validate(data)
            except (json.JSONDecodeError, ValidationError):
                pass

        return None
