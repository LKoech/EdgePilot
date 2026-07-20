"""Eval scenarios — scripted requests with expected outcomes.

Each scenario defines:
  - user_input: what the user says
  - expected_tool: which tool should be called (or None for text response)
  - planner_calls: sequence of ToolCall dicts the mock planner will emit
  - inject_failure: if True, the executor will be swapped for one that fails
  - expect_recovery: whether adaptive recovery should kick in
  - expect_success: whether the pipeline should ultimately succeed
"""

from dataclasses import dataclass


@dataclass
class Scenario:
    id: str
    description: str
    user_input: str
    expected_tool: str | None
    planner_calls: list[dict]
    expect_recovery: bool = False
    expect_success: bool = True


# fmt: off
SCENARIOS: list[Scenario] = [
    # --- Happy-path: valid tool calls succeed on first try ---
    Scenario(
        id="time_basic",
        description="Simple time query",
        user_input="What time is it?",
        expected_tool="get_time",
        planner_calls=[{"tool_name": "get_time", "arguments": {}}],
    ),
    Scenario(
        id="time_with_tz",
        description="Time query with timezone param",
        user_input="What time is it in UTC?",
        expected_tool="get_time",
        planner_calls=[{"tool_name": "get_time", "arguments": {"timezone": "UTC"}}],
    ),
    Scenario(
        id="system_os",
        description="Query OS info",
        user_input="What operating system am I running?",
        expected_tool="system_query",
        planner_calls=[{"tool_name": "system_query", "arguments": {"query": "os"}}],
    ),
    Scenario(
        id="system_disk",
        description="Query disk info",
        user_input="How much disk space is free?",
        expected_tool="system_query",
        planner_calls=[{"tool_name": "system_query", "arguments": {"query": "disk"}}],
    ),
    Scenario(
        id="system_all",
        description="Query all system info",
        user_input="Give me all system information",
        expected_tool="system_query",
        planner_calls=[{"tool_name": "system_query", "arguments": {"query": "all"}}],
    ),
    Scenario(
        id="reminder_set",
        description="Set a reminder",
        user_input="Remind me to call mom at 5pm",
        expected_tool="set_reminder",
        planner_calls=[{
            "tool_name": "set_reminder",
            "arguments": {"message": "Call mom", "time": "17:00"},
        }],
    ),
    Scenario(
        id="text_greeting",
        description="Simple greeting — no tool needed",
        user_input="Hello!",
        expected_tool=None,
        planner_calls=[],  # text response, no tool call
    ),
    Scenario(
        id="text_help",
        description="Help request — text response",
        user_input="What can you do?",
        expected_tool=None,
        planner_calls=[],
    ),

    # --- Validation failures: malformed calls caught pre-execution ---
    Scenario(
        id="val_unknown_tool",
        description="Planner hallucinates a tool that doesn't exist",
        user_input="Search the web for cats",
        expected_tool="get_time",
        planner_calls=[
            {"tool_name": "web_search", "arguments": {"query": "cats"}},
            {"tool_name": "get_time", "arguments": {}},
        ],
        expect_recovery=True,
        expect_success=True,
    ),
    Scenario(
        id="val_bad_arg_type",
        description="Planner sends wrong arg type",
        user_input="What time is it?",
        expected_tool="get_time",
        planner_calls=[
            {"tool_name": "get_time", "arguments": {"timezone": 12345}},
            {"tool_name": "get_time", "arguments": {"timezone": "local"}},
        ],
        expect_recovery=True,
        expect_success=True,
    ),
    Scenario(
        id="val_missing_required",
        description="Planner omits required argument",
        user_input="What OS am I on?",
        expected_tool="system_query",
        planner_calls=[
            {"tool_name": "system_query", "arguments": {}},
            {"tool_name": "system_query", "arguments": {"query": "os"}},
        ],
        expect_recovery=True,
        expect_success=True,
    ),
    Scenario(
        id="val_empty_tool_name",
        description="Planner sends empty tool name",
        user_input="Do something",
        expected_tool="get_time",
        planner_calls=[
            {"tool_name": "", "arguments": {}},
            {"tool_name": "get_time", "arguments": {}},
        ],
        expect_recovery=True,
        expect_success=True,
    ),

    # --- Execution failures: tool runs but raises ---
    Scenario(
        id="exec_flaky_succeeds",
        description="Flaky tool called with fail_rate=0 (always succeeds)",
        user_input="Look up flaky data with no failures",
        expected_tool="flaky_lookup",
        planner_calls=[{
            "tool_name": "flaky_lookup",
            "arguments": {"query": "test", "fail_rate": 0.0},
        }],
        expect_success=True,
    ),
    Scenario(
        id="exec_flaky_fails_then_recovers",
        description="Flaky tool fails, recovery retries with fail_rate=0",
        user_input="Look up data (will fail first)",
        expected_tool="flaky_lookup",
        planner_calls=[
            {"tool_name": "flaky_lookup", "arguments": {"query": "test", "fail_rate": 1.0}},
            {"tool_name": "flaky_lookup", "arguments": {"query": "test", "fail_rate": 0.0}},
        ],
        expect_recovery=True,
        expect_success=True,
    ),
    Scenario(
        id="exec_flaky_exhausts_budget",
        description="Flaky tool fails every time — budget exhausted",
        user_input="Look up data (always fails)",
        expected_tool="flaky_lookup",
        planner_calls=[
            {"tool_name": "flaky_lookup", "arguments": {"query": "test", "fail_rate": 1.0}},
            {"tool_name": "flaky_lookup", "arguments": {"query": "test", "fail_rate": 1.0}},
            {"tool_name": "flaky_lookup", "arguments": {"query": "test", "fail_rate": 1.0}},
            {"tool_name": "flaky_lookup", "arguments": {"query": "test", "fail_rate": 1.0}},
        ],
        expect_recovery=True,
        expect_success=False,
    ),

    # --- Mixed failure patterns ---
    Scenario(
        id="mix_val_then_exec_fail_then_ok",
        description="Validation fail -> execution fail -> success",
        user_input="Complex multi-failure scenario",
        expected_tool="get_time",
        planner_calls=[
            {"tool_name": "nonexistent", "arguments": {}},
            {"tool_name": "flaky_lookup", "arguments": {"query": "x", "fail_rate": 1.0}},
            {"tool_name": "get_time", "arguments": {}},
        ],
        expect_recovery=True,
        expect_success=True,
    ),
    Scenario(
        id="mix_all_fail",
        description="Every attempt uses a bad tool — full budget exhaustion",
        user_input="Use a tool that doesn't exist",
        expected_tool=None,
        planner_calls=[
            {"tool_name": "fake_tool_1", "arguments": {}},
            {"tool_name": "fake_tool_2", "arguments": {}},
            {"tool_name": "fake_tool_3", "arguments": {}},
            {"tool_name": "fake_tool_4", "arguments": {}},
        ],
        expect_recovery=True,
        expect_success=False,
    ),

    # --- Edge cases ---
    Scenario(
        id="edge_system_cpu",
        description="Query CPU info specifically",
        user_input="How many CPU cores do I have?",
        expected_tool="system_query",
        planner_calls=[
            {"tool_name": "system_query", "arguments": {"query": "cpu"}},
        ],
    ),
    Scenario(
        id="edge_reminder_special_chars",
        description="Reminder with special characters",
        user_input="Remind me: buy eggs & milk @ store! (urgent)",
        expected_tool="set_reminder",
        planner_calls=[{
            "tool_name": "set_reminder",
            "arguments": {
                "message": "buy eggs & milk @ store! (urgent)",
                "time": "today",
            },
        }],
    ),
    Scenario(
        id="edge_system_bad_query_type",
        description="system_query with unsupported query type",
        user_input="What's the GPU temperature?",
        expected_tool="system_query",
        planner_calls=[
            {"tool_name": "system_query", "arguments": {"query": "gpu"}},
        ],
        expect_success=True,  # Returns "Unknown query type" but doesn't crash
    ),
]
# fmt: on
