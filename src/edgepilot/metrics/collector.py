"""Prometheus metrics for EdgePilot."""

from prometheus_client import Counter, Histogram, Summary


class EdgePilotMetrics:
    """Centralized metrics collector."""

    def __init__(self) -> None:
        self.validation_passes = Counter(
            "edgepilot_validation_passes_total",
            "Tool calls that passed schema validation",
        )
        self.validation_rejections = Counter(
            "edgepilot_validation_rejections_total",
            "Tool calls rejected at the validation gate",
        )
        self.tool_executions = Counter(
            "edgepilot_tool_executions_total",
            "Tool executions by tool and status",
            ["tool", "status"],
        )
        self.recovery_attempts = Counter(
            "edgepilot_recovery_attempts_total",
            "Adaptive recovery re-plan attempts",
        )
        self.recovery_successes = Counter(
            "edgepilot_recovery_successes_total",
            "Adaptive recovery attempts that succeeded",
        )
        self.recovery_exhausted = Counter(
            "edgepilot_recovery_exhausted_total",
            "Tasks where recovery budget was exhausted",
        )
        self.planner_latency = Histogram(
            "edgepilot_planner_latency_seconds",
            "Planner call latency in seconds",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )
        self.planner_tokens_per_sec = Summary(
            "edgepilot_planner_tokens_per_sec",
            "Planner output tokens per second",
        )
        self.end_to_end_latency = Histogram(
            "edgepilot_end_to_end_latency_seconds",
            "End-to-end request latency in seconds",
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )


# Singleton
metrics = EdgePilotMetrics()
