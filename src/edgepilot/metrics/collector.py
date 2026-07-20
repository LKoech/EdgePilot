"""Prometheus metrics for EdgePilot."""

from prometheus_client import Counter, Gauge, Histogram, Summary


class EdgePilotMetrics:
    """Centralized metrics collector."""

    def __init__(self) -> None:
        # --- Validation gate ---
        self.validation_passes = Counter(
            "edgepilot_validation_passes_total",
            "Tool calls that passed schema validation",
        )
        self.validation_rejections = Counter(
            "edgepilot_validation_rejections_total",
            "Tool calls rejected at the validation gate",
        )

        # --- Tool execution ---
        self.tool_executions = Counter(
            "edgepilot_tool_executions_total",
            "Tool executions by tool and status",
            ["tool", "status"],
        )

        # --- Adaptive recovery ---
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

        # --- Planner ---
        self.planner_latency = Histogram(
            "edgepilot_planner_latency_seconds",
            "Planner call latency in seconds",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )
        self.planner_tokens_per_sec = Summary(
            "edgepilot_planner_tokens_per_sec",
            "Planner output tokens per second",
        )

        # --- End-to-end ---
        self.end_to_end_latency = Histogram(
            "edgepilot_end_to_end_latency_seconds",
            "End-to-end request latency in seconds",
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        self.requests_total = Counter(
            "edgepilot_requests_total",
            "Total requests by status",
            ["status"],
        )

        # --- STT ---
        self.stt_latency = Histogram(
            "edgepilot_stt_latency_seconds",
            "STT processing latency in seconds",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self.stt_audio_duration = Summary(
            "edgepilot_stt_audio_duration_seconds",
            "Duration of audio transcribed",
        )
        self.stt_realtime_factor = Summary(
            "edgepilot_stt_realtime_factor",
            "STT processing time / audio duration (< 1 = faster than real-time)",
        )

        # --- TTS ---
        self.tts_latency = Histogram(
            "edgepilot_tts_latency_seconds",
            "TTS synthesis latency in seconds",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self.tts_audio_duration = Summary(
            "edgepilot_tts_audio_duration_seconds",
            "Duration of audio synthesized",
        )

        # --- System ---
        self.active_requests = Gauge(
            "edgepilot_active_requests",
            "Number of requests currently being processed",
        )


# Singleton
metrics = EdgePilotMetrics()
