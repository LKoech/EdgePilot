"""Tests for the metrics collector and /metrics endpoint."""

from edgepilot.metrics.collector import metrics


class TestEdgePilotMetrics:
    def test_all_counters_exist(self) -> None:
        assert metrics.validation_passes is not None
        assert metrics.validation_rejections is not None
        assert metrics.recovery_attempts is not None
        assert metrics.recovery_successes is not None
        assert metrics.recovery_exhausted is not None
        assert metrics.requests_total is not None

    def test_all_histograms_exist(self) -> None:
        assert metrics.planner_latency is not None
        assert metrics.end_to_end_latency is not None
        assert metrics.stt_latency is not None
        assert metrics.tts_latency is not None

    def test_all_summaries_exist(self) -> None:
        assert metrics.planner_tokens_per_sec is not None
        assert metrics.stt_audio_duration is not None
        assert metrics.stt_realtime_factor is not None
        assert metrics.tts_audio_duration is not None

    def test_gauge_exists(self) -> None:
        assert metrics.active_requests is not None

    def test_tool_executions_labeled(self) -> None:
        # Should be able to create labeled children without error
        metrics.tool_executions.labels(tool="test_tool", status="success")
        metrics.requests_total.labels(status="success")


class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_prometheus_format(self) -> None:
        from fastapi.testclient import TestClient

        from edgepilot.main import app

        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.text
        assert "edgepilot_validation_passes_total" in body
        assert "edgepilot_end_to_end_latency" in body
        assert "edgepilot_stt_latency" in body
        assert "edgepilot_tts_latency" in body
        assert "edgepilot_active_requests" in body
