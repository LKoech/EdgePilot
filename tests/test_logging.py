"""Tests for structured JSON logging."""

import json
import logging

from edgepilot.logging_config import StructuredFormatter


class TestStructuredLogging:
    def test_basic_log_is_valid_json(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert "timestamp" in parsed

    def test_extras_included(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="recovery event",
            args=(),
            exc_info=None,
        )
        record.event_type = "recovery"  # type: ignore[attr-defined]
        record.tool_name = "get_time"  # type: ignore[attr-defined]
        record.attempt = 2  # type: ignore[attr-defined]
        record.failure_type = "validation"  # type: ignore[attr-defined]

        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["event_type"] == "recovery"
        assert parsed["tool_name"] == "get_time"
        assert parsed["attempt"] == 2
        assert parsed["failure_type"] == "validation"

    def test_missing_extras_omitted(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="no extras",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "event_type" not in parsed
        assert "tool_name" not in parsed
