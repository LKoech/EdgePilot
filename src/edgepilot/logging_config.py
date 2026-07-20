"""Structured JSON logging for EdgePilot — machine-parseable recovery and pipeline events."""

import json
import logging
from datetime import UTC, datetime


class StructuredFormatter(logging.Formatter):
    """Emits log records as single-line JSON for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach structured extras if present
        for key in (
            "event_type",
            "attempt",
            "tool_name",
            "error",
            "user_input",
            "success",
            "elapsed_sec",
            "tokens_per_sec",
            "failure_type",
            "recovery_count",
        ):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, default=str)


def configure_logging(structured: bool = False, level: int = logging.INFO) -> None:
    """Configure root logging. Use structured=True for JSON output."""
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers
    root.handlers.clear()
    handler = logging.StreamHandler()
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(handler)
