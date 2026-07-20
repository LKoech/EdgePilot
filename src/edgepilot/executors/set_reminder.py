"""set_reminder executor — stores a reminder locally (in-memory for now)."""

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from edgepilot.executors.base import BaseExecutor

logger = logging.getLogger(__name__)

# Simple in-memory store. Phase 3+ could persist to SQLite.
_reminders: list[dict[str, str]] = []


class SetReminderArgs(BaseModel):
    message: str = Field(description="The reminder text.")
    time: str = Field(
        description="When to remind, e.g. '2025-01-15 14:30' or 'in 5 minutes'."
    )


class SetReminderExecutor(BaseExecutor):
    name = "set_reminder"
    description = "Set a reminder with a message and a time."
    args_schema = SetReminderArgs

    def execute(self, *, message: str, time: str) -> str:
        reminder = {
            "message": message,
            "time": time,
            "created_at": datetime.now().isoformat(),
        }
        _reminders.append(reminder)
        logger.info("Reminder set: %s at %s", message, time)
        return f"Reminder set: '{message}' at {time}. Total reminders: {len(_reminders)}."


def get_reminders() -> list[dict[str, str]]:
    return list(_reminders)
