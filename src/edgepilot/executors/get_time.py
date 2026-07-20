"""get_time executor — returns current local date/time."""

from datetime import datetime

from pydantic import BaseModel

from edgepilot.executors.base import BaseExecutor


class GetTimeArgs(BaseModel):
    timezone: str = "local"


class GetTimeExecutor(BaseExecutor):
    name = "get_time"
    description = "Get the current date and time. Optional timezone parameter (default: local)."
    args_schema = GetTimeArgs

    def execute(self, *, timezone: str = "local") -> str:
        now = datetime.now()
        return f"Current local time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
