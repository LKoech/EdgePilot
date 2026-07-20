"""system_query executor — returns local system information."""

import platform
import shutil

from pydantic import BaseModel, Field

from edgepilot.executors.base import BaseExecutor


class SystemQueryArgs(BaseModel):
    query: str = Field(
        description="What to query: 'os', 'cpu', 'memory', 'disk', or 'all'."
    )


class SystemQueryExecutor(BaseExecutor):
    name = "system_query"
    description = (
        "Query local system information. Supports: 'os', 'cpu', 'disk', or 'all'."
    )
    args_schema = SystemQueryArgs

    def execute(self, *, query: str) -> str:
        info: dict[str, str] = {}

        if query in ("os", "all"):
            info["os"] = f"{platform.system()} {platform.release()}"
            info["machine"] = platform.machine()
            info["python"] = platform.python_version()

        if query in ("cpu", "all"):
            info["processor"] = platform.processor() or "unknown"
            try:
                import os

                info["cpu_count"] = str(os.cpu_count() or "unknown")
            except Exception:
                info["cpu_count"] = "unknown"

        if query in ("disk", "all"):
            usage = shutil.disk_usage(".")
            info["disk_total_gb"] = f"{usage.total / (1024**3):.1f}"
            info["disk_free_gb"] = f"{usage.free / (1024**3):.1f}"

        if not info:
            return f"Unknown query type: {query}. Use 'os', 'cpu', 'disk', or 'all'."

        return "\n".join(f"{k}: {v}" for k, v in info.items())
