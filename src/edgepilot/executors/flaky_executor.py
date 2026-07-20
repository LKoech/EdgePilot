"""A deliberately flaky executor for testing adaptive recovery."""

import random

from pydantic import BaseModel, Field

from edgepilot.executors.base import BaseExecutor


class FlakyArgs(BaseModel):
    query: str = Field(description="A query to process.")
    fail_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of failure (0.0-1.0). Default 0.5.",
    )


class FlakyExecutor(BaseExecutor):
    """An executor that fails randomly — used to demonstrate adaptive recovery."""

    name = "flaky_lookup"
    description = (
        "Look up information (intentionally unreliable — may fail). "
        "Use for testing recovery. Has a 'fail_rate' parameter (0.0-1.0)."
    )
    args_schema = FlakyArgs

    def execute(self, *, query: str, fail_rate: float = 0.5) -> str:
        if random.random() < fail_rate:
            raise RuntimeError(
                f"Flaky lookup failed for query '{query}' "
                f"(simulated failure, rate={fail_rate})"
            )
        return f"Flaky lookup result for '{query}': [simulated data — 42 records found]"
