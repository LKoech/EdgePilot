"""EdgePilot main entry point — FastAPI app, text loop, and voice loop."""

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from edgepilot.executors.flaky_executor import FlakyExecutor
from edgepilot.executors.get_time import GetTimeExecutor
from edgepilot.executors.registry import ExecutorRegistry
from edgepilot.executors.set_reminder import SetReminderExecutor
from edgepilot.executors.system_query import SystemQueryExecutor
from edgepilot.logging_config import configure_logging
from edgepilot.planner.ollama_planner import OllamaPlanner
from edgepilot.recovery.controller import RecoveryController

configure_logging(
    structured=os.environ.get("EDGEPILOT_LOG_FORMAT") == "json",
)
logger = logging.getLogger(__name__)


def build_registry() -> ExecutorRegistry:
    registry = ExecutorRegistry()
    registry.register(GetTimeExecutor())
    registry.register(SystemQueryExecutor())
    registry.register(SetReminderExecutor())
    registry.register(FlakyExecutor())
    return registry


def build_controller() -> RecoveryController:
    registry = build_registry()
    planner = OllamaPlanner()
    return RecoveryController(planner=planner, registry=registry, max_retries=3)


controller: RecoveryController | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global controller
    controller = build_controller()
    logger.info(
        "EdgePilot started — all processing is local, "
        "no data leaves this machine."
    )
    yield
    logger.info("EdgePilot shutting down.")


app = FastAPI(
    title="EdgePilot",
    description=(
        "Fully local voice assistant with typed executors "
        "and adaptive recovery."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    success: bool
    tool_used: str | None = None
    attempts: int = 1
    recovery_events: int = 0
    elapsed_sec: float = 0.0


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    assert controller is not None
    result = controller.handle_request(request.message)
    return ChatResponse(
        response=result.response,
        success=result.success,
        tool_used=result.tool_name,
        attempts=result.total_attempts,
        recovery_events=len(result.recovery_events),
        elapsed_sec=round(result.elapsed_sec, 3),
    )


@app.get("/tools")
async def list_tools() -> list[dict]:
    assert controller is not None
    return controller.registry.list_tools()


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(
        content=generate_latest(), media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local-only"}


def interactive_loop() -> None:
    """REPL-style text loop."""
    ctrl = build_controller()
    print("\n=== EdgePilot (text mode) ===")
    print("Type a request, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = ctrl.handle_request(user_input)

        print(f"\nEdgePilot: {result.response}")
        if result.tool_name:
            print(f"  [tool: {result.tool_name}]")
        if result.recovery_events:
            print(
                f"  [recovery attempts: {len(result.recovery_events)}]"
            )
        print(
            f"  [attempts: {result.total_attempts}, "
            f"elapsed: {result.elapsed_sec:.2f}s]\n"
        )


def voice_mode() -> None:
    """Full voice loop — mic -> STT -> plan -> execute -> TTS -> speaker."""
    from edgepilot.stt.whisper_stt import WhisperSTT
    from edgepilot.tts.piper_tts import PiperTTS
    from edgepilot.voice_loop import VoiceLoop

    print("\n=== EdgePilot (voice mode) ===")
    print("Loading models... (first run downloads ~200MB)\n")

    ctrl = build_controller()
    stt = WhisperSTT(model_size="base", device="cpu", compute_type="int8")
    tts = PiperTTS()

    loop = VoiceLoop(stt=stt, tts=tts, controller=ctrl)
    loop.run()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif "--voice" in sys.argv:
        voice_mode()
    else:
        interactive_loop()
