"""Voice loop — full mic -> STT -> plan -> execute -> TTS -> speaker pipeline."""

import logging
import time

from edgepilot.audio import play_audio, record_until_silence
from edgepilot.metrics.collector import metrics
from edgepilot.recovery.controller import RecoveryController
from edgepilot.stt.base import BaseSTT
from edgepilot.tts.base import BaseTTS

logger = logging.getLogger(__name__)


class VoiceLoop:
    """Orchestrates the full voice pipeline.

    mic -> STT -> RecoveryController -> TTS -> speaker
    """

    def __init__(
        self,
        stt: BaseSTT,
        tts: BaseTTS,
        controller: RecoveryController,
        silence_threshold: float = 0.02,
        silence_duration: float = 1.5,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.controller = controller
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

    def run_once(self) -> str | None:
        """Run a single voice interaction cycle."""
        e2e_start = time.perf_counter()
        metrics.active_requests.inc()

        try:
            return self._run_once_inner(e2e_start)
        finally:
            metrics.active_requests.dec()

    def _run_once_inner(self, e2e_start: float) -> str | None:
        # 1. Record from mic
        print("\n  [Listening... speak now]")
        audio_bytes = record_until_silence(
            silence_threshold=self.silence_threshold,
            silence_duration=self.silence_duration,
        )
        if not audio_bytes:
            logger.info("No audio captured.")
            return None

        # 2. STT (with metrics)
        print("  [Transcribing...]")
        transcription = self.stt.transcribe_bytes(audio_bytes)
        metrics.stt_latency.observe(transcription.processing_sec)
        metrics.stt_audio_duration.observe(transcription.duration_sec)
        if transcription.realtime_factor > 0:
            metrics.stt_realtime_factor.observe(transcription.realtime_factor)

        user_text = transcription.text.strip()
        if not user_text:
            logger.info("Empty transcription, skipping.")
            return None

        print(f"\n  You: {user_text}")

        # Check for exit commands
        if user_text.lower() in ("quit", "exit", "stop", "goodbye", "bye"):
            self._speak("Goodbye!")
            return "EXIT"

        # 3. Plan + validate + execute (with recovery)
        print("  [Thinking...]")
        result = self.controller.handle_request(user_text)
        metrics.requests_total.labels(status="success" if result.success else "failure").inc()

        # 4. TTS + play
        response_text = result.response
        print(f"\n  EdgePilot: {response_text}")

        if result.tool_name:
            print(f"  [tool: {result.tool_name}]")
        if result.recovery_events:
            print(f"  [recovery: {len(result.recovery_events)} events]")

        self._speak(response_text)

        e2e_elapsed = time.perf_counter() - e2e_start
        metrics.end_to_end_latency.observe(e2e_elapsed)
        print(f"  [e2e: {e2e_elapsed:.2f}s]\n")

        return response_text

    def _speak(self, text: str) -> None:
        """Synthesize and play a text response."""
        try:
            print("  [Speaking...]")
            result = self.tts.synthesize(text)
            metrics.tts_latency.observe(result.processing_sec)
            metrics.tts_audio_duration.observe(result.duration_sec)
            play_audio(result.audio_bytes, result.sample_rate)
        except Exception:
            logger.exception("TTS/playback failed")
            print(f"  [TTS error — text response: {text}]")

    def run(self) -> None:
        """Run the voice loop continuously until the user says goodbye."""
        print("\n" + "=" * 50)
        print("  EdgePilot Voice Assistant")
        print("  Fully local — no data leaves this machine")
        print("  Say 'quit' or 'goodbye' to exit")
        print("=" * 50)

        while True:
            try:
                result = self.run_once()
                if result == "EXIT":
                    break
            except KeyboardInterrupt:
                print("\n  [Interrupted]")
                self._speak("Goodbye!")
                break
            except Exception:
                logger.exception("Voice loop error")
                print("  [Error in voice loop, retrying...]")

        print("\nEdgePilot shut down.\n")
