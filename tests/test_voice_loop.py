"""Tests for the voice loop — uses mocks, no real audio hardware needed."""

import struct
from unittest.mock import MagicMock, patch

import numpy as np

from edgepilot.executors.base import ToolCall
from edgepilot.executors.get_time import GetTimeExecutor
from edgepilot.executors.registry import ExecutorRegistry
from edgepilot.planner.base import BasePlanner, PlannerResult
from edgepilot.recovery.controller import RecoveryController
from edgepilot.stt.base import BaseSTT, TranscriptionResult
from edgepilot.tts.base import BaseTTS, SynthesisResult
from edgepilot.voice_loop import VoiceLoop


class MockSTT(BaseSTT):
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self._index = 0

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        return self._next()

    def transcribe_bytes(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
        return self._next()

    def _next(self) -> TranscriptionResult:
        if self._index < len(self._texts):
            text = self._texts[self._index]
            self._index += 1
        else:
            text = "quit"
        return TranscriptionResult(text=text, duration_sec=1.0, processing_sec=0.1)


class MockTTS(BaseTTS):
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> SynthesisResult:
        self.spoken.append(text)
        n_samples = 1000
        pcm = struct.pack(f"<{n_samples}h", *([0] * n_samples))
        return SynthesisResult(
            audio_bytes=pcm,
            sample_rate=22050,
            duration_sec=0.05,
            processing_sec=0.01,
        )

    @property
    def sample_rate(self) -> int:
        return 22050


class FixedPlanner(BasePlanner):
    def plan(
        self, user_input: str, tool_descriptions: list[dict], context: str = ""
    ) -> PlannerResult:
        return PlannerResult(
            tool_call=ToolCall(tool_name="get_time", arguments={}),
        )


def _build_voice_loop(
    stt_texts: list[str],
) -> tuple[VoiceLoop, MockTTS]:
    registry = ExecutorRegistry()
    registry.register(GetTimeExecutor())
    planner = FixedPlanner()
    ctrl = RecoveryController(planner, registry, max_retries=1)

    stt = MockSTT(stt_texts)
    tts = MockTTS()
    loop = VoiceLoop(stt=stt, tts=tts, controller=ctrl)
    return loop, tts


class TestVoiceLoop:
    @patch("edgepilot.voice_loop.record_until_silence")
    @patch("edgepilot.voice_loop.play_audio")
    def test_run_once_success(
        self, mock_play: MagicMock, mock_record: MagicMock
    ) -> None:
        """Single voice cycle: record -> transcribe -> plan -> execute -> speak."""
        # Simulate 1s of audio
        mock_record.return_value = np.zeros(16000, dtype=np.int16).tobytes()
        mock_play.return_value = None

        loop, tts = _build_voice_loop(["what time is it"])
        result = loop.run_once()

        assert result is not None
        assert "Current local time" in result
        assert len(tts.spoken) == 1  # TTS was called once
        mock_record.assert_called_once()
        mock_play.assert_called_once()

    @patch("edgepilot.voice_loop.record_until_silence")
    @patch("edgepilot.voice_loop.play_audio")
    def test_empty_audio_returns_none(
        self, mock_play: MagicMock, mock_record: MagicMock
    ) -> None:
        mock_record.return_value = b""
        loop, tts = _build_voice_loop([])
        result = loop.run_once()
        assert result is None
        assert len(tts.spoken) == 0

    @patch("edgepilot.voice_loop.record_until_silence")
    @patch("edgepilot.voice_loop.play_audio")
    def test_quit_command(
        self, mock_play: MagicMock, mock_record: MagicMock
    ) -> None:
        mock_record.return_value = np.zeros(16000, dtype=np.int16).tobytes()
        loop, tts = _build_voice_loop(["goodbye"])
        result = loop.run_once()
        assert result == "EXIT"
        assert any("Goodbye" in s for s in tts.spoken)

    @patch("edgepilot.voice_loop.record_until_silence")
    @patch("edgepilot.voice_loop.play_audio")
    def test_tts_failure_doesnt_crash(
        self, mock_play: MagicMock, mock_record: MagicMock
    ) -> None:
        """If TTS fails, the loop should still return the text response."""
        mock_record.return_value = np.zeros(16000, dtype=np.int16).tobytes()
        mock_play.side_effect = Exception("Speaker not found")

        loop, tts = _build_voice_loop(["what time is it"])
        # Override TTS to also raise
        tts.synthesize = MagicMock(  # type: ignore[method-assign]
            side_effect=Exception("TTS broken")
        )
        # Should not raise despite TTS/playback failure
        result = loop.run_once()
        assert result is not None
        assert "Current local time" in result
