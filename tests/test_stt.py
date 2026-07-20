"""Tests for STT interface and Whisper integration."""

import numpy as np

from edgepilot.stt.base import BaseSTT, TranscriptionResult


class MockSTT(BaseSTT):
    """Mock STT that returns a fixed transcription."""

    def __init__(self, text: str = "hello world") -> None:
        self._text = text

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        return TranscriptionResult(
            text=self._text,
            language="en",
            duration_sec=2.0,
            processing_sec=0.5,
        )

    def transcribe_bytes(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
        duration = len(audio_data) / 2 / sample_rate  # int16 = 2 bytes/sample
        return TranscriptionResult(
            text=self._text,
            language="en",
            duration_sec=duration,
            processing_sec=0.1,
        )


class TestTranscriptionResult:
    def test_realtime_factor(self) -> None:
        r = TranscriptionResult(
            text="hi", duration_sec=4.0, processing_sec=1.0
        )
        assert r.realtime_factor == 0.25

    def test_realtime_factor_zero_duration(self) -> None:
        r = TranscriptionResult(text="", duration_sec=0.0, processing_sec=0.0)
        assert r.realtime_factor == 0.0


class TestMockSTT:
    def test_transcribe_path(self) -> None:
        stt = MockSTT("what time is it")
        result = stt.transcribe("fake_path.wav")
        assert result.text == "what time is it"
        assert result.language == "en"

    def test_transcribe_bytes(self) -> None:
        stt = MockSTT("hello")
        # 1 second of silence at 16kHz int16 mono
        audio = np.zeros(16000, dtype=np.int16).tobytes()
        result = stt.transcribe_bytes(audio, sample_rate=16000)
        assert result.text == "hello"
        assert abs(result.duration_sec - 1.0) < 0.01

    def test_empty_audio(self) -> None:
        stt = MockSTT("")
        result = stt.transcribe_bytes(b"", sample_rate=16000)
        assert result.text == ""
        assert result.duration_sec == 0.0
