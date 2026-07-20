"""Tests for TTS interface."""

import struct

from edgepilot.tts.base import BaseTTS, SynthesisResult


class MockTTS(BaseTTS):
    """Mock TTS that generates a short sine wave."""

    def __init__(self, sr: int = 22050) -> None:
        self._sample_rate = sr

    def synthesize(self, text: str) -> SynthesisResult:
        # Generate 0.1s of silence as int16 bytes
        n_samples = int(self._sample_rate * 0.1)
        pcm_bytes = struct.pack(f"<{n_samples}h", *([0] * n_samples))
        return SynthesisResult(
            audio_bytes=pcm_bytes,
            sample_rate=self._sample_rate,
            duration_sec=0.1,
            processing_sec=0.01,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class TestSynthesisResult:
    def test_fields(self) -> None:
        r = SynthesisResult(
            audio_bytes=b"\x00\x00",
            sample_rate=16000,
            duration_sec=1.0,
            processing_sec=0.5,
        )
        assert r.sample_rate == 16000
        assert r.duration_sec == 1.0


class TestMockTTS:
    def test_synthesize(self) -> None:
        tts = MockTTS(sr=16000)
        result = tts.synthesize("hello")
        assert len(result.audio_bytes) > 0
        assert result.sample_rate == 16000
        assert result.duration_sec == 0.1

    def test_sample_rate(self) -> None:
        tts = MockTTS(sr=22050)
        assert tts.sample_rate == 22050
