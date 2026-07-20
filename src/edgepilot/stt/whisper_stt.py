"""Local STT via faster-whisper — runs fully on-device."""

import logging
import time

import numpy as np
from faster_whisper import WhisperModel

from edgepilot.stt.base import BaseSTT, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperSTT(BaseSTT):
    """Speech-to-text using faster-whisper (CTranslate2).

    Downloads the model on first use (~150MB for 'base').
    Runs on CPU by default — set device='cuda' for GPU.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        logger.info(
            "Loading Whisper model: %s (device=%s, compute=%s)",
            model_size, device, compute_type,
        )
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        logger.info("Whisper model loaded.")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        start = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_path, beam_size=5, language="en"
        )
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - start

        logger.info(
            "Transcribed %.1fs audio in %.2fs: '%s'",
            info.duration, elapsed, text[:80],
        )
        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_sec=info.duration,
            processing_sec=elapsed,
        )

    def transcribe_bytes(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
        # Convert raw PCM int16 bytes to float32 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(
            np.float32
        ) / 32768.0
        duration = len(audio_np) / sample_rate

        start = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_np, beam_size=5, language="en"
        )
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - start

        logger.info(
            "Transcribed %.1fs audio in %.2fs: '%s'",
            duration, elapsed, text[:80],
        )
        return TranscriptionResult(
            text=text,
            language="en",
            duration_sec=duration,
            processing_sec=elapsed,
        )
