"""Local TTS via piper — runs fully on-device."""

import io
import logging
import time
import wave

from piper import PiperVoice

from edgepilot.tts.base import BaseTTS, SynthesisResult

logger = logging.getLogger(__name__)

# Default model — downloaded on first use
DEFAULT_MODEL = "en_US-lessac-medium"


class PiperTTS(BaseTTS):
    """Text-to-speech using Piper (ONNX-based, fully local).

    Downloads the voice model on first use (~60MB for medium quality).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._voice: PiperVoice | None = None
        self._sample_rate_val: int = 22050

    def _ensure_loaded(self) -> PiperVoice:
        if self._voice is not None:
            return self._voice

        logger.info("Loading Piper voice model: %s", self._model_name)
        try:
            # piper-tts auto-downloads models to ~/.local/share/piper_tts
            from piper.download import ensure_voice_exists, find_voice

            data_dir = None
            try:
                from pathlib import Path

                data_dir = Path.home() / ".local" / "share" / "piper_tts"
                data_dir.mkdir(parents=True, exist_ok=True)
                ensure_voice_exists(self._model_name, [data_dir], data_dir, None)
                onnx_path, config_path = find_voice(self._model_name, [data_dir])
            except Exception:
                # Fallback: try loading by name directly
                raise

            self._voice = PiperVoice.load(str(onnx_path), str(config_path))
            if hasattr(self._voice, "config") and hasattr(self._voice.config, "sample_rate"):
                self._sample_rate_val = self._voice.config.sample_rate
            logger.info("Piper voice loaded (sample_rate=%d)", self._sample_rate_val)
        except Exception:
            logger.exception("Failed to load Piper voice model")
            raise

        return self._voice

    def synthesize(self, text: str) -> SynthesisResult:
        voice = self._ensure_loaded()
        start = time.perf_counter()

        # Synthesize to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize(text, wav_file)

        elapsed = time.perf_counter() - start
        wav_buffer.seek(0)

        # Read the raw PCM data from the WAV
        with wave.open(wav_buffer, "rb") as wav_file:
            n_frames = wav_file.getnframes()
            sr = wav_file.getframerate()
            pcm_bytes = wav_file.readframes(n_frames)
            duration = n_frames / sr

        logger.info(
            "Synthesized %.1fs audio in %.2fs for: '%s'",
            duration,
            elapsed,
            text[:60],
        )
        return SynthesisResult(
            audio_bytes=pcm_bytes,
            sample_rate=sr,
            duration_sec=duration,
            processing_sec=elapsed,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_val
