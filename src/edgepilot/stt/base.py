"""Abstract STT interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result from speech-to-text."""

    text: str
    language: str = "en"
    duration_sec: float = 0.0
    processing_sec: float = 0.0

    @property
    def realtime_factor(self) -> float:
        """Processing time / audio duration. < 1.0 means faster than real-time."""
        if self.duration_sec > 0:
            return self.processing_sec / self.duration_sec
        return 0.0


class BaseSTT(ABC):
    """Abstract speech-to-text. Subclasses implement transcribe()."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file to text."""
        ...

    @abstractmethod
    def transcribe_bytes(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe raw audio bytes (PCM int16 mono) to text."""
        ...
