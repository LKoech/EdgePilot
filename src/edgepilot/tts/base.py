"""Abstract TTS interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SynthesisResult:
    """Result from text-to-speech."""

    audio_bytes: bytes
    sample_rate: int
    duration_sec: float
    processing_sec: float


class BaseTTS(ABC):
    """Abstract text-to-speech. Subclasses implement synthesize()."""

    @abstractmethod
    def synthesize(self, text: str) -> SynthesisResult:
        """Convert text to speech audio (PCM int16 mono bytes)."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the output sample rate."""
        ...
