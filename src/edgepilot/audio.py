"""Audio I/O — mic recording (VAD-based silence detection) and speaker playback."""

import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # 16kHz mono, matches Whisper's expected input
CHANNELS = 1
DTYPE = "int16"
BLOCK_SIZE = 1024  # samples per callback block


def record_until_silence(
    silence_threshold: float = 0.02,
    silence_duration: float = 1.5,
    max_duration: float = 30.0,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Record from the default microphone until silence is detected.

    Args:
        silence_threshold: RMS amplitude below which audio is "silent" (0.0-1.0).
        silence_duration: Seconds of continuous silence before stopping.
        max_duration: Maximum recording duration in seconds.
        sample_rate: Recording sample rate.

    Returns:
        Raw PCM int16 mono bytes.
    """
    frames: list[np.ndarray] = []
    silent_blocks = 0
    blocks_for_silence = int(silence_duration * sample_rate / BLOCK_SIZE)
    max_blocks = int(max_duration * sample_rate / BLOCK_SIZE)
    speaking_started = False
    block_count = 0

    logger.info("Recording... (speak now, silence to stop)")

    def callback(indata: np.ndarray, frame_count: int, time_info: object, status: object) -> None:
        nonlocal silent_blocks, speaking_started, block_count
        if status:
            logger.warning("Audio input status: %s", status)
        frames.append(indata.copy())
        block_count += 1

        # Compute RMS of the block
        rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2)) / 32768.0

        if rms > silence_threshold:
            speaking_started = True
            silent_blocks = 0
        elif speaking_started:
            silent_blocks += 1

    with sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        while True:
            sd.sleep(50)  # 50ms poll interval
            if speaking_started and silent_blocks >= blocks_for_silence:
                break
            if block_count >= max_blocks:
                logger.warning("Max recording duration reached (%.0fs)", max_duration)
                break

    if not frames:
        return b""

    audio_np = np.concatenate(frames, axis=0).flatten()
    audio_bytes = audio_np.astype(np.int16).tobytes()

    duration = len(audio_np) / sample_rate
    logger.info("Recorded %.1fs of audio (%d bytes)", duration, len(audio_bytes))
    return audio_bytes


def play_audio(
    audio_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Play raw PCM int16 mono audio through the default speaker."""
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    duration = len(audio_np) / sample_rate
    logger.info("Playing %.1fs of audio...", duration)
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()
