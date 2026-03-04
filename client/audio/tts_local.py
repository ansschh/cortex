"""Local Piper TTS — ultra-low-latency VITS-based text-to-speech for FastPath responses.

Uses Piper (VITS architecture, single forward pass) for 50-150ms synthesis,
replacing ElevenLabs round-trips for short template responses.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PIPER_SAMPLE_RATE = 22050


class LocalTTS:
    """Local TTS using Piper for FastPath short responses.

    Produces int16 PCM audio at 22050Hz. Client playback must resample
    if needed (sounddevice handles this automatically).
    """

    def __init__(
        self,
        model_path: str = "data/models/piper/en_US-amy-medium.onnx",
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Piper model not found: {path}")

        from piper import PiperVoice
        self._voice = PiperVoice.load(str(path))
        self._sample_rate = self._voice.config.sample_rate
        self._ready = True
        logger.info(f"Piper TTS loaded | model={path.name} | sr={self._sample_rate}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize_sync(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize text to int16 PCM audio (blocking).

        Returns int16 numpy array at model sample rate (22050Hz).
        """
        t0 = time.perf_counter()

        all_audio = []
        for chunk in self._voice.synthesize(text):
            all_audio.append(chunk.audio_int16_array)

        if not all_audio:
            return np.array([], dtype=np.int16)

        audio = np.concatenate(all_audio)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration = len(audio) / self._sample_rate
        logger.info(f"Piper synth: {elapsed_ms:.0f}ms for {duration:.2f}s audio | \"{text[:60]}\"")

        return audio

    async def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Async wrapper — runs synthesis in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize_sync, text, speed)
