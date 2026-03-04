"""Audio playback — plays PCM audio chunks in real-time using sounddevice."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Plays PCM audio with support for streaming chunks and barge-in cancellation."""

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        dtype: str = "int16",
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self._playing = False
        self._cancel = False
        self._stream: Optional[sd.OutputStream] = None
        self._audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning(f"Playback status: {status}")

        try:
            data = self._audio_queue.get_nowait()
        except queue.Empty:
            data = None

        if data is None or self._cancel:
            outdata.fill(0)
            return

        audio = np.frombuffer(data, dtype=np.int16)
        if len(audio) < frames:
            padded = np.zeros(frames, dtype=np.int16)
            padded[:len(audio)] = audio
            outdata[:, 0] = padded
        else:
            outdata[:, 0] = audio[:frames]

    async def play_audio(self, audio_bytes: bytes) -> None:
        """Play a complete audio buffer."""
        self._cancel = False
        self._playing = True

        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            chunk_size = 1024
            sd.play(audio, samplerate=self.sample_rate, device=self.device)

            # Wait for playback to complete or be cancelled
            duration = len(audio) / self.sample_rate
            elapsed = 0.0
            while elapsed < duration and not self._cancel:
                await asyncio.sleep(0.05)
                elapsed += 0.05

            if self._cancel:
                sd.stop()
                logger.info("Playback cancelled.")
        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            self._playing = False

    async def play_chunks(self, chunk_generator) -> None:
        """Play audio chunks as they arrive from a streaming source."""
        self._cancel = False
        self._playing = True
        buffer = bytearray()

        try:
            async for chunk in chunk_generator:
                if self._cancel:
                    break
                buffer.extend(chunk)

                # Play when we have enough data
                if len(buffer) >= 4800:  # ~100ms at 24kHz 16-bit
                    audio = np.frombuffer(bytes(buffer), dtype=np.int16)
                    sd.play(audio, samplerate=self.sample_rate, device=self.device, blocking=False)
                    # Wait for chunk to play
                    duration = len(audio) / self.sample_rate
                    await asyncio.sleep(duration * 0.9)
                    buffer.clear()

            # Play remaining buffer
            if buffer and not self._cancel:
                audio = np.frombuffer(bytes(buffer), dtype=np.int16)
                sd.play(audio, samplerate=self.sample_rate, device=self.device)
                duration = len(audio) / self.sample_rate
                await asyncio.sleep(duration)

        except Exception as e:
            logger.error(f"Streaming playback error: {e}")
        finally:
            self._playing = False

    def cancel(self) -> None:
        """Stop current playback (barge-in)."""
        self._cancel = True
        sd.stop()

    @property
    def is_playing(self) -> bool:
        return self._playing
