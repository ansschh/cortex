"""Microphone capture — provides a continuous 16kHz mono audio stream."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import AsyncIterator, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512  # ~32ms at 16kHz
DTYPE = "int16"


class MicStream:
    """Continuous microphone capture that yields audio chunks."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        block_size: int = BLOCK_SIZE,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._muted = False
        self._latest_energy: float = 0.0

        # Pre-barge-in ring buffer — captures audio even while muted
        # 3 seconds at 16kHz / 512 block_size = ~94 chunks
        self._ring_buffer_size = int(3.0 * sample_rate / block_size)
        self._ring_buffer: list[np.ndarray] = []
        self._ring_lock = threading.Lock()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status and "input overflow" not in str(status):
            logger.warning(f"Mic status: {status}")
        self._latest_energy = float(np.abs(indata).mean())

        # Always append to ring buffer (even when muted)
        with self._ring_lock:
            self._ring_buffer.append(indata.copy())
            if len(self._ring_buffer) > self._ring_buffer_size:
                self._ring_buffer.pop(0)

        if not self._muted:
            try:
                self._queue.put_nowait(indata.copy())
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(indata.copy())

    def start(self) -> None:
        if self._running:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            dtype=DTYPE,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        logger.info(f"Mic started: {self.sample_rate}Hz, {self.channels}ch, block={self.block_size}")

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False

    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def async_chunks(self) -> AsyncIterator[np.ndarray]:
        """Async generator yielding audio chunks."""
        loop = asyncio.get_event_loop()
        while self._running:
            chunk = await loop.run_in_executor(None, self.get_chunk, 0.05)
            if chunk is not None:
                yield chunk

    def drain(self) -> None:
        """Clear the audio queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def current_energy(self) -> float:
        """Real-time audio energy — works even when muted (for barge-in detection)."""
        return self._latest_energy

    def get_ring_buffer(self) -> list[np.ndarray]:
        """Return a copy of the pre-barge-in ring buffer and clear it."""
        with self._ring_lock:
            chunks = list(self._ring_buffer)
            self._ring_buffer.clear()
            return chunks

    def peek_latest_chunk(self) -> Optional[np.ndarray]:
        """Return the most recent ring buffer chunk without clearing.

        Used by barge-in VAD monitoring to check for speech during playback
        without consuming the buffer (which is still needed for pre-barge-in replay).
        """
        with self._ring_lock:
            if self._ring_buffer:
                return self._ring_buffer[-1].copy()
            return None
