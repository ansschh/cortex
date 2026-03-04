"""Local GPU Whisper STT using faster-whisper — VAD-gated batch transcription.

Replaces AssemblyAI streaming STT. Audio chunks are buffered locally.
A background task runs Whisper periodically for partial transcripts.
Final transcripts are triggered by force_endpoint() when the client's
VAD + Smart Turn detects end-of-utterance.

Barge-in is trivial: pause() sets a flag, resume() clears the buffer.
No WebSocket, no network, no segfault risk.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Default model configuration
DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_DEVICE = "cuda"

# Partial transcript timing
PARTIAL_INTERVAL_S = 0.8        # Run partial transcription every 0.8s
MIN_AUDIO_FOR_PARTIAL_S = 0.5   # Don't transcribe < 0.5s of audio
MIN_AUDIO_FOR_FINAL_S = 0.3     # Minimum audio for final transcription
MAX_PARTIAL_SECONDS = 10        # Sliding window: only transcribe last N seconds for partials

# Custom vocabulary prompt — biases Whisper toward these words
DEFAULT_VOCAB_PROMPT = (
    "Jarvis, Nova, Hey Jarvis, calorie, calories, "
    "dorm room, dormitory, entrepreneurship"
)


class WhisperLocalSTT:
    """Local GPU Whisper STT with callback-driven interface.

    Drop-in replacement for AssemblyAIStreamingSTT. Audio chunks are
    buffered locally, Whisper runs on GPU for transcription.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        vocab_prompt: str = DEFAULT_VOCAB_PROMPT,
        language: str = "en",
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._vocab_prompt = vocab_prompt
        self._language = language

        self._model = None  # faster_whisper.WhisperModel
        self._connected = False
        self._paused = False

        # Audio buffer — accumulated speech audio (int16, 16kHz)
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_lock = asyncio.Lock()

        # Transcript state
        self._current_partial: str = ""

        # Callbacks (set by client_main before connect)
        self.on_partial: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_final: Optional[Callable[[str], Awaitable[None]]] = None

        # Background tasks
        self._partial_task: Optional[asyncio.Task] = None

        # Thread pool for blocking Whisper inference (single worker = no GPU contention)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

        # Partial dedup
        self._last_partial_text: str = ""

        # Guard against concurrent transcriptions
        self._transcribing = False

    async def connect(self) -> None:
        """Load the Whisper model onto GPU. Call once per session."""
        if self._connected:
            return

        loop = asyncio.get_event_loop()

        # Auto-detect CUDA availability and fall back to CPU
        device = self._device
        compute_type = self._compute_type
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available — falling back to CPU")
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            if device == "cuda":
                logger.warning("torch not installed — falling back to CPU for Whisper")
                device = "cpu"
                compute_type = "int8"

        logger.info(
            f"Loading faster-whisper model '{self._model_size}' "
            f"on {device} ({compute_type})..."
        )

        _dev, _ct = device, compute_type  # capture for closure

        def _load_model():
            from faster_whisper import WhisperModel
            return WhisperModel(
                self._model_size,
                device=_dev,
                compute_type=_ct,
            )

        try:
            self._model = await loop.run_in_executor(self._executor, _load_model)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

        self._connected = True
        self._paused = False
        self._audio_buffer.clear()
        self._current_partial = ""
        self._last_partial_text = ""

        # Start background partial transcription loop
        self._partial_task = asyncio.create_task(self._partial_loop())

        logger.info(f"Whisper model loaded on {self._device}")

    async def send_audio(self, audio_chunk: np.ndarray) -> None:
        """Buffer an audio chunk for transcription. No-op if paused."""
        if not self._connected or self._paused:
            return

        # Ensure int16
        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk * 32767).astype(np.int16)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()

        async with self._buffer_lock:
            self._audio_buffer.append(audio_chunk.copy())

    def pause(self) -> None:
        """Stop accepting audio (during SPEAKING). Model stays loaded."""
        self._paused = True
        logger.debug("Whisper STT paused")

    def resume(self) -> None:
        """Resume accepting audio. Clears buffer for fresh utterance."""
        self._paused = False
        self._current_partial = ""
        self._last_partial_text = ""
        # Clear buffer synchronously — safe because pause was set
        self._audio_buffer.clear()
        logger.debug("Whisper STT resumed")

    async def set_endpointing_threshold(self, ms: int) -> None:
        """No-op for compatibility. Endpointing handled by VAD + Smart Turn."""
        pass

    async def force_endpoint(self) -> None:
        """Transcribe the current buffer as a final transcript and clear it."""
        if not self._connected or not self._model:
            return

        async with self._buffer_lock:
            if not self._audio_buffer:
                return
            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer.clear()

        duration_s = len(audio) / SAMPLE_RATE
        if duration_s < MIN_AUDIO_FOR_FINAL_S:
            logger.debug(f"Audio too short for final ({duration_s:.2f}s), skipping")
            return

        start = time.monotonic()
        text = await self._transcribe(audio)
        elapsed_ms = (time.monotonic() - start) * 1000

        if text and text.strip():
            logger.info(
                f"Whisper final ({duration_s:.1f}s audio, {elapsed_ms:.0f}ms inference): "
                f"'{text[:80]}'"
            )
            self._current_partial = ""
            self._last_partial_text = ""
            if self.on_final:
                await self.on_final(text)
        else:
            logger.debug(f"Whisper returned empty for {duration_s:.1f}s audio")

    async def _partial_loop(self) -> None:
        """Background: periodically transcribe the growing buffer for partials."""
        try:
            while self._connected:
                await asyncio.sleep(PARTIAL_INTERVAL_S)

                if self._paused or not self._connected:
                    continue

                # Skip if already transcribing
                if self._transcribing:
                    continue

                async with self._buffer_lock:
                    if not self._audio_buffer:
                        continue
                    audio = np.concatenate(self._audio_buffer)

                duration_s = len(audio) / SAMPLE_RATE
                if duration_s < MIN_AUDIO_FOR_PARTIAL_S:
                    continue

                # Sliding window: only transcribe last MAX_PARTIAL_SECONDS for partials
                # This caps inference at ~500ms regardless of utterance length
                if duration_s > MAX_PARTIAL_SECONDS:
                    samples_to_keep = int(MAX_PARTIAL_SECONDS * SAMPLE_RATE)
                    audio = audio[-samples_to_keep:]

                text = await self._transcribe(audio)

                if text and text.strip() and text != self._last_partial_text:
                    self._current_partial = text
                    self._last_partial_text = text
                    if self.on_partial and not self._paused:
                        await self.on_partial(text)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Partial transcription loop error: {e}")

    async def _transcribe(self, audio_int16: np.ndarray) -> str:
        """Run faster-whisper inference on audio. Runs in thread pool."""
        if not self._model:
            return ""

        self._transcribing = True
        loop = asyncio.get_event_loop()

        try:
            def _run_whisper():
                # Convert int16 to float32 normalized [-1, 1]
                audio_f32 = audio_int16.astype(np.float32) / 32768.0

                segments, _info = self._model.transcribe(
                    audio_f32,
                    language=self._language,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    initial_prompt=self._vocab_prompt,
                    vad_filter=False,       # We do our own VAD
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                )

                texts = []
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        texts.append(text)

                return " ".join(texts).strip()

            text = await loop.run_in_executor(self._executor, _run_whisper)
            return text

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""
        finally:
            self._transcribing = False

    @property
    def current_partial(self) -> str:
        return self._current_partial

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def close(self) -> None:
        """Unload model and free GPU memory."""
        self._connected = False
        self._paused = False

        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()

        self._model = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self._audio_buffer.clear()
        self._current_partial = ""

        logger.info("Whisper STT closed, GPU memory released")
