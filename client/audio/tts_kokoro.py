"""Local Kokoro TTS — low-latency ONNX-based text-to-speech for FastPath responses.

Uses the Kokoro-82M model (quantized, ~88MB) for sub-200ms first-audio latency,
replacing ElevenLabs round-trips for short template responses.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000


class KokoroLocalTTS:
    """Local TTS using Kokoro-82M ONNX for FastPath short responses.

    Produces int16 PCM audio at 24kHz, matching ElevenLabs output format.
    """

    def __init__(
        self,
        model_dir: str = "data/models/kokoro",
        voice: str = "af_nova",
    ):
        model_path = Path(model_dir)
        onnx_path = model_path / "onnx" / "model_quantized.onnx"
        voice_path = model_path / "voices" / f"{voice}.bin"

        if not onnx_path.exists():
            raise FileNotFoundError(f"Kokoro model not found: {onnx_path}")
        if not voice_path.exists():
            raise FileNotFoundError(f"Kokoro voice not found: {voice_path}")

        # Load ONNX session
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = self._session.get_providers()
        logger.info(f"Kokoro TTS loaded | providers={active}")

        # Load voice embedding — raw float32 array (510 style vectors of 256 dims)
        self._voice_data = np.fromfile(str(voice_path), dtype=np.float32).reshape(-1, 256)
        self._voice_name = voice
        logger.info(f"Kokoro voice loaded: {voice} ({self._voice_data.shape[0]} styles)")

        # Import kokoro tokenizer and phonemizer
        from kokoro_onnx.tokenizer import Tokenizer
        from kokoro_onnx.config import EspeakConfig, MAX_PHONEME_LENGTH
        self._tokenizer = Tokenizer(EspeakConfig())
        self._max_phoneme_length = MAX_PHONEME_LENGTH

        # Check ONNX input names (newer exports use "input_ids")
        input_names = [i.name for i in self._session.get_inputs()]
        self._use_input_ids = "input_ids" in input_names

        self._ready = True
        logger.info(f"Kokoro TTS ready | voice={voice} | inputs={input_names}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def synthesize_sync(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize text to int16 PCM audio (blocking).

        Returns int16 numpy array at 24kHz, ready for sounddevice output.
        """
        t0 = time.perf_counter()

        # Text → phonemes → tokens
        phonemes = self._tokenizer.phonemize(text, "en-us")
        if not phonemes:
            return np.array([], dtype=np.int16)

        # Truncate phonemes if too long
        phonemes = phonemes[:self._max_phoneme_length]
        tokens = self._tokenizer.tokenize(phonemes)

        if not tokens:
            return np.array([], dtype=np.int16)

        # Get style vector for this token length
        style_idx = min(len(tokens), self._voice_data.shape[0] - 1)
        style = self._voice_data[style_idx]

        # Wrap tokens with pad token 0 at start/end
        padded_tokens = [[0] + tokens + [0]]

        # Build ONNX inputs — all inputs need correct shapes and dtypes
        token_array = np.array(padded_tokens, dtype=np.int64)
        style_array = style[np.newaxis, :] if style.ndim == 1 else style
        speed_array = np.array([speed], dtype=np.float32)

        if self._use_input_ids:
            inputs = {
                "input_ids": token_array,
                "style": style_array,
                "speed": speed_array,
            }
        else:
            inputs = {
                "tokens": token_array,
                "style": style_array,
                "speed": speed_array,
            }

        # Run inference
        result = self._session.run(None, inputs)
        audio_f32 = result[0].squeeze()

        # Trim leading/trailing silence
        audio_f32 = _trim_silence(audio_f32)

        # Convert float32 to int16
        audio_int16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration = len(audio_int16) / KOKORO_SAMPLE_RATE
        logger.info(f"Kokoro synth: {elapsed_ms:.0f}ms for {duration:.2f}s audio | \"{text[:60]}\"")

        return audio_int16

    async def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Async wrapper — runs synthesis in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize_sync, text, speed)


def _trim_silence(audio: np.ndarray, threshold: float = 0.01, min_samples: int = 2400) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    if len(audio) < min_samples:
        return audio
    abs_audio = np.abs(audio)
    # Find first sample above threshold
    above = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return audio
    start = max(0, above[0] - 240)  # 10ms lead-in
    end = min(len(audio), above[-1] + 240)  # 10ms lead-out
    return audio[start:end]
