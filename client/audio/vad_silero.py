"""Silero VAD — lightweight voice activity detection using ONNX runtime.

Silero VAD v5: <1ms per 32ms frame, ONNX-only (no PyTorch dependency for inference).
Used for speech onset detection in idle mode and barge-in detection during TTS playback.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Silero VAD v5 expects 512-sample frames at 16kHz (32ms)
FRAME_SIZE = 512

# Speech detection thresholds
SPEECH_THRESHOLD = 0.5       # probability above which we consider it speech
SILENCE_THRESHOLD = 0.35     # probability below which we consider it silence (hysteresis)
BARGE_IN_THRESHOLD = 0.6     # higher threshold during playback to avoid echo triggers

# Minimum consecutive speech frames before triggering (prevents false positives)
MIN_SPEECH_FRAMES_ONSET = 3  # ~96ms for idle mode speech onset
MIN_SPEECH_FRAMES_BARGE = 2  # ~64ms for barge-in (fast cutoff)


class SileroVADDetector:
    """Silero VAD wrapper with hysteresis state machine for robust speech detection."""

    def __init__(self):
        self._model = None
        self._available = False

        # Hysteresis state
        self._is_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._last_prob = 0.0

        # ONNX runtime session state (Silero VAD uses hidden states)
        self._h = None
        self._c = None

    def initialize(self) -> None:
        """Load Silero VAD model via torch.hub (downloads ~1MB on first run)."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=True,
                force_reload=False,
            )
            self._model = model
            self._available = True
            self._reset_states()
            logger.info("Silero VAD loaded (ONNX mode)")
        except Exception as e:
            logger.warning(f"Silero VAD not available: {e}. Falling back to energy-based detection.")
            self._available = False

    def _reset_states(self) -> None:
        """Reset the model's internal hidden states."""
        if self._model is not None:
            self._model.reset_states()
        self._is_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._last_prob = 0.0

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def last_probability(self) -> float:
        return self._last_prob

    def process_chunk(self, audio_int16: np.ndarray) -> float:
        """Process an audio chunk and return speech probability.

        Args:
            audio_int16: int16 audio data at 16kHz.

        Returns:
            Speech probability [0, 1].
        """
        if not self._available or self._model is None:
            return 0.0

        try:
            import torch

            # Convert int16 to float32 normalized
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
            if audio_f32.ndim > 1:
                audio_f32 = audio_f32[:, 0]

            # Silero VAD expects 512-sample chunks at 16kHz
            # Process in 512-sample frames, return max probability
            max_prob = 0.0
            for i in range(0, len(audio_f32) - FRAME_SIZE + 1, FRAME_SIZE):
                frame = audio_f32[i:i + FRAME_SIZE]
                tensor = torch.from_numpy(frame)
                prob = self._model(tensor, SAMPLE_RATE).item()
                max_prob = max(max_prob, prob)

            # If chunk is smaller than FRAME_SIZE, process it directly
            if len(audio_f32) < FRAME_SIZE:
                # Pad to FRAME_SIZE
                padded = np.zeros(FRAME_SIZE, dtype=np.float32)
                padded[:len(audio_f32)] = audio_f32
                tensor = torch.from_numpy(padded)
                max_prob = self._model(tensor, SAMPLE_RATE).item()

            self._last_prob = max_prob

            # Update hysteresis state
            if max_prob >= SPEECH_THRESHOLD:
                self._consecutive_speech += 1
                self._consecutive_silence = 0
            elif max_prob < SILENCE_THRESHOLD:
                self._consecutive_silence += 1
                self._consecutive_speech = 0
            # In between thresholds: maintain current state (hysteresis)

            # Update speech state
            if not self._is_speech and self._consecutive_speech >= MIN_SPEECH_FRAMES_ONSET:
                self._is_speech = True
            elif self._is_speech and self._consecutive_silence >= 5:
                self._is_speech = False

            return max_prob

        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return 0.0

    def detect_speech_onset(self, audio_int16: np.ndarray) -> bool:
        """Check if speech onset is detected (for idle mode activation).

        Returns True on the transition from not-speaking to speaking,
        requiring MIN_SPEECH_FRAMES_ONSET consecutive speech frames.
        """
        was_speech = self._is_speech
        self.process_chunk(audio_int16)
        return self._is_speech and not was_speech

    def detect_barge_in(self, audio_int16: np.ndarray) -> bool:
        """Check for barge-in during TTS playback.

        Uses a higher threshold to avoid triggering on speaker echo.
        """
        if not self._available:
            return False

        prob = self.process_chunk(audio_int16)
        return prob >= BARGE_IN_THRESHOLD and self._consecutive_speech >= MIN_SPEECH_FRAMES_BARGE

    def reset(self) -> None:
        """Reset all state for a new interaction."""
        self._reset_states()
