"""Wake word detection using openWakeWord."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default wake word models that ship with openWakeWord
DEFAULT_MODELS = ["hey_jarvis"]
DETECTION_THRESHOLD = 0.5


class WakeWordDetector:
    """Wraps openWakeWord for continuous wake word detection on audio chunks."""

    def __init__(
        self,
        models: Optional[list[str]] = None,
        threshold: float = DETECTION_THRESHOLD,
    ):
        self.threshold = threshold
        self._model_names = models or DEFAULT_MODELS
        self._model = None
        self._enabled = True

    def initialize(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model

            openwakeword.utils.download_models()
            self._model = Model(
                wakeword_models=self._model_names,
                inference_framework="onnx",
            )
            logger.info(f"Wake word detector initialized with models: {self._model_names}")
        except Exception as e:
            logger.error(f"Failed to initialize wake word detector: {e}")
            if "DLL load failed" in str(e) or "onnxruntime" in str(e):
                logger.warning(
                    "onnxruntime DLL issue detected — known on Python 3.13/Windows. "
                    "Wake word disabled. Use push-to-talk (type 'ptt' in console) or "
                    "try: pip install onnxruntime --force-reinstall"
                )
            logger.info("Falling back to push-to-talk mode. Commands: 't' to talk, 'say <text>' for text input, 'enroll' for speaker, 'status', 'q' to quit.")
            self._model = None

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[dict]:
        """Process an audio chunk and return detection result if wake word detected.

        Args:
            audio_chunk: int16 numpy array at 16kHz mono

        Returns:
            Dict with model_name and confidence if detected, None otherwise.
        """
        if not self._enabled or self._model is None:
            return None

        # openWakeWord expects int16 audio
        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk * 32767).astype(np.int16)

        # Flatten if multi-dimensional
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()

        prediction = self._model.predict(audio_chunk)

        for model_name, score in prediction.items():
            if score >= self.threshold:
                logger.info(f"Wake word detected: {model_name} (confidence={score:.3f})")
                self._model.reset()
                return {"model_name": model_name, "confidence": float(score)}

        return None

    def reset(self) -> None:
        if self._model:
            self._model.reset()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_available(self) -> bool:
        return self._model is not None
