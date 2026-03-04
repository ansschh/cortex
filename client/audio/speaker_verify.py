"""Speaker verification using SpeechBrain — enroll once, verify on each command."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

ENROLLMENT_DIR = "data/speaker_enrollment"
ENROLLMENT_EMBEDDING_PATH = os.path.join(ENROLLMENT_DIR, "reference_embedding.npy")
SAMPLE_RATE = 16000


class SpeakerVerifier:
    """Verifies if the current speaker matches the enrolled user."""

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self._model = None
        self._reference_embedding: Optional[np.ndarray] = None
        self._available = False

    def initialize(self) -> None:
        """Load SpeechBrain speaker verification model and reference embedding."""
        try:
            import shutil

            # Patch huggingface_hub to accept deprecated 'use_auth_token' kwarg
            # (speechbrain 1.0.x passes it, but huggingface_hub >=1.0 removed it)
            try:
                import huggingface_hub
                _orig_hf_download = huggingface_hub.hf_hub_download
                def _patched_download(*args, **kwargs):
                    kwargs.pop("use_auth_token", None)
                    return _orig_hf_download(*args, **kwargs)
                huggingface_hub.hf_hub_download = _patched_download

                if hasattr(huggingface_hub, "snapshot_download"):
                    _orig_snap = huggingface_hub.snapshot_download
                    def _patched_snap(*args, **kwargs):
                        kwargs.pop("use_auth_token", None)
                        return _orig_snap(*args, **kwargs)
                    huggingface_hub.snapshot_download = _patched_snap
            except Exception:
                pass

            savedir = os.path.abspath("data/speechbrain_model")

            # On Windows, SpeechBrain's Pretrainer creates symlinks which fail
            # without Developer Mode / admin (WinError 1314).
            # Workaround: temporarily replace os.symlink with shutil.copy2.
            patched = False
            _original_symlink = getattr(os, "symlink", None)
            if _IS_WINDOWS:
                os.symlink = lambda src, dst, *a, **kw: shutil.copy2(src, dst)
                patched = True

            try:
                from speechbrain.inference.speaker import SpeakerRecognition
                # Load from local savedir first (avoids broken HF fetches).
                # Fall back to HF source if local files don't exist yet.
                if os.path.exists(os.path.join(savedir, "hyperparams.yaml")):
                    self._model = SpeakerRecognition.from_hparams(
                        source=savedir,
                        savedir=savedir,
                    )
                else:
                    self._model = SpeakerRecognition.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir=savedir,
                    )
            finally:
                if patched and _original_symlink:
                    os.symlink = _original_symlink

            self._available = True
            logger.info("Speaker verification model loaded.")

            # Load reference embedding if it exists
            if os.path.exists(ENROLLMENT_EMBEDDING_PATH):
                self._reference_embedding = np.load(ENROLLMENT_EMBEDDING_PATH)
                logger.info("Reference speaker embedding loaded.")
            else:
                logger.warning("No speaker enrollment found. Run enrollment first.")

        except Exception as e:
            logger.error(f"Failed to load speaker verification model: {e}")
            self._available = False

    def enroll(self, audio_samples: list[np.ndarray]) -> bool:
        """Enroll the user by computing a reference embedding from audio samples.

        Args:
            audio_samples: List of int16 numpy arrays at 16kHz
        Returns:
            True if enrollment succeeded
        """
        if not self._available or self._model is None:
            logger.error("Speaker model not available for enrollment.")
            return False

        try:
            import torch

            embeddings = []
            for sample in audio_samples:
                if sample.dtype == np.int16:
                    audio_float = sample.astype(np.float32) / 32768.0
                else:
                    audio_float = sample.astype(np.float32)

                tensor = torch.tensor(audio_float).unsqueeze(0)
                embedding = self._model.encode_batch(tensor)
                embeddings.append(embedding.squeeze().numpy())

            # Average all embeddings for a robust reference
            self._reference_embedding = np.mean(embeddings, axis=0)

            # Save to disk
            os.makedirs(ENROLLMENT_DIR, exist_ok=True)
            np.save(ENROLLMENT_EMBEDDING_PATH, self._reference_embedding)
            logger.info(f"Speaker enrolled with {len(audio_samples)} samples.")
            return True

        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            return False

    def verify(self, audio_chunk: np.ndarray) -> dict:
        """Verify if the audio matches the enrolled speaker.

        Args:
            audio_chunk: int16 numpy array at 16kHz (at least ~1 second)

        Returns:
            Dict with is_verified, confidence, and speaker_label
        """
        if not self._available or self._model is None:
            return {"is_verified": False, "confidence": 0.0, "speaker_label": "model_unavailable"}

        if self._reference_embedding is None:
            return {"is_verified": False, "confidence": 0.0, "speaker_label": "not_enrolled"}

        try:
            import torch

            if audio_chunk.dtype == np.int16:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk.astype(np.float32)

            if audio_float.ndim > 1:
                audio_float = audio_float.flatten()

            tensor = torch.tensor(audio_float).unsqueeze(0)
            embedding = self._model.encode_batch(tensor).squeeze().numpy()

            # Cosine similarity
            similarity = float(np.dot(embedding, self._reference_embedding) /
                             (np.linalg.norm(embedding) * np.linalg.norm(self._reference_embedding)))

            is_verified = similarity >= self.threshold

            return {
                "is_verified": is_verified,
                "confidence": similarity,
                "speaker_label": "owner" if is_verified else "unknown",
            }

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"is_verified": False, "confidence": 0.0, "speaker_label": "error"}

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_enrolled(self) -> bool:
        return self._reference_embedding is not None
