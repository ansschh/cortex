"""Two-tier scene analysis — fast OpenCV + rich LLM description.

Fast tier (every ~0.5s): Pure OpenCV, no ML, ~5ms per frame.
  - Motion magnitude and bounding region
  - Brightness/color histogram change (detect lights on/off)
  - Person presence via MOG2 background subtraction

Rich tier (on-demand): SmolVLM local model, ~1s.
  - Full natural-language scene description
  - Only triggered when fast tier detects significant change
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneState:
    """Structured output from fast-tier analysis."""
    timestamp: float = 0.0
    motion_detected: bool = False
    motion_magnitude: float = 0.0         # Total contour area of motion
    motion_region: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) bounding box
    brightness: float = 0.0               # Mean frame brightness (0-255)
    brightness_delta: float = 0.0         # Change from previous frame
    person_present: bool = False          # MOG2 foreground detection
    person_area_fraction: float = 0.0     # Fraction of frame occupied by foreground
    scene_stable_seconds: float = 0.0     # How long scene has been unchanged

    @property
    def is_significant_change(self) -> bool:
        """Whether this state represents a noteworthy scene change."""
        return (
            self.motion_magnitude > 5000
            or abs(self.brightness_delta) > 30
        )


class SceneAnalyzer:
    """Fast local scene analysis using OpenCV only (no ML, no VRAM)."""

    def __init__(self, motion_threshold: int = 3000):
        self._motion_threshold = motion_threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_brightness: float = 0.0
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        self._last_change_time: float = time.time()
        self._frame_count: int = 0

    def analyze(self, jpeg_bytes: bytes) -> SceneState:
        """Run fast-tier analysis on a JPEG frame. ~5ms, CPU only.

        Thread-safe: call from asyncio.to_thread().
        """
        self._frame_count += 1
        now = time.time()

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return SceneState(timestamp=now)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

        state = SceneState(timestamp=now)

        # --- Brightness ---
        state.brightness = float(np.mean(gray))
        state.brightness_delta = state.brightness - self._prev_brightness
        self._prev_brightness = state.brightness

        # --- Motion detection via frame differencing ---
        if self._prev_gray is not None:
            delta = cv2.absdiff(self._prev_gray, gray_blur)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_area = 0.0
            min_x, min_y = frame.shape[1], frame.shape[0]
            max_x, max_y = 0, 0

            for c in contours:
                area = cv2.contourArea(c)
                if area > 500:  # Ignore tiny noise
                    total_area += area
                    x, y, w, h = cv2.boundingRect(c)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)

            state.motion_magnitude = total_area
            state.motion_detected = total_area > self._motion_threshold

            if state.motion_detected and max_x > 0:
                state.motion_region = (min_x, min_y, max_x - min_x, max_y - min_y)
                self._last_change_time = now

        self._prev_gray = gray_blur

        # --- Person presence via MOG2 background subtraction ---
        fg_mask = self._bg_subtractor.apply(frame)
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        fg_pixels = np.count_nonzero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        state.person_area_fraction = fg_pixels / total_pixels if total_pixels > 0 else 0.0
        # Heuristic: if >3% of frame is foreground, someone is probably there
        state.person_present = state.person_area_fraction > 0.03

        # --- Scene stability ---
        state.scene_stable_seconds = now - self._last_change_time

        return state

    def format_state(self, state: SceneState) -> str:
        """Format scene state as a concise text summary for context injection."""
        parts = []
        if state.person_present:
            parts.append(f"person present ({state.person_area_fraction:.0%} of frame)")
        else:
            parts.append("no person detected")

        if state.motion_detected:
            parts.append(f"motion detected (magnitude: {state.motion_magnitude:.0f})")
        elif state.scene_stable_seconds > 5:
            parts.append(f"scene stable for {state.scene_stable_seconds:.0f}s")

        brightness_label = "dark" if state.brightness < 50 else "dim" if state.brightness < 100 else "normal" if state.brightness < 180 else "bright"
        parts.append(f"lighting: {brightness_label} ({state.brightness:.0f}/255)")

        return " | ".join(parts)
