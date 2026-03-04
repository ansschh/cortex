"""Presence detection via motion-based frame differencing (OpenCV, no ML)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from server.app.config import settings

logger = logging.getLogger(__name__)


class PresenceMonitor:
    """Background loop that checks for motion via frame differencing.

    v1: simple OpenCV frame diff (no ML)
    v2 (future): YOLOv8-nano for person detection, face_recognition for ID
    """

    def __init__(self, camera_manager, on_presence_change=None):
        self._camera = camera_manager
        self._on_change = on_presence_change  # async callback(present: bool)
        self._present = False
        self._prev_gray = None
        self._task: Optional[asyncio.Task] = None
        self._motion_threshold = 5000  # Minimum contour area to count as motion

    @property
    def is_present(self) -> bool:
        return self._present

    def start(self) -> None:
        if not settings.presence_monitoring_enabled:
            logger.info("Presence monitoring is disabled")
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Presence monitor started (interval={settings.presence_check_interval}s)")

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(settings.presence_check_interval)
                await self._check()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Presence check error: {e}")

    async def _check(self) -> None:
        try:
            jpeg_bytes, _ = await self._camera.capture()
        except Exception as e:
            logger.debug(f"Presence capture failed: {e}")
            return

        motion = await asyncio.to_thread(self._detect_motion, jpeg_bytes)

        if motion and not self._present:
            self._present = True
            logger.info("Presence detected — someone entered")
            if self._on_change:
                await self._on_change(True)
        elif not motion and self._present:
            self._present = False
            logger.info("Presence lost — room appears empty")
            if self._on_change:
                await self._on_change(False)

    def _detect_motion(self, jpeg_bytes: bytes) -> bool:
        """Frame differencing motion detection."""
        import cv2
        import numpy as np

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return False

        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = frame
            return False

        delta = cv2.absdiff(self._prev_gray, frame)
        self._prev_gray = frame

        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) > self._motion_threshold:
                return True

        return False
