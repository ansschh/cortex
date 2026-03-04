"""Camera management — capture frames from USB, IP, or Pi CSI cameras."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CameraType(str, Enum):
    USB = "usb"
    IP = "ip"
    PI_CSI = "pi_csi"


@dataclass
class CameraSource:
    id: str = ""
    name: str = "Default Camera"
    camera_type: CameraType = CameraType.USB
    address: str = "0"  # Device index for USB, URL for IP
    enabled: bool = True


class CameraManager:
    """Manages camera sources and captures frames."""

    def __init__(self):
        self._sources: dict[str, CameraSource] = {}
        self._lock = asyncio.Lock()
        # Persistent capture (kept open for high-frequency reads)
        self._persistent_cap = None
        self._persistent_source_id: Optional[str] = None
        self._persistent_lock = asyncio.Lock()
        # Register a default USB camera
        self.add_source(CameraSource(id="default", name="Default USB Camera", camera_type=CameraType.USB, address="0"))

    def add_source(self, source: CameraSource) -> None:
        self._sources[source.id] = source
        logger.info(f"Camera source registered: {source.name} ({source.camera_type.value})")

    def remove_source(self, camera_id: str) -> bool:
        if camera_id in self._sources:
            del self._sources[camera_id]
            return True
        return False

    def list_sources(self) -> list[CameraSource]:
        return list(self._sources.values())

    async def capture(self, camera_id: Optional[str] = None) -> tuple[bytes, str]:
        """Capture a single JPEG frame. Returns (jpeg_bytes, camera_id).

        Uses asyncio.to_thread to avoid blocking the event loop.
        Lock prevents concurrent access to the same camera device.
        """
        # Pick camera
        if camera_id and camera_id in self._sources:
            source = self._sources[camera_id]
        elif self._sources:
            source = next(iter(self._sources.values()))
        else:
            raise RuntimeError("No camera sources registered")

        if not source.enabled:
            raise RuntimeError(f"Camera '{source.name}' is disabled")

        async with self._lock:
            jpeg_bytes = await asyncio.to_thread(self._capture_sync, source)

        return jpeg_bytes, source.id

    async def stream_frames(self, camera_id: Optional[str] = None, fps: int = 10):
        """Async generator yielding JPEG frames for MJPEG streaming.

        Keeps the camera open for the duration of the stream.
        """
        if camera_id and camera_id in self._sources:
            source = self._sources[camera_id]
        elif self._sources:
            source = next(iter(self._sources.values()))
        else:
            raise RuntimeError("No camera sources registered")

        if not source.enabled:
            raise RuntimeError(f"Camera '{source.name}' is disabled")

        import cv2

        delay = 1.0 / fps

        # Open camera once, yield frames until cancelled
        cap = self._open_capture(source)
        try:
            while True:
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    logger.warning(f"Frame read failed from {source.name}, retrying...")
                    await asyncio.sleep(0.5)
                    continue
                _, buffer = await asyncio.to_thread(cv2.imencode, ".jpg", frame)
                yield buffer.tobytes()
                await asyncio.sleep(delay)
        finally:
            cap.release()
            logger.info(f"Camera stream closed: {source.name}")

    # ------------------------------------------------------------------
    # Persistent capture — keeps camera open for high-frequency reads
    # ------------------------------------------------------------------

    async def open_persistent(self, camera_id: Optional[str] = None) -> None:
        """Open a camera and keep it open for fast repeated captures.

        Call capture_fast() to read frames without open/close overhead.
        """
        source = self._resolve_source(camera_id)
        async with self._persistent_lock:
            if self._persistent_cap is not None:
                self._persistent_cap.release()
            self._persistent_cap = await asyncio.to_thread(self._open_capture, source)
            self._persistent_source_id = source.id
            # Warm up: discard initial dark frames
            for _ in range(5):
                await asyncio.to_thread(self._persistent_cap.read)
            logger.info(f"Persistent camera opened: {source.name}")

    async def capture_fast(self) -> tuple[bytes, str]:
        """Read a frame from the persistent camera. ~5-10ms, no open/close.

        Falls back to regular capture() if persistent camera not open.
        """
        async with self._persistent_lock:
            if self._persistent_cap is None or not self._persistent_cap.isOpened():
                return await self.capture()
            jpeg_bytes = await asyncio.to_thread(self._read_persistent)
            return jpeg_bytes, self._persistent_source_id or "default"

    def _read_persistent(self) -> bytes:
        """Sync read from persistent camera (runs in thread)."""
        import cv2
        ret, frame = self._persistent_cap.read()
        if not ret:
            raise RuntimeError("Failed to read from persistent camera")
        _, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes()

    async def close_persistent(self) -> None:
        """Close the persistent camera."""
        async with self._persistent_lock:
            if self._persistent_cap is not None:
                self._persistent_cap.release()
                self._persistent_cap = None
                self._persistent_source_id = None
                logger.info("Persistent camera closed")

    def _resolve_source(self, camera_id: Optional[str] = None) -> CameraSource:
        """Find the camera source by ID or return the first available."""
        if camera_id and camera_id in self._sources:
            return self._sources[camera_id]
        if self._sources:
            return next(iter(self._sources.values()))
        raise RuntimeError("No camera sources registered")

    @staticmethod
    def _open_capture(source: CameraSource):
        """Open a cv2.VideoCapture for the given source."""
        import cv2

        if source.camera_type == CameraType.USB:
            return cv2.VideoCapture(int(source.address))
        elif source.camera_type == CameraType.IP:
            return cv2.VideoCapture(source.address)
        elif source.camera_type == CameraType.PI_CSI:
            return cv2.VideoCapture(0)
        else:
            raise RuntimeError(f"Unknown camera type: {source.camera_type}")

    @staticmethod
    def _capture_sync(source: CameraSource) -> bytes:
        """Synchronous frame capture (runs in thread pool)."""
        import cv2

        if source.camera_type == CameraType.USB:
            device = int(source.address)
            cap = cv2.VideoCapture(device)
        elif source.camera_type == CameraType.IP:
            cap = cv2.VideoCapture(source.address)
        elif source.camera_type == CameraType.PI_CSI:
            # libcamera / picamera2 would go here; fallback to device 0
            cap = cv2.VideoCapture(0)
        else:
            raise RuntimeError(f"Unknown camera type: {source.camera_type}")

        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture frame from {source.name}")
            _, buffer = cv2.imencode(".jpg", frame)
            return buffer.tobytes()
        finally:
            cap.release()
