"""Always-on vision context — continuous scene awareness for NOVA.

Three decoupled async loops for minimum latency:
1. FAST capture loop (every 0.1s) — grabs frames via persistent camera,
   runs lightweight OpenCV analysis (motion, brightness, person presence).
2. DESCRIBE loop (on-demand) — picks up latest frame when scene changes,
   sends to vision LLM for a one-sentence description. Skips when static.
3. EVENT publisher — emits structured events to the EventBus for the
   proactive behavior system to consume.

The recent visual context is injected into ALL user queries (not just vision),
giving NOVA persistent awareness of what's happening around the user.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from server.app.behaviors.events import EventBus

logger = logging.getLogger(__name__)


@dataclass
class VisionFrame:
    """A single timestamped visual observation."""
    timestamp: float
    description: str
    changed: bool = True


@dataclass
class FrameMetadata:
    """Lightweight metadata from fast-tier analysis."""
    timestamp: float
    has_motion: bool = False
    motion_magnitude: float = 0.0
    brightness: float = 0.0
    person_present: bool = False
    scene_stable_seconds: float = 0.0


class VisionContext:
    """Continuous visual awareness engine with decoupled capture, analyze & describe."""

    def __init__(
        self,
        camera_manager,
        vision_llm,
        capture_interval: float = 0.1,
        change_check_interval: float = 0.5,
        describe_interval: float = 2.0,
        buffer_seconds: float = 60.0,
        max_buffer_size: int = 30,
        motion_threshold: int = 3000,
        skip_static: bool = True,
        event_bus: Optional["EventBus"] = None,
    ):
        self._camera = camera_manager
        self._llm = vision_llm
        self._capture_interval = capture_interval
        self._change_check_interval = change_check_interval
        self._describe_interval = describe_interval
        self._buffer_seconds = buffer_seconds
        self._max_buffer_size = max_buffer_size
        self._motion_threshold = motion_threshold
        self._skip_static = skip_static
        self._event_bus = event_bus

        # Description buffer (rich tier results)
        self._buffer: deque[VisionFrame] = deque(maxlen=max_buffer_size)
        self._last_description = ""

        # Frame metadata buffer (fast tier results)
        self._metadata_buffer: deque[FrameMetadata] = deque(maxlen=100)

        # Latest frame ready for description
        self._pending_frame: Optional[bytes] = None
        self._pending_has_change: bool = False
        self._pending_lock = asyncio.Lock()

        # Track describe state
        self._describing = False
        self._last_describe_time: float = 0.0
        self._last_significant_change: float = time.time()

        # Scene analyzer (fast-tier OpenCV)
        self._analyzer = None  # Lazy-loaded to avoid import at module level

        # Person presence tracking for events
        self._person_was_present: bool = False
        self._person_absent_since: float = 0.0
        self._person_static_since: float = time.time()
        self._last_motion_time: float = time.time()

        # Frame counter for decimated analysis
        self._frame_counter: int = 0
        self._frames_per_analysis = max(1, int(change_check_interval / capture_interval))

        # Tasks
        self._capture_task: Optional[asyncio.Task] = None
        self._describe_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start both background loops."""
        if self._running:
            return
        self._running = True
        self._capture_task = asyncio.create_task(self._capture_loop())
        self._describe_task = asyncio.create_task(self._describe_loop())
        logger.info(
            f"Vision context started: capture every {self._capture_interval}s, "
            f"analyze every {self._change_check_interval}s, "
            f"describe every {self._describe_interval}s, "
            f"buffer {self._buffer_seconds}s"
        )

    def stop(self) -> None:
        """Stop both background loops."""
        self._running = False
        for task in (self._capture_task, self._describe_task):
            if task and not task.done():
                task.cancel()
        logger.info("Vision context stopped")

    def get_recent_context(self, seconds: float = 30.0, max_entries: int = 10) -> str:
        """Get recent visual observations as formatted text for prompt injection.

        Returns both structured scene state and natural language descriptions.
        """
        if not self._buffer and not self._metadata_buffer:
            return ""

        now = time.time()
        parts = []

        # Add current scene state from fast tier
        if self._metadata_buffer:
            latest = self._metadata_buffer[-1]
            state_parts = []
            if latest.person_present:
                state_parts.append("person in room")
            else:
                state_parts.append("room appears empty")
            if latest.has_motion:
                state_parts.append("movement detected")
            elif latest.scene_stable_seconds > 10:
                state_parts.append(f"scene unchanged for {latest.scene_stable_seconds:.0f}s")
            brightness = "dark" if latest.brightness < 50 else "dim" if latest.brightness < 100 else "normal" if latest.brightness < 180 else "bright"
            state_parts.append(f"lighting: {brightness}")
            parts.append(f"[Scene: {' | '.join(state_parts)}]")

        # Add recent descriptions from rich tier
        cutoff = now - seconds
        recent = [f for f in self._buffer if f.timestamp >= cutoff]

        if not recent and self._buffer:
            recent = [self._buffer[-1]]

        if recent:
            for frame in list(recent)[-max_entries:]:
                ago = int(now - frame.timestamp)
                parts.append(f"- {ago}s ago: {frame.description}")

        if not parts:
            return ""

        return "[Visual Context - what the camera sees]\n" + "\n".join(parts)

    def get_last_description(self) -> str:
        return self._last_description

    def get_current_scene_state(self) -> Optional[FrameMetadata]:
        """Get the latest fast-tier scene state."""
        return self._metadata_buffer[-1] if self._metadata_buffer else None

    # ------------------------------------------------------------------
    # Fast capture loop — persistent camera, OpenCV analysis decimated
    # ------------------------------------------------------------------

    async def _capture_loop(self) -> None:
        """Capture frames at 10 FPS, run OpenCV analysis every Nth frame."""
        await asyncio.sleep(1)

        # Open persistent camera for fast reads
        try:
            await self._camera.open_persistent()
        except Exception as e:
            logger.error(f"Failed to open persistent camera: {e}")
            logger.info("Falling back to per-capture mode")

        # Lazy-load scene analyzer
        from server.app.vision.scene_analyzer import SceneAnalyzer
        self._analyzer = SceneAnalyzer(motion_threshold=self._motion_threshold)

        while self._running:
            try:
                await self._capture_tick()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Vision capture tick error: {e}")

            await asyncio.sleep(self._capture_interval)

        # Clean up persistent camera
        try:
            await self._camera.close_persistent()
        except Exception:
            pass

    async def _capture_tick(self) -> None:
        """Single capture: grab frame, optionally run analysis, queue for describe."""
        try:
            jpeg_bytes, _ = await self._camera.capture_fast()
        except Exception as e:
            logger.debug(f"Vision capture failed: {e}")
            return

        self._frame_counter += 1

        # Run OpenCV analysis every Nth frame (decimation)
        if self._frame_counter % self._frames_per_analysis == 0:
            scene_state = await asyncio.to_thread(self._analyzer.analyze, jpeg_bytes)

            metadata = FrameMetadata(
                timestamp=scene_state.timestamp,
                has_motion=scene_state.motion_detected,
                motion_magnitude=scene_state.motion_magnitude,
                brightness=scene_state.brightness,
                person_present=scene_state.person_present,
                scene_stable_seconds=scene_state.scene_stable_seconds,
            )
            self._metadata_buffer.append(metadata)

            # Publish events to EventBus
            if self._event_bus:
                await self._publish_events(metadata)

            # Queue frame for describe loop if significant change
            has_change = scene_state.is_significant_change
            if has_change:
                self._last_significant_change = time.time()

            async with self._pending_lock:
                if has_change or self._pending_frame is None:
                    self._pending_frame = jpeg_bytes
                    self._pending_has_change = has_change

        else:
            # Between analysis frames, just store the latest for potential describe
            async with self._pending_lock:
                if self._pending_frame is None:
                    self._pending_frame = jpeg_bytes
                    self._pending_has_change = False

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    async def _publish_events(self, metadata: FrameMetadata) -> None:
        """Publish vision events to the EventBus."""
        if not self._event_bus:
            return

        from server.app.behaviors.events import Event

        now = metadata.timestamp

        # Motion event
        if metadata.has_motion:
            self._last_motion_time = now
            self._event_bus.publish(Event(
                type="vision.motion_detected",
                timestamp=now,
                data={"magnitude": metadata.motion_magnitude},
                source="vision",
            ))

        # Person appeared (was absent, now present)
        if metadata.person_present and not self._person_was_present:
            absence_duration = now - self._person_absent_since if self._person_absent_since > 0 else 0
            self._event_bus.publish(Event(
                type="vision.person_appeared",
                timestamp=now,
                data={"absence_duration": absence_duration},
                source="vision",
            ))
            self._person_static_since = now  # Reset static timer

        # Person left (was present, now absent)
        if not metadata.person_present and self._person_was_present:
            self._person_absent_since = now
            self._event_bus.publish(Event(
                type="vision.person_left",
                timestamp=now,
                data={},
                source="vision",
            ))

        # Person static (present but no motion for a while)
        if metadata.person_present and not metadata.has_motion:
            static_duration = now - self._last_motion_time
            if static_duration > 60:  # Only report after 1 minute of stillness
                self._event_bus.publish(Event(
                    type="vision.person_static",
                    timestamp=now,
                    data={"duration": static_duration},
                    source="vision",
                ))

        # Scene changed (significant visual difference)
        if metadata.has_motion and metadata.motion_magnitude > 10000:
            self._event_bus.publish(Event(
                type="vision.scene_changed",
                timestamp=now,
                data={"magnitude": metadata.motion_magnitude, "brightness": metadata.brightness},
                source="vision",
            ))

        self._person_was_present = metadata.person_present

    # ------------------------------------------------------------------
    # Describe loop — LLM descriptions, only on change
    # ------------------------------------------------------------------

    async def _describe_loop(self) -> None:
        """Describe the latest captured frame. Skips static scenes."""
        await asyncio.sleep(3)  # Let capture loop + analyzer warm up

        while self._running:
            try:
                await self._describe_tick()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Vision describe tick error: {e}")

            await asyncio.sleep(self._describe_interval)

    async def _describe_tick(self) -> None:
        """Pick up the latest pending frame and describe it."""
        if self._llm is None:
            return

        async with self._pending_lock:
            frame_bytes = self._pending_frame
            has_change = self._pending_has_change
            self._pending_frame = None
            self._pending_has_change = False

        if frame_bytes is None:
            return

        now = time.time()

        # Skip describe if scene is static and we have a recent description
        if self._skip_static and not has_change:
            time_since_last = now - self._last_describe_time
            if time_since_last < 10.0 and self._last_description:
                return  # Scene hasn't changed, reuse existing description

        description = await self._describe_frame(frame_bytes)
        if not description:
            return

        self._last_describe_time = now
        self._buffer.append(VisionFrame(
            timestamp=now,
            description=description,
            changed=has_change,
        ))
        self._last_description = description

        # Prune old entries
        cutoff = now - self._buffer_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

        if has_change:
            logger.debug(f"Vision context (change): {description[:80]}...")
        else:
            logger.debug(f"Vision context (periodic): {description[:80]}...")

    # ------------------------------------------------------------------
    # Vision LLM description
    # ------------------------------------------------------------------

    async def _describe_frame(self, jpeg_bytes: bytes) -> str:
        """Send frame to vision LLM for a quick description."""
        import base64
        from server.app.llm.base import LLMMessage

        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

        messages = [
            LLMMessage(role="user", content=[
                {
                    "type": "text",
                    "text": (
                        "Describe what you see in this image in ONE concise sentence. "
                        "Focus on: people (what they're doing, wearing, holding, gestures), "
                        "objects on desk/in view, text on screens/papers/clothing, "
                        "and any hand gestures or signs. Be specific and factual."
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]),
        ]

        try:
            response = await self._llm.chat(messages, temperature=0.2, max_tokens=150)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Vision context describe error: {e}")
            return ""
