"""AssemblyAI Universal-Streaming v3 STT — persistent WebSocket, callback-driven.

Keeps the WebSocket open for the entire conversation session. Audio streams
continuously during HOT_LISTENING, pauses during SPEAKING (connection stays open),
and only disconnects on DEEP_IDLE or shutdown.

v3 protocol: raw binary PCM audio, Turn messages with end_of_turn flag.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional

import numpy as np
import websockets

logger = logging.getLogger(__name__)

ASSEMBLYAI_WS_URL = "wss://streaming.assemblyai.com/v3/ws"
DEFAULT_SAMPLE_RATE = 16000


class AssemblyAIStreamingSTT:
    """Persistent AssemblyAI v3 WebSocket STT with continuous streaming."""

    def __init__(self, api_key: str, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._paused = False

        # Transcript state
        self._current_partial: str = ""

        # Callbacks — set by the client before/after connect
        self.on_partial: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_final: Optional[Callable[[str], Awaitable[None]]] = None

        # Audio send buffer — v3 requires 50-1000ms per message
        # At 16kHz, 50ms = 800 samples = 1600 bytes
        self._send_buffer = bytearray()
        self._min_send_bytes = int(self._sample_rate * 0.06 * 2)  # 60ms worth of int16

        # Background tasks
        self._recv_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._last_audio_sent: float = 0.0
        self._connect_time: float = 0.0

    async def connect(self) -> None:
        """Connect to AssemblyAI v3. Call once per conversation session."""
        if self._connected:
            return

        url = f"{ASSEMBLYAI_WS_URL}?sample_rate={self._sample_rate}&encoding=pcm_s16le"

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers={"Authorization": self._api_key},
                ping_interval=20,
                ping_timeout=10,
            )
        except Exception as e:
            logger.error(f"AssemblyAI connection failed: {e}")
            raise

        # Wait for Begin message
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            msg = json.loads(raw)
            if msg.get("type") != "Begin":
                raise RuntimeError(f"Unexpected first message: {msg}")
            session_id = msg.get("id", "?")
            expires = msg.get("expires_at", "?")
            logger.info(f"AssemblyAI v3 connected (session={session_id}, expires={expires})")
        except asyncio.TimeoutError:
            raise RuntimeError("AssemblyAI session start timed out")

        self._connected = True
        self._paused = False
        self._current_partial = ""
        self._connect_time = time.monotonic()

        # Start background tasks
        self._recv_task = asyncio.create_task(self._receive_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def send_audio(self, audio_chunk: np.ndarray) -> None:
        """Send raw PCM audio to AssemblyAI. No-op if paused or disconnected."""
        if not self._connected or not self._ws or self._paused:
            return

        try:
            # Ensure int16
            if audio_chunk.dtype != np.int16:
                audio_chunk = (audio_chunk * 32767).astype(np.int16)
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten()

            # Buffer until we have >= 60ms of audio (v3 requires 50-1000ms per message)
            self._send_buffer.extend(audio_chunk.tobytes())
            if len(self._send_buffer) < self._min_send_bytes:
                return

            raw = bytes(self._send_buffer)
            self._send_buffer.clear()
            await self._ws.send(raw)
            self._last_audio_sent = time.monotonic()
        except websockets.ConnectionClosed:
            logger.warning("AssemblyAI connection closed during send")
            self._connected = False
        except Exception as e:
            logger.error(f"AssemblyAI send error: {e}")

    def pause(self) -> None:
        """Stop sending audio (during SPEAKING). Connection stays open."""
        self._paused = True
        logger.debug("STT paused (speaking)")

    def resume(self) -> None:
        """Resume sending audio (after SPEAKING ends)."""
        self._paused = False
        self._current_partial = ""
        self._send_buffer.clear()
        logger.debug("STT resumed (listening)")

    async def set_endpointing_threshold(self, ms: int) -> None:
        """Dynamically adjust end-of-turn confidence threshold.

        v3 uses confidence (0-1) not milliseconds. We map:
          500ms  → 0.3  (trigger quickly)
          600ms  → 0.4
          1000ms → 0.5  (default)
        """
        if not self._connected or not self._ws:
            return
        # Map ms to confidence: lower ms = lower confidence threshold = faster triggering
        confidence = max(0.1, min(0.9, ms / 2000.0))
        try:
            await self._ws.send(json.dumps({
                "type": "UpdateConfiguration",
                "end_of_turn_confidence_threshold": confidence,
            }))
            logger.info(f"Endpointing threshold set: {ms}ms → confidence={confidence:.2f}")
        except Exception as e:
            logger.error(f"Failed to set endpointing threshold: {e}")

    async def force_endpoint(self) -> None:
        """Force AssemblyAI to emit an end-of-turn now."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"type": "ForceEndpoint"}))
        except Exception as e:
            logger.error(f"Failed to force endpoint: {e}")

    async def _receive_loop(self) -> None:
        """Background task: receive transcripts from AssemblyAI v3."""
        try:
            async for raw in self._ws:
                # v3 sends JSON text messages
                if isinstance(raw, bytes):
                    continue  # skip any binary messages

                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "Turn":
                    transcript = msg.get("transcript", "")
                    end_of_turn = msg.get("end_of_turn", False)

                    if end_of_turn:
                        # Final transcript for this turn
                        if transcript.strip():
                            self._current_partial = ""
                            if self.on_final:
                                await self.on_final(transcript)
                    else:
                        # Partial / in-progress transcript
                        if transcript:
                            self._current_partial = transcript
                            if self.on_partial:
                                await self.on_partial(transcript)

                elif msg_type == "Termination":
                    duration = msg.get("audio_duration_seconds", 0)
                    logger.info(f"AssemblyAI session terminated (audio={duration:.1f}s)")
                    self._connected = False
                    break

                elif msg_type == "Error":
                    error = msg.get("error", "Unknown error")
                    logger.error(f"AssemblyAI error: {error}")

        except websockets.ConnectionClosed as e:
            logger.warning(f"AssemblyAI connection closed: code={e.code}, reason='{e.reason}'")
            self._connected = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"AssemblyAI receive error: {e}")
            self._connected = False

    async def _keepalive_loop(self) -> None:
        """Send silent audio every 10s during pause to prevent timeout."""
        silence_bytes = np.zeros(1600, dtype=np.int16).tobytes()
        try:
            while self._connected:
                await asyncio.sleep(10)
                if self._paused and self._connected and self._ws:
                    elapsed = time.monotonic() - self._last_audio_sent
                    if elapsed > 8:
                        try:
                            await self._ws.send(silence_bytes)
                            self._last_audio_sent = time.monotonic()
                        except Exception:
                            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Keepalive error: {e}")

    @property
    def current_partial(self) -> str:
        return self._current_partial

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def close(self) -> None:
        """Disconnect. Call on DEEP_IDLE or shutdown."""
        self._connected = False
        self._paused = False

        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()

        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "Terminate"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info("AssemblyAI STT disconnected")
