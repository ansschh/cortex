"""ElevenLabs Realtime STT via WebSocket — streams audio, receives transcripts.

Uses the Scribe v1/v2 realtime endpoint:
  wss://api.elevenlabs.io/v1/speech-to-text/realtime

Protocol reference:
  https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator, Callable, Optional

import numpy as np
import websockets

logger = logging.getLogger(__name__)

ELEVENLABS_STT_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


class ElevenLabsSTT:
    """Streams audio to ElevenLabs Realtime STT WebSocket, yields transcripts."""

    def __init__(
        self,
        api_key: str,
        model: str = "scribe_v1",
        language: str = "en",
        sample_rate: int = 16000,
    ):
        self._api_key = api_key
        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._first_chunk = True

        # Callbacks
        self.on_partial: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None

    async def connect(self) -> None:
        """Open WebSocket connection to ElevenLabs STT."""
        params = f"?language_code={self._language}"
        url = f"{ELEVENLABS_STT_WS_URL}{params}"
        extra_headers = {"xi-api-key": self._api_key}

        self._ws = await websockets.connect(url, additional_headers=extra_headers)
        self._running = True
        self._first_chunk = True
        logger.info("ElevenLabs STT WebSocket connected.")

    async def send_audio(self, audio_chunk: np.ndarray) -> None:
        """Send an audio chunk to the STT service."""
        if not self._ws or not self._running:
            return

        # Ensure int16
        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk * 32767).astype(np.int16)

        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()

        audio_bytes = audio_chunk.tobytes()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        msg: dict = {
            "message_type": "input_audio_chunk",
            "audio_base_64": b64_audio,
            "commit": False,
            "sample_rate": self._sample_rate,
        }
        try:
            await self._ws.send(json.dumps(msg))
            self._first_chunk = False
        except Exception as e:
            logger.error(f"STT send error: {e}")
            self._running = False

    async def end_of_speech(self) -> None:
        """Signal end of speech — send an empty commit chunk to flush VAD."""
        if not self._ws or not self._running:
            return
        try:
            msg = {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": self._sample_rate,
            }
            await self._ws.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"STT commit error: {e}")

    async def recv_json(self) -> dict:
        """Receive and parse a single JSON message from the WebSocket."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        raw = await self._ws.recv()
        return json.loads(raw)

    async def receive_transcripts(self) -> AsyncIterator[dict]:
        """Listen for transcript events from the WebSocket."""
        if not self._ws:
            logger.warning("receive_transcripts called but no WebSocket")
            return

        logger.info("Starting to receive transcripts...")
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("message_type", "")

                if msg_type == "session_started":
                    logger.info(f"STT session started: {data.get('session_id', '')} config={data.get('config', {})}")

                elif msg_type == "partial_transcript":
                    text = data.get("text", "")
                    logger.info(f"Partial transcript: '{text}'")
                    if text:
                        yield {
                            "text": text,
                            "is_final": False,
                            "confidence": 0.0,
                        }

                elif msg_type == "committed_transcript":
                    text = data.get("text", "")
                    logger.info(f"Committed transcript: '{text}'")
                    if text:
                        yield {
                            "text": text,
                            "is_final": True,
                            "confidence": 1.0,
                        }

                elif msg_type == "committed_transcript_with_timestamps":
                    text = data.get("text", "")
                    if text:
                        yield {
                            "text": text,
                            "is_final": True,
                            "confidence": 1.0,
                            "language": data.get("language_code", ""),
                            "words": data.get("words", []),
                        }

                elif msg_type in ("error", "auth_error", "quota_exceeded",
                                  "rate_limited", "input_error",
                                  "transcriber_error", "resource_exhausted",
                                  "session_time_limit_exceeded"):
                    logger.error(f"STT error ({msg_type}): {data.get('error', '')}")
                    if msg_type in ("auth_error", "quota_exceeded"):
                        break

        except websockets.ConnectionClosed:
            logger.info("STT WebSocket connection closed.")
        except Exception as e:
            logger.error(f"STT receive error: {e}")
        finally:
            self._running = False

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None
