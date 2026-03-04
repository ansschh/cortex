"""ElevenLabs TTS — WebSocket streaming for low-latency sentence-by-sentence synthesis."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator, Optional

import websockets

logger = logging.getLogger(__name__)

ELEVENLABS_WS_TTS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"


class ElevenLabsTTS:
    """WebSocket-based TTS for low-latency streaming. Supports sending text
    incrementally (sentence-by-sentence) and receiving audio chunks in real-time."""

    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_turbo_v2_5",
        output_format: str = "pcm_24000",
        speed: float = 1.15,
    ):
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._speed = speed
        self._cancel_event = asyncio.Event()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False

    async def connect(self) -> None:
        """Open a WebSocket TTS connection for a new response session."""
        self._cancel_event.clear()
        url = ELEVENLABS_WS_TTS_URL.format(voice_id=self._voice_id)
        url += f"?model_id={self._model_id}&output_format={self._output_format}"

        try:
            self._ws = await websockets.connect(url, close_timeout=5)
            # Send BOS (beginning of stream) with auth and voice settings
            bos = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.4,
                    "similarity_boost": 0.8,
                    "style": 0.15,
                    "use_speaker_boost": True,
                    "speed": self._speed,
                },
                "xi_api_key": self._api_key,
            }
            await self._ws.send(json.dumps(bos))
            self._connected = True
            logger.info("TTS WebSocket connected.")
        except Exception as e:
            logger.error(f"TTS WebSocket connection failed: {e}")
            self._connected = False

    async def send_text(self, text: str) -> None:
        """Send a text chunk (sentence) for synthesis. Audio will be received via recv_audio()."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"text": text + " "}))
        except Exception as e:
            logger.error(f"TTS send_text error: {e}")

    async def flush(self) -> None:
        """Send EOS (end of stream) to signal no more text is coming."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"text": ""}))
        except Exception as e:
            logger.error(f"TTS flush error: {e}")

    async def recv_audio(self) -> AsyncIterator[bytes]:
        """Receive audio chunks from the WebSocket. Yields raw PCM bytes."""
        if not self._connected or not self._ws:
            return
        try:
            async for raw_msg in self._ws:
                if self._cancel_event.is_set():
                    logger.info("TTS cancelled (barge-in).")
                    return
                msg = json.loads(raw_msg)
                # Log errors from ElevenLabs
                if "error" in msg or "message" in msg:
                    err = msg.get("error") or msg.get("message")
                    if err:
                        logger.error(f"ElevenLabs error: {err}")
                audio_b64 = msg.get("audio")
                if audio_b64:
                    yield base64.b64decode(audio_b64)
                if msg.get("isFinal"):
                    return
        except websockets.ConnectionClosed as e:
            logger.info(f"TTS WebSocket closed: code={e.code}, reason={e.reason}")
        except Exception as e:
            if not self._cancel_event.is_set():
                logger.error(f"TTS recv_audio error: {e}")

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def synthesize_chunks(self, text: str) -> AsyncIterator[bytes]:
        """One-shot: connect, send full text, yield audio chunks, close.
        Fallback for non-streaming use."""
        await self.connect()
        if not self._connected:
            return
        try:
            await self.send_text(text)
            await self.flush()
            async for chunk in self.recv_audio():
                if self._cancel_event.is_set():
                    return
                yield chunk
        finally:
            await self.close()

    def cancel(self) -> None:
        """Cancel ongoing TTS synthesis (barge-in).

        Sets the cancel flag AND closes the WebSocket to unblock
        recv_audio()'s `async for raw_msg in self._ws` loop, which
        would otherwise hang forever waiting for messages that will
        never come.
        """
        self._cancel_event.set()
        self._connected = False
        ws = self._ws
        if ws:
            try:
                asyncio.ensure_future(ws.close())
            except RuntimeError:
                pass  # No running event loop

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()
