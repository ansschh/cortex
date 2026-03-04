"""Client main — always-on voice loop with 3-state architecture.

States:
  DEEP_IDLE     → Whisper model loaded but idle, wake word / VAD / PTT only
  HOT_LISTENING → Audio buffered, Whisper runs partials every ~0.8s,
                  Smart Turn detects endpoint → force_endpoint → final transcript
  SPEAKING      → STT paused (model stays loaded), barge-in monitoring

No PROCESSING state — when Smart Turn triggers force_endpoint and Whisper
produces a final transcript, the client sends it to the server and stays in
HOT_LISTENING. The server's tts_start event asynchronously transitions to SPEAKING.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from client.audio.mic import MicStream
from client.audio.playback import AudioPlayer
from client.audio.stt_whisper_local import WhisperLocalSTT
from client.audio.tts_elevenlabs import ElevenLabsTTS
from client.audio.tts_local import LocalTTS
from client.audio.turn import EndOfTurnState, LocalSmartTurnAnalyzerV3
from client.audio.vad_silero import SileroVADDetector
from client.audio.wakeword import WakeWordDetector
from client.audio.speaker_verify import SpeakerVerifier

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BARGE_IN_GRACE_PERIOD = 0.1      # ignore mic for first 0.1s of playback
DEEP_IDLE_SILENCE_FIRST = 15     # seconds — first activation, no conversation yet
DEEP_IDLE_SILENCE_DEFAULT = 90   # seconds — after first turn, generous window
DEEP_IDLE_SILENCE_ACTIVE = 180   # seconds — active back-and-forth, 3 minutes


# ---------------------------------------------------------------------------
# Conversation engagement tracker — adaptive timeouts
# ---------------------------------------------------------------------------

@dataclass
class ConversationEngagement:
    """Tracks conversation activity to dynamically adjust timeouts."""

    turn_count: int = 0
    last_activity: float = 0.0
    last_assistant_asked_question: bool = False
    rapid_exchange_count: int = 0
    _last_turn_time: float = 0.0

    def record_turn(self) -> None:
        now = time.monotonic()
        if self._last_turn_time > 0 and (now - self._last_turn_time) < 5.0:
            self.rapid_exchange_count += 1
        else:
            self.rapid_exchange_count = max(0, self.rapid_exchange_count - 1)
        self._last_turn_time = now
        self.last_activity = now
        self.turn_count += 1

    def record_assistant_response(self, follow_up_expected: bool) -> None:
        self.last_assistant_asked_question = follow_up_expected
        self.last_activity = time.monotonic()

    @property
    def deep_idle_timeout(self) -> float:
        if self.turn_count == 0:
            return DEEP_IDLE_SILENCE_FIRST
        if self.rapid_exchange_count >= 2:
            return DEEP_IDLE_SILENCE_ACTIVE
        return DEEP_IDLE_SILENCE_DEFAULT

    @property
    def endpoint_silence_ms(self) -> int:
        """AssemblyAI endpointing threshold in ms.

        Higher values = user can pause longer mid-thought without triggering.
        Maps to confidence via: confidence = ms / 2000.0
          1200ms → 0.60 confidence (default — allows natural thinking pauses)
           800ms → 0.40 confidence (rapid exchange — quicker turn-taking)
           600ms → 0.30 confidence (follow-up expected — short reply likely)
        """
        if self.last_assistant_asked_question:
            return 600   # expecting a reply, but still allow brief pauses
        if self.rapid_exchange_count >= 2:
            return 800   # rapid back-and-forth
        return 1200      # default — generous, lets user think mid-sentence


# ---------------------------------------------------------------------------
# VAD state tracker — wraps SileroVADDetector to produce start/stop events
# ---------------------------------------------------------------------------

class VADStateTracker:
    """Tracks VAD start/stop transitions with timing.

    Wraps SileroVADDetector to produce discrete start/stop events with
    start_secs/stop_secs timing, matching pipecat's VADAnalyzer semantics.
    The underlying SileroVADDetector is NOT modified — it's still used
    directly for DEEP_IDLE wake-up and SPEAKING barge-in.
    """

    def __init__(self, vad: SileroVADDetector, start_secs: float = 0.2, stop_secs: float = 0.3):
        self._vad = vad
        self._start_secs = start_secs
        self._stop_secs = stop_secs
        self._is_speaking = False
        self._consecutive_speech_s = 0.0
        self._consecutive_silence_s = 0.0
        self._chunk_duration_s = 512 / 16000  # ~32ms per chunk

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def start_secs(self) -> float:
        """How many seconds of speech were needed to trigger the start event."""
        return self._start_secs

    @property
    def stop_secs(self) -> float:
        """How many seconds of silence were needed to trigger the stop event."""
        return self._stop_secs

    def process_chunk(self, audio_int16: np.ndarray) -> tuple[bool | None, float]:
        """Process an audio chunk and return (event, probability).

        Returns:
            event: True = just started speaking, False = just stopped speaking,
                   None = no transition
            prob: VAD probability for this chunk
        """
        prob = self._vad.process_chunk(audio_int16)
        is_speech = prob >= 0.5

        event = None

        if is_speech:
            self._consecutive_silence_s = 0.0
            self._consecutive_speech_s += self._chunk_duration_s
            if not self._is_speaking and self._consecutive_speech_s >= self._start_secs:
                self._is_speaking = True
                event = True  # VAD start
        else:
            self._consecutive_speech_s = 0.0
            if self._is_speaking:
                self._consecutive_silence_s += self._chunk_duration_s
                if self._consecutive_silence_s >= self._stop_secs:
                    self._is_speaking = False
                    event = False  # VAD stop

        return event, prob

    def reset(self):
        self._is_speaking = False
        self._consecutive_speech_s = 0.0
        self._consecutive_silence_s = 0.0


# ---------------------------------------------------------------------------
# Client states
# ---------------------------------------------------------------------------

class ClientState(str, Enum):
    DEEP_IDLE = "deep_idle"
    HOT_LISTENING = "hot_listening"
    SPEAKING = "speaking"


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class AssistantClient:
    """Always-on voice client — 3-state architecture."""

    def __init__(self):
        # Config from env
        self._server_url = os.getenv("SERVER_WS_URL", "ws://localhost:8000/ws/client")
        self._eleven_key = os.getenv("ELEVEN_API_KEY", "")
        self._eleven_voice = os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self._verify_threshold = float(os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.65"))

        # Components
        self.mic = MicStream()
        self.wakeword = WakeWordDetector()
        self.stt = WhisperLocalSTT(
            model_size=os.getenv("WHISPER_MODEL", "large-v3"),
            device=os.getenv("WHISPER_DEVICE", "cuda"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
            vocab_prompt=os.getenv(
                "WHISPER_VOCAB_PROMPT",
                "Jarvis, Nova, Hey Jarvis, calorie, calories, dorm room, entrepreneurship",
            ),
        )
        self.tts = ElevenLabsTTS(
            api_key=self._eleven_key,
            voice_id=self._eleven_voice,
        )
        # Local TTS for FastPath responses (Piper VITS, ~100ms on CPU)
        self.local_tts: Optional[LocalTTS] = None
        try:
            piper_model = os.getenv("PIPER_MODEL_PATH", "data/models/piper/en_US-amy-medium.onnx")
            self.local_tts = LocalTTS(model_path=piper_model)
        except Exception as e:
            logger.warning(f"Local TTS (Piper) not available: {e}")
        self.player = AudioPlayer(sample_rate=24000)
        self.speaker = SpeakerVerifier(threshold=self._verify_threshold)
        self.vad = SileroVADDetector()

        # Pipecat SmartTurn v3 — ML-based turn detection (primary gate)
        self.turn_analyzer = LocalSmartTurnAnalyzerV3(sample_rate=16000)
        self.turn_analyzer.set_sample_rate(16000)
        # VAD state tracker created after vad.initialize() in start()
        self.vad_tracker: Optional[VADStateTracker] = None

        # Engagement tracker
        self.engagement = ConversationEngagement()

        # State
        self._state = ClientState.DEEP_IDLE
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._mic_muted = False  # UI-driven mute — blocks all audio processing

        # Audio buffer for speaker verification
        self._utterance_buffer: list[np.ndarray] = []

        # Push-to-talk fallback
        self._ptt_active = False

        # Conversation context
        self._last_assistant_text = ""
        self._follow_up_expected = False

        # Silence tracking for deep idle transition
        self._last_speech_time = 0.0

        # Pre-barge-in audio buffer
        self._prebuffer_chunks: list[np.ndarray] = []

        # Post-speaking grace period — suppress SmartTurn for 0.5s after playback
        self._hot_listening_start: float = 0.0

        # Streaming TTS state
        self._speak_stream: Optional[sd.OutputStream] = None
        self._speak_barge_in_count = 0
        self._speak_start = 0.0
        self._speak_barged = False
        self._tts_audio_task: Optional[asyncio.Task] = None
        self._barge_in_task: Optional[asyncio.Task] = None
        self._tts_preconnected = False
        self._tts_connect_lock = asyncio.Lock()

        # Pipecat-style turn detection state
        self._turn_text: str = ""
        self._turn_complete: bool = False
        self._turn_vad_speaking: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all components and start the main loop."""
        logger.info("Initializing assistant client...")

        self.mic.start()
        self.wakeword.initialize()
        self.speaker.initialize()
        self.vad.initialize()
        self.vad_tracker = VADStateTracker(self.vad, start_secs=0.2, stop_secs=0.3)

        # Pre-load Whisper model onto GPU so first activation is instant
        await self.stt.connect()
        self.stt.on_partial = self._on_stt_partial
        self.stt.on_final = self._on_stt_final

        await self._connect_server()

        self._running = True
        logger.info("Client started. Say something or use wake word...")

        await asyncio.gather(
            self._audio_loop(),
            self._server_listener(),
            self._keyboard_listener(),
        )

    async def _connect_server(self) -> None:
        """Establish WebSocket connection to the server."""
        retry_count = 0
        while retry_count < 10:
            try:
                self._ws = await websockets.connect(self._server_url)
                logger.info(f"Connected to server: {self._server_url}")
                return
            except Exception as e:
                retry_count += 1
                logger.warning(f"Server connection failed (attempt {retry_count}): {e}")
                await asyncio.sleep(2)
        logger.error("Could not connect to server. Running in offline mode.")

    async def _send_event(self, event: dict) -> None:
        """Send an event to the server."""
        if self._ws:
            try:
                await self._ws.send(json.dumps(event))
            except Exception as e:
                logger.error(f"Send error: {e}")
                await self._connect_server()

    # ------------------------------------------------------------------
    # Main audio loop
    # ------------------------------------------------------------------

    async def _audio_loop(self) -> None:
        """Main audio processing loop — dispatches by state."""
        while self._running:
            try:
                if self._state == ClientState.DEEP_IDLE:
                    await self._deep_idle_tick()
                elif self._state == ClientState.HOT_LISTENING:
                    await self._hot_listening_tick()
                elif self._state == ClientState.SPEAKING:
                    await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"Audio loop error: {e}")
                self._state = ClientState.DEEP_IDLE
                await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # DEEP_IDLE: wake word, VAD onset, or PTT
    # ------------------------------------------------------------------

    async def _deep_idle_tick(self) -> None:
        """Listen for wake word, push-to-talk, or speech onset."""
        if self._mic_muted:
            await asyncio.sleep(0.1)
            return

        loop = asyncio.get_event_loop()
        chunk = await loop.run_in_executor(None, self.mic.get_chunk, 0.05)
        if chunk is None:
            return

        # Check wake word
        if self.wakeword.is_available:
            detection = self.wakeword.process_chunk(chunk)
            if detection:
                await self._enter_hot_listening(detection)
                return

        # Check push-to-talk
        if self._ptt_active:
            self._ptt_active = False
            await self._enter_hot_listening({"model_name": "push_to_talk", "confidence": 1.0})
            return

        # VAD speech onset
        if self.vad.is_available:
            if self.vad.detect_speech_onset(chunk):
                prob = self.vad.last_probability
                logger.info(f"VAD speech onset detected (prob={prob:.2f})")
                await self._enter_hot_listening(
                    {"model_name": "vad_auto", "confidence": prob},
                )
                return
        else:
            # Fallback: energy threshold
            energy = np.abs(chunk).mean()
            if energy > 500:
                logger.info(f"Energy speech onset detected (energy={energy:.0f})")
                await self._enter_hot_listening(
                    {"model_name": "vad_auto", "confidence": min(energy / 5000, 1.0)},
                )

    # ------------------------------------------------------------------
    # Transition: → HOT_LISTENING
    # ------------------------------------------------------------------

    async def _enter_hot_listening(
        self,
        detection: dict | None = None,
        prebuffer: list[np.ndarray] | None = None,
    ) -> None:
        """Transition to HOT_LISTENING — connect or resume STT."""
        logger.info(f"Entering HOT_LISTENING (from={self._state.value}, trigger={detection})")

        self._utterance_buffer.clear()
        self._prebuffer_chunks = prebuffer or []
        self._last_speech_time = time.monotonic()
        self.vad.reset()

        # Reset pipecat-style turn detection state
        self._turn_text = ""
        self._turn_complete = False
        self._turn_vad_speaking = False
        self.turn_analyzer.clear()
        if self.vad_tracker:
            self.vad_tracker.reset()

        if not self.stt.is_connected:
            # Fresh connection (model not pre-loaded — fallback)
            try:
                await self.stt.connect()
                self.stt.on_partial = self._on_stt_partial
                self.stt.on_final = self._on_stt_final
            except Exception as e:
                logger.error(f"STT connection failed: {e}")
                self._state = ClientState.DEEP_IDLE
                return
        else:
            # Model already loaded — resume (clears buffer for fresh utterance)
            self.stt.resume()

        # Reset speech timer so model load time doesn't count toward silence timeout
        self._last_speech_time = time.monotonic()
        self._hot_listening_start = time.monotonic()
        self._state = ClientState.HOT_LISTENING

        # Notify server of activation (only on fresh activation, not resume)
        if detection:
            await self._send_event({
                "event": "wakeword_detected",
                "model_name": detection.get("model_name", ""),
                "confidence": detection.get("confidence", 0.0),
                "timestamp": time.time(),
            })

        # Replay pre-barge-in audio into STT
        if self._prebuffer_chunks:
            logger.info(f"Replaying {len(self._prebuffer_chunks)} pre-barge-in chunks into STT")
            for pchunk in self._prebuffer_chunks:
                self._utterance_buffer.append(pchunk.copy())
                await self.stt.send_audio(pchunk)
            self._prebuffer_chunks = []

    # ------------------------------------------------------------------
    # HOT_LISTENING: pipecat-style turn detection
    # ------------------------------------------------------------------

    async def _hot_listening_tick(self) -> None:
        """Stream mic audio to STT and turn analyzer. Pipecat-style turn detection.

        The ML model is the PRIMARY gate — if it says INCOMPLETE, the turn
        never ends regardless of pause length (up to the 3-second hard timeout
        in BaseSmartTurn.append_audio).
        """
        loop = asyncio.get_event_loop()
        chunk = await loop.run_in_executor(None, self.mic.get_chunk, 0.03)
        if chunk is None:
            return

        self._utterance_buffer.append(chunk.copy())
        await self.stt.send_audio(chunk)

        # --- VAD state tracking (produces start/stop events) ---
        vad_event, vad_prob = self.vad_tracker.process_chunk(chunk)

        if vad_event is True:
            await self._on_vad_start()
        elif vad_event is False:
            await self._on_vad_stop()

        # --- Feed audio to turn analyzer ---
        audio_bytes = chunk.tobytes()
        state = self.turn_analyzer.append_audio(audio_bytes, self._turn_vad_speaking)

        if state == EndOfTurnState.COMPLETE:
            # 3-second silence timeout hit inside append_audio
            logger.info("SmartTurn: silence timeout (3s) — forcing endpoint")
            await self.stt.force_endpoint()
            self.turn_analyzer.clear()

        # Track speech for deep idle timeout
        if vad_prob >= 0.5:
            self._last_speech_time = time.monotonic()

        # Check for deep idle timeout
        silence_duration = time.monotonic() - self._last_speech_time
        if silence_duration > self.engagement.deep_idle_timeout:
            logger.info(f"Silence timeout ({silence_duration:.0f}s) — entering DEEP_IDLE")
            await self._enter_deep_idle()

    # ------------------------------------------------------------------
    # VAD event handlers (pipecat algorithm)
    # ------------------------------------------------------------------

    async def _on_vad_start(self) -> None:
        """Handle VAD start event — reset turn state.

        When the user resumes speaking, any previous ML model decision
        is thrown away. This is how long pauses work: the model said
        INCOMPLETE, we waited, user spoke again, everything resets.
        """
        logger.debug("VAD: user started speaking")
        self._turn_complete = False
        self._turn_vad_speaking = True
        self._last_speech_time = time.monotonic()

        # Tell turn analyzer about VAD start timing
        self.turn_analyzer.update_vad_start_secs(self.vad_tracker.start_secs)

    async def _on_vad_stop(self) -> None:
        """Handle VAD stop event — run ML model, maybe force endpoint.

        This is where the magic happens. The ML model analyzes the last
        8 seconds of audio (including the trailing silence) and decides:
        is the user DONE or just PAUSING?
        """
        logger.debug("VAD: user stopped speaking")
        self._turn_vad_speaking = False

        # Grace period after returning from SPEAKING — suppress SmartTurn
        # for 0.5s to avoid false triggers on residual playback audio
        since_hot = time.monotonic() - self._hot_listening_start
        if since_hot < 0.5:
            logger.debug(f"SmartTurn: suppressed (post-speak grace, {since_hot:.2f}s)")
            return

        # Run ML model inference (in ThreadPoolExecutor, ~12ms)
        state, prediction = await self.turn_analyzer.analyze_end_of_turn()

        if prediction:
            logger.info(
                f"SmartTurn: complete={prediction.is_complete}, "
                f"prob={prediction.probability:.3f}, "
                f"time={prediction.e2e_processing_time_ms:.1f}ms"
            )

        if state == EndOfTurnState.COMPLETE:
            # ML model says user is DONE — force endpoint
            logger.info("SmartTurn: turn COMPLETE — forcing endpoint")
            await self.stt.force_endpoint()
            self.turn_analyzer.clear()
        else:
            # ML model says INCOMPLETE — user is just pausing, do nothing.
            # They can pause as long as they want (up to 3s silence timeout).
            logger.info("SmartTurn: turn INCOMPLETE — waiting for user to continue")

    # ------------------------------------------------------------------
    # STT callbacks (fired by Whisper partial loop / force_endpoint)
    # ------------------------------------------------------------------

    async def _on_stt_partial(self, text: str) -> None:
        """Handle partial transcript from Whisper."""
        if self._state != ClientState.HOT_LISTENING:
            return
        # Track text for turn detection (pipecat uses this to gate endpoint)
        self._turn_text = text
        await self._send_event({
            "event": "stt_partial",
            "text": text,
            "timestamp": time.time(),
        })

    async def _on_stt_final(self, text: str) -> None:
        """Handle final transcript from Whisper (triggered by force_endpoint)."""
        if self._state != ClientState.HOT_LISTENING or self._mic_muted:
            return

        text = text.strip()
        if not text:
            return

        logger.info(f"STT final: '{text}'")
        self._last_speech_time = time.monotonic()

        await self._dispatch_final_transcript(text)

    async def _dispatch_final_transcript(self, text: str) -> None:
        """Send final transcript to server. Stay in HOT_LISTENING."""
        # Reset turn text after dispatching
        self._turn_text = ""

        # Send transcript FIRST — don't block on speaker verification
        await self._send_event({
            "event": "stt_final",
            "text": text,
            "timestamp": time.time(),
        })

        # Update engagement
        self.engagement.record_turn()

        # Pre-connect TTS and run speaker verification in background (non-blocking)
        asyncio.create_task(self._preconnect_tts())
        asyncio.create_task(self._send_speaker_verification())

        # Stay in HOT_LISTENING — no state change
        logger.info(f"Transcript sent: '{text}' (staying in HOT_LISTENING)")

    async def _send_speaker_verification(self) -> None:
        """Send speaker verification in background — doesn't block dispatch."""
        try:
            verification = self._verify_speaker()
            await self._send_event({
                "event": "speaker_verified",
                **verification,
                "timestamp": time.time(),
            })
        except Exception as e:
            logger.error(f"Speaker verification error: {e}")

    async def _preconnect_tts(self) -> None:
        """Pre-connect TTS WebSocket while server is processing."""
        async with self._tts_connect_lock:
            if self._tts_preconnected or self.tts._connected:
                return
            if self._state == ClientState.SPEAKING:
                return
            try:
                await self.tts.connect()
                if self._state == ClientState.SPEAKING:
                    await self.tts.close()
                    return
                self._tts_preconnected = True
                logger.info("TTS pre-connected during server processing")
            except Exception as e:
                logger.warning(f"TTS pre-connect failed (will retry on tts_start): {e}")
                self._tts_preconnected = False

    def _verify_speaker(self) -> dict:
        """Run speaker verification on the collected utterance."""
        if not self.speaker.is_available or not self.speaker.is_enrolled:
            return {"is_verified": False, "confidence": 0.0, "speaker_label": "not_enrolled"}
        if not self._utterance_buffer:
            return {"is_verified": False, "confidence": 0.0, "speaker_label": "no_audio"}
        full_audio = np.concatenate(self._utterance_buffer)
        return self.speaker.verify(full_audio)

    # ------------------------------------------------------------------
    # Transition: → DEEP_IDLE
    # ------------------------------------------------------------------

    async def _enter_deep_idle(self) -> None:
        """Pause STT, reset engagement, go to DEEP_IDLE."""
        logger.info("Entering DEEP_IDLE")

        # Force-endpoint any remaining audio before going idle — but NOT
        # if mic was muted (user wants everything to stop, not trigger a
        # new server response from the partial transcript)
        if not self._mic_muted and self.stt.current_partial:
            await self.stt.force_endpoint()
        self.turn_analyzer.clear()

        # Pause STT but keep model loaded in VRAM for instant reactivation
        self.stt.pause()
        self.engagement = ConversationEngagement()
        self._follow_up_expected = False
        self._tts_preconnected = False
        self._state = ClientState.DEEP_IDLE

        await self._send_event({
            "event": "client_state",
            "state": "deep_idle",
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # Server listener
    # ------------------------------------------------------------------

    async def _server_listener(self) -> None:
        """Listen for events from the server and handle them."""
        while self._running:
            if not self._ws:
                await asyncio.sleep(1)
                continue

            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=0.1)
                data = json.loads(raw)
                event_type = data.get("event", "")

                if event_type == "assistant_tts_start":
                    await self._handle_tts_stream_start()
                elif event_type == "assistant_tts_chunk":
                    await self._handle_tts_stream_chunk(data.get("text", ""))
                elif event_type == "assistant_tts_end":
                    # Read follow-up flag from server
                    self._follow_up_expected = data.get("follow_up_expected", False)
                    self.engagement.record_assistant_response(self._follow_up_expected)
                    await self._handle_tts_stream_end()
                elif event_type == "assistant_tts_text":
                    await self._handle_tts_oneshot(
                        data.get("text", ""),
                        use_local_tts=data.get("use_local_tts", False),
                    )
                elif event_type == "assistant_text":
                    logger.info(f"Assistant: {data.get('text', '')[:100]}")
                elif event_type == "assistant_audio_control":
                    if data.get("action") == "stop":
                        self.player.cancel()
                elif event_type == "mic_mute":
                    logger.info("Mic muted via UI")
                    self._mic_muted = True
                    if self._state == ClientState.HOT_LISTENING:
                        await self._enter_deep_idle()
                    elif self._state == ClientState.SPEAKING:
                        self._speak_barged = True  # stop TTS
                        await self._enter_deep_idle()
                elif event_type == "mic_unmute":
                    logger.info("Mic unmuted via UI")
                    self._mic_muted = False
                    if self._state == ClientState.DEEP_IDLE:
                        await self._enter_hot_listening(
                            {"model_name": "ui_unmute", "confidence": 1.0},
                        )
                elif event_type == "stop_speaking":
                    logger.info("Stop speaking via UI")
                    if self._state == ClientState.SPEAKING:
                        self._speak_barged = True
                        self.tts.cancel()

            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                logger.warning("Server connection lost. Reconnecting...")
                await self._connect_server()
            except Exception as e:
                logger.error(f"Server listener error: {e}")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Streaming TTS
    # ------------------------------------------------------------------

    async def _handle_tts_stream_start(self) -> None:
        """Server signals start of a streaming response."""
        if self._mic_muted:
            logger.info("Ignoring tts_start — mic is muted")
            return
        self._state = ClientState.SPEAKING
        self.wakeword.disable()
        self.mic.mute()

        # Pause STT (keep connection open)
        self.stt.pause()

        # Open speaker output stream
        self._speak_stream = sd.OutputStream(
            samplerate=24000, channels=1, dtype="int16", blocksize=4096,
        )
        self._speak_stream.start()
        self._speak_barge_in_count = 0
        self._speak_start = time.monotonic()
        self._speak_barged = False

        # Use pre-connected TTS if available, otherwise connect now (locked)
        async with self._tts_connect_lock:
            if self._tts_preconnected and self.tts._connected:
                logger.info("TTS stream session started (pre-connected).")
            else:
                await self.tts.connect()
                logger.info("TTS stream session started (fresh connect).")
            self._tts_preconnected = False

        # Start background tasks
        self._tts_audio_task = asyncio.create_task(self._tts_audio_player())
        self._barge_in_task = asyncio.create_task(self._barge_in_monitor())

    async def _handle_tts_stream_chunk(self, text: str) -> None:
        """Server sends a sentence. Forward to TTS WebSocket."""
        if not text.strip() or self._speak_barged:
            return
        logger.info(f"Speaking chunk: {text[:60]}...")
        self._last_assistant_text = text
        await self.tts.send_text(text)

    async def _handle_tts_stream_end(self) -> None:
        """Server signals end of streaming response. Flush TTS and wait for audio."""
        if self._speak_barged:
            return
        await self.tts.flush()

        if self._tts_audio_task:
            try:
                await asyncio.wait_for(self._tts_audio_task, timeout=30)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        if self._barge_in_task and not self._barge_in_task.done():
            self._barge_in_task.cancel()

        if not self._speak_barged:
            await self._tts_cleanup()

    async def _tts_audio_player(self) -> None:
        """Background task: receive audio from TTS WS, play it via speaker."""
        loop = asyncio.get_event_loop()
        try:
            async for audio_bytes in self.tts.recv_audio():
                if self._speak_barged or self._state != ClientState.SPEAKING:
                    break
                stream = self._speak_stream
                if stream is None:
                    break
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                await loop.run_in_executor(
                    None, stream.write, audio_chunk.reshape(-1, 1)
                )

            if self._speak_stream and not self._speak_barged:
                await asyncio.sleep(0.15)  # brief drain for output buffer

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._speak_barged:
                logger.error(f"TTS audio player error: {e}")
        finally:
            # If barge-in happened, clean up stream HERE — guaranteed no
            # writes in flight because we only reach finally after the
            # last run_in_executor(stream.write) has returned.
            if self._speak_barged and self._speak_stream:
                try:
                    self._speak_stream.abort()
                    self._speak_stream.close()
                except Exception:
                    pass
                self._speak_stream = None

    async def _barge_in_monitor(self) -> None:
        """Monitor for barge-in during TTS playback."""
        check_count = 0
        try:
            while self._state == ClientState.SPEAKING and not self._speak_barged:
                await asyncio.sleep(0.015)
                check_count += 1

                elapsed = time.monotonic() - self._speak_start
                if elapsed < BARGE_IN_GRACE_PERIOD:
                    continue

                barge_detected = False
                if self.vad.is_available:
                    latest = self.mic.peek_latest_chunk()
                    if latest is not None:
                        barge_detected = self.vad.detect_barge_in(latest)
                        if check_count % 33 == 0:
                            logger.debug(f"Barge-in monitor: vad_prob={self.vad.last_probability:.2f}")
                else:
                    energy = self.mic.current_energy
                    if energy > 300:
                        self._speak_barge_in_count += 1
                        barge_detected = self._speak_barge_in_count >= 4
                    else:
                        self._speak_barge_in_count = 0

                if barge_detected:
                    logger.info("Barge-in detected!")
                    # 1. Signal audio player to stop + cancel TTS immediately
                    self._speak_barged = True
                    self.tts.cancel()

                    # 2. Grab pre-barge-in audio and unmute mic RIGHT AWAY
                    prebuffer = self.mic.get_ring_buffer()
                    logger.info(f"Prebuffer: {len(prebuffer)} chunks (~{len(prebuffer)*32:.0f}ms)")
                    self.mic.unmute()
                    self.mic.drain()
                    self.wakeword.enable()

                    # 3. Wait for audio player to exit. It will see _speak_barged,
                    # stop after its current stream.write(), and clean up the
                    # stream in its finally block. Use asyncio.wait (not wait_for)
                    # so the task is NOT cancelled on timeout — cancelling leaves
                    # run_in_executor(stream.write) orphaned in the thread pool,
                    # and closing the stream while it writes causes a segfault.
                    if self._tts_audio_task and not self._tts_audio_task.done():
                        done, _ = await asyncio.wait(
                            [self._tts_audio_task], timeout=2.0
                        )
                        if not done:
                            logger.warning("Audio task didn't finish in 2s, proceeding anyway")

                    # 4. Close TTS in background
                    asyncio.create_task(self.tts.close())

                    # 5. Resume STT and go to HOT_LISTENING
                    await self._enter_hot_listening(
                        detection={"model_name": "barge_in", "confidence": 1.0},
                        prebuffer=prebuffer,
                    )
                    return

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Barge-in monitor error: {e}")

    async def _tts_cleanup(self) -> None:
        """Clean up TTS and return to HOT_LISTENING (not DEEP_IDLE)."""
        if self._speak_barged:
            return
        await self.tts.close()
        if self._speak_stream:
            try:
                self._speak_stream.stop()
                self._speak_stream.close()
            except Exception:
                pass
            self._speak_stream = None

        self.mic.unmute()
        self.wakeword.enable()

        # Return to HOT_LISTENING — resume STT, stay conversational
        if self._state == ClientState.SPEAKING:
            await self._enter_hot_listening()
            logger.info("Finished speaking, returning to HOT_LISTENING.")

    # ------------------------------------------------------------------
    # Legacy one-shot TTS (for tool call responses)
    # ------------------------------------------------------------------

    async def _handle_tts_oneshot(self, text: str, use_local_tts: bool = False) -> None:
        """Single-shot TTS for non-streaming responses (tool calls etc).

        If use_local_tts=True and Piper is available, synthesizes locally (~100ms).
        Otherwise reuses pre-connected ElevenLabs WebSocket (saves ~200ms).
        """
        if not text.strip() or self._mic_muted:
            return
        self._state = ClientState.SPEAKING
        self._last_assistant_text = text
        self.wakeword.disable()
        self.mic.mute()
        self.stt.pause()

        stream_out = None
        try:
            # --- LOCAL TTS PATH (Piper, ~100ms) ---
            if use_local_tts and self.local_tts and self.local_tts.is_ready:
                t0 = time.perf_counter()
                audio_int16 = await self.local_tts.synthesize(text)
                synth_ms = (time.perf_counter() - t0) * 1000
                logger.info(f"Local TTS: {synth_ms:.0f}ms | {text[:80]}")

                if len(audio_int16) > 0:
                    sr = self.local_tts.sample_rate
                    stream_out = sd.OutputStream(
                        samplerate=sr, channels=1, dtype="int16", blocksize=4096,
                    )
                    stream_out.start()
                    # Write audio in chunks for barge-in responsiveness
                    chunk_size = sr // 4  # 250ms chunks
                    for i in range(0, len(audio_int16), chunk_size):
                        if self._state != ClientState.SPEAKING:
                            break
                        chunk = audio_int16[i:i + chunk_size]
                        stream_out.write(chunk.reshape(-1, 1))
                    if stream_out:
                        await asyncio.sleep(0.1)
            else:
                # --- CLOUD TTS PATH (ElevenLabs) ---
                logger.info(f"Speaking (oneshot): {text[:80]}...")
                stream_out = sd.OutputStream(
                    samplerate=24000, channels=1, dtype="int16", blocksize=4096,
                )
                stream_out.start()

                # Reuse pre-connected TTS WebSocket if available
                async with self._tts_connect_lock:
                    if self._tts_preconnected and self.tts._connected:
                        logger.info("TTS oneshot using pre-connected WebSocket")
                    else:
                        await self.tts.connect()
                        logger.info("TTS oneshot fresh connect")
                    self._tts_preconnected = False

                if self.tts._connected:
                    await self.tts.send_text(text)
                    await self.tts.flush()
                    async for chunk in self.tts.recv_audio():
                        if self._state != ClientState.SPEAKING:
                            break
                        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
                        stream_out.write(audio_chunk.reshape(-1, 1))

                if stream_out:
                    await asyncio.sleep(0.15)

        except Exception as e:
            logger.error(f"TTS oneshot error: {e}")
        finally:
            # Always close cloud TTS after oneshot
            if not (use_local_tts and self.local_tts and self.local_tts.is_ready):
                try:
                    await self.tts.close()
                except Exception:
                    pass
            if stream_out:
                try:
                    stream_out.stop()
                    stream_out.close()
                except Exception:
                    pass
            self.mic.unmute()
            self.wakeword.enable()
            # Return to HOT_LISTENING
            if self._state == ClientState.SPEAKING:
                await self._enter_hot_listening()
                logger.info("Finished speaking (oneshot), returning to HOT_LISTENING.")

    # ------------------------------------------------------------------
    # Keyboard listener
    # ------------------------------------------------------------------

    async def _keyboard_listener(self) -> None:
        """Listen for keyboard input as push-to-talk fallback and enrollment commands."""
        loop = asyncio.get_event_loop()
        print("\n[NOVA] Commands: t=talk, say <text>, enroll, status, q=quit")
        while self._running:
            try:
                line = await loop.run_in_executor(None, self._read_input)
                if not line:
                    continue

                cmd = line.strip().lower()
                if not cmd:
                    continue

                if cmd in ("q", "quit"):
                    logger.info("Quit command received.")
                    self._running = False
                    break

                elif cmd in ("t", "talk"):
                    if self._state == ClientState.DEEP_IDLE:
                        await self._enter_hot_listening(
                            {"model_name": "push_to_talk", "confidence": 1.0},
                        )
                    elif self._state == ClientState.HOT_LISTENING:
                        print("[NOVA] Already listening!")
                    else:
                        print(f"[NOVA] Can't talk now — state is {self._state.value}")

                elif cmd == "enroll":
                    await self._enroll_speaker()

                elif cmd.startswith("say "):
                    text = cmd[4:].strip()
                    if text:
                        await self._send_event({
                            "event": "stt_final",
                            "text": text,
                            "timestamp": time.time(),
                        })

                elif cmd == "status":
                    logger.info(
                        f"State: {self._state.value}, "
                        f"STT: {'connected' if self.stt.is_connected else 'disconnected'}, "
                        f"Mic: {'muted' if self.mic.is_muted else 'active'}, "
                        f"Engagement: turns={self.engagement.turn_count}, "
                        f"idle_timeout={self.engagement.deep_idle_timeout:.0f}s, "
                        f"endpoint={self.engagement.endpoint_silence_ms}ms"
                    )

                else:
                    print(f"[NOVA] Unknown command: {cmd}")

            except Exception as e:
                logger.error(f"Keyboard listener error: {e}")
                await asyncio.sleep(0.5)

    def _read_input(self) -> Optional[str]:
        try:
            return sys.stdin.readline()
        except EOFError:
            return None

    async def _enroll_speaker(self) -> None:
        """Interactive speaker enrollment — record voice samples."""
        if not self.speaker.is_available:
            logger.error("Speaker verification model not available.")
            return

        logger.info("Starting speaker enrollment. Speak naturally for 10 seconds...")
        samples = []
        start = time.time()

        while time.time() - start < 10:
            chunk = self.mic.get_chunk(timeout=0.05)
            if chunk is not None:
                samples.append(chunk)
            await asyncio.sleep(0.01)

        if not samples:
            logger.error("No audio captured during enrollment.")
            return

        full = np.concatenate(samples)
        segment_len = 16000 * 2  # 2 seconds
        segments = [full[i:i+segment_len] for i in range(0, len(full) - segment_len, segment_len)]

        if not segments:
            logger.error("Not enough audio for enrollment.")
            return

        success = self.speaker.enroll(segments)
        if success:
            logger.info("Speaker enrollment complete!")
        else:
            logger.error("Speaker enrollment failed.")

    async def stop(self) -> None:
        self._running = False
        self.mic.stop()
        await self.stt.close()
        await self.turn_analyzer.cleanup()
        if self.tts._connected:
            await self.tts.close()
        if self._ws:
            await self._ws.close()
        logger.info("Client stopped.")


async def main():
    client = AssistantClient()
    try:
        await client.start()
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
