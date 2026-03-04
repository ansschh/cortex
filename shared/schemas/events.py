"""Event schemas — the contract between client, server, and UI."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClientState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    MIC_MUTED = "mic_muted"


class SensitivityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

class BaseEvent(BaseModel):
    event: str
    timestamp: float = Field(default_factory=time.time)
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Client → Server events
# ---------------------------------------------------------------------------

class WakeWordDetected(BaseEvent):
    event: str = "wakeword_detected"
    model_name: str = ""
    confidence: float = 0.0


class STTPartial(BaseEvent):
    event: str = "stt_partial"
    text: str = ""


class STTFinal(BaseEvent):
    event: str = "stt_final"
    text: str = ""
    language: str = "en"


class ClientStateUpdate(BaseEvent):
    event: str = "client_state"
    state: ClientState = ClientState.IDLE


class SpeakerVerified(BaseEvent):
    event: str = "speaker_verified"
    is_verified: bool = False
    confidence: float = 0.0
    speaker_label: str = ""


class UserConfirmation(BaseEvent):
    event: str = "user_confirmation"
    confirmed: bool = False
    pending_action_id: str = ""


# ---------------------------------------------------------------------------
# Server → Client events
# ---------------------------------------------------------------------------

class AssistantText(BaseEvent):
    event: str = "assistant_text"
    text: str = ""


class AssistantTTSText(BaseEvent):
    event: str = "assistant_tts_text"
    text: str = ""
    voice_id: Optional[str] = None
    use_local_tts: bool = False


class AssistantAudioControl(BaseEvent):
    event: str = "assistant_audio_control"
    action: str = "start"  # "start" | "stop"


class ToolRequest(BaseEvent):
    event: str = "tool_request"
    tool_name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Server → UI (Projector) events
# ---------------------------------------------------------------------------

class UICard(BaseModel):
    card_type: str
    card_id: str = ""
    title: str = ""
    body: Any = None
    actions: list[dict[str, str]] = Field(default_factory=list)
    priority: int = 0
    expires_at: Optional[float] = None


class UICardsUpdate(BaseEvent):
    event: str = "ui_cards_update"
    cards: list[UICard] = Field(default_factory=list)


class UIToast(BaseEvent):
    event: str = "ui_toast"
    message: str = ""
    level: str = "info"  # "info" | "success" | "warning" | "error"
    duration_ms: int = 4000


class UIStatusUpdate(BaseEvent):
    event: str = "ui_status_update"
    assistant_state: str = "idle"  # idle | listening | thinking | speaking
    speaker_verified: bool = False
    speaker_label: str = ""
    transcript: str = ""


# ---------------------------------------------------------------------------
# Generic event routing
# ---------------------------------------------------------------------------

CLIENT_EVENTS = {
    "wakeword_detected": WakeWordDetected,
    "stt_partial": STTPartial,
    "stt_final": STTFinal,
    "client_state": ClientStateUpdate,
    "speaker_verified": SpeakerVerified,
    "user_confirmation": UserConfirmation,
}

SERVER_EVENTS = {
    "assistant_text": AssistantText,
    "assistant_tts_text": AssistantTTSText,
    "assistant_audio_control": AssistantAudioControl,
    "tool_request": ToolRequest,
    "ui_cards_update": UICardsUpdate,
    "ui_toast": UIToast,
    "ui_status_update": UIStatusUpdate,
}


def parse_event(raw: dict[str, Any]) -> BaseEvent:
    """Parse a raw JSON dict into the appropriate event model."""
    event_type = raw.get("event", "")
    model = CLIENT_EVENTS.get(event_type) or SERVER_EVENTS.get(event_type)
    if model is None:
        return BaseEvent(**raw)
    return model(**raw)
