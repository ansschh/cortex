"""FastPath — sub-50ms tool execution via JointBERT NLU, bypassing all LLM calls.

Intercepts user text BEFORE any LLM call. A JointBERT ONNX model (~4ms) classifies
intent and extracts slot parameters in a single forward pass. If confidence is high
enough and the intent maps to a NOVA tool, executes directly with a template response.
Falls through to the normal LLM pipeline for anything it can't handle.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from server.app.nlu.inference import NLUInference, NLUResult
from server.app.nlu.intent_map import IntentToolMapper, ToolMapping
from server.app.tools.base import ToolRegistry
from shared.schemas.tool_calls import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class FastPathResult:
    tool_name: str
    kwargs: dict[str, Any]
    response_fn: Callable[[ToolResult], str]
    intent: str = ""
    confidence: float = 0.0
    slots: dict[str, str] = field(default_factory=dict)


def _human_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s and not h:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " and ".join(parts) if parts else "0 seconds"


# ------------------------------------------------------------------
# Response template functions
# ------------------------------------------------------------------

def _todo_add_response(r: ToolResult) -> str:
    return f"Got it, added {r.result['text']} to your list."


def _todo_list_response(r: ToolResult) -> str:
    count = r.result["count"]
    if count == 0:
        return "Your to-do list is all clear, nothing on it."
    tasks = r.result["tasks"]
    items = ", ".join(t["text"] for t in tasks[:3])
    if count == 1:
        return f"You've got one thing on your list: {items}."
    if count <= 3:
        return f"You've got {count} things to do: {items}."
    return f"You've got {count} things on your list. The top ones are {items}."


def _todo_complete_response(r: ToolResult) -> str:
    if not r.success:
        return "Hmm, I couldn't find that task."
    return "Nice, marked that one as done."


def _timer_set_response(r: ToolResult) -> str:
    return f"Alright, {r.result['duration']} on the clock. I'll let you know when it's up."


def _timer_list_response(r: ToolResult) -> str:
    count = r.result["count"]
    if count == 0:
        return "No timers running right now."
    timers = r.result["timers"]
    parts = []
    for t in timers[:3]:
        remaining = int(t["remaining_seconds"])
        m, s = divmod(remaining, 60)
        parts.append(f"{t['name']} with {m} minutes and {s} seconds left")
    return "You've got " + ", ".join(parts) + "."


def _timer_cancel_response(r: ToolResult) -> str:
    if not r.success:
        return r.result.get("error", "There's no timer to cancel.")
    return "Timer's cancelled."


def _alarm_set_response(r: ToolResult) -> str:
    return f"You're all set, alarm going off at {r.result['time']}."


def _alarm_list_response(r: ToolResult) -> str:
    count = r.result["count"]
    if count == 0:
        return "You don't have any alarms set."
    alarms = r.result["alarms"]
    parts = [f"{a['name']} at {a['time']}" for a in alarms[:3]]
    return "Your alarms: " + ", ".join(parts) + "."


def _alarm_cancel_response(r: ToolResult) -> str:
    if not r.success:
        return "There's no alarm to cancel."
    return "Alarm's off."


def _reminder_set_response(r: ToolResult) -> str:
    return f"I'll remind you about {r.result['text']} at {r.result['remind_at']}."


def _reminder_list_response(r: ToolResult) -> str:
    count = r.result["count"]
    if count == 0:
        return "No reminders coming up."
    reminders = r.result["reminders"]
    parts = [f"{rm['text']} at {rm['remind_at']}" for rm in reminders[:3]]
    if count == 1:
        return f"You've got one reminder: {parts[0]}."
    return f"You've got {count} reminders: " + ", ".join(parts) + "."


def _weather_current_response(r: ToolResult) -> str:
    w = r.result
    return f"Right now it's {w['temp_f']:.0f} degrees and {w['description']} in {w['city']}."


def _weather_forecast_response(r: ToolResult) -> str:
    city = r.result["city"]
    forecasts = r.result.get("forecasts", [])
    if not forecasts:
        return f"I couldn't pull up the forecast for {city}."
    parts = [f"{f['date']}, {f['temp_f']:.0f} degrees and {f['description']}" for f in forecasts[:3]]
    return f"Here's the forecast for {city}: " + ". ".join(parts) + "."


def _weather_hourly_response(r: ToolResult) -> str:
    city = r.result["city"]
    hours = r.result.get("hourly", [])
    if not hours:
        return f"No hourly data for {city} right now."
    parts = [f"{h['time'].split(' ')[1][:5]} will be {h['temp_f']:.0f} degrees" for h in hours[:4]]
    return f"Over the next few hours in {city}: " + ", ".join(parts) + "."


def _weather_alerts_response(r: ToolResult) -> str:
    count = r.result["count"]
    if count == 0:
        return "No weather alerts right now, all clear."
    alerts = r.result["alerts"]
    parts = [a["headline"] for a in alerts[:2]]
    return "Heads up, " + " ".join(parts)


def _weather_outfit_response(r: ToolResult) -> str:
    w = r.result
    items = ", ".join(w["recommendation"][:4])
    return f"It's {w['temp_f']:.0f} degrees and {w['conditions']} out there. I'd go with {items}."


def _spotify_play_response(r: ToolResult) -> str:
    if not r.success:
        return r.result.get("error", "Couldn't play that, sorry.")
    name = r.result["playing"]
    artist = r.result.get("artist", "")
    if artist:
        return f"Playing {name} by {artist}."
    return f"Playing {name} for you."


def _spotify_pause_response(r: ToolResult) -> str:
    return "Paused."


def _spotify_skip_response(r: ToolResult) -> str:
    return "Skipping."


def _spotify_queue_response(r: ToolResult) -> str:
    if not r.success:
        return r.result.get("error", "Couldn't add that to the queue.")
    return f"Added {r.result['queued']} by {r.result['artist']} to your queue."


def _spotify_now_playing_response(r: ToolResult) -> str:
    if r.result.get("status") == "Nothing playing":
        return "Nothing's playing right now."
    return f"Right now you're listening to {r.result['track']} by {r.result['artist']}."


def _spotify_volume_response(r: ToolResult) -> str:
    return f"Volume's at {r.result['volume']} percent."


def _spotify_playlist_response(r: ToolResult) -> str:
    count = r.result["count"]
    if count == 0:
        return "You don't have any playlists yet."
    names = [p["name"] for p in r.result["playlists"][:5]]
    return f"Here are your playlists: " + ", ".join(names) + "."


def _spotify_search_response(r: ToolResult) -> str:
    if not r.success:
        return r.result.get("error", "Search didn't work.")
    results = r.result["results"]
    if not results:
        return "Didn't find anything for that."
    parts = []
    for item in results[:3]:
        if "artist" in item:
            parts.append(f"{item['name']} by {item['artist']}")
        else:
            parts.append(item["name"])
    return "I found " + ", ".join(parts) + "."


def _study_start_response(r: ToolResult) -> str:
    return f"Study session started for {r.result['subject']}. Good luck!"


def _study_end_response(r: ToolResult) -> str:
    if not r.success:
        return "You don't have an active study session."
    return f"Nice work! You studied {r.result['subject']} for {r.result['duration_minutes']} minutes."


def _study_stats_response(r: ToolResult) -> str:
    s = r.result
    if s["total_minutes"] == 0:
        return f"No study time logged for {s['period']} yet."
    return f"You've put in {s['total_hours']} hours this {s['period']}."


def _calc_math_response(r: ToolResult) -> str:
    if not r.success:
        return f"Hmm, that didn't compute: {r.result.get('error', 'unknown error')}"
    return f"That's {r.result['result']}."


def _calc_convert_response(r: ToolResult) -> str:
    if not r.success:
        return r.result.get("error", "I can't do that conversion.")
    return f"That's {r.result['result']} {r.result['to']}."


def _flashcard_list_response(r: ToolResult) -> str:
    total = r.result["total_decks"]
    if total == 0:
        return "You haven't created any flashcard decks yet."
    names = [d["deck"] + f" with {d['card_count']} cards" for d in r.result["decks"][:5]]
    return f"You've got {total} deck{'s' if total != 1 else ''}: " + ", ".join(names) + "."


def _calendar_today_response(r: ToolResult) -> str:
    events = r.result
    if not events:
        return "Your calendar's clear today, nothing scheduled."
    parts = [e["summary"] + (" at " + e["start"].split("T")[1][:5] if "T" in e["start"] else "") for e in events[:3]]
    if len(events) == 1:
        return f"You've got one thing today: {parts[0]}."
    return f"You've got {len(events)} things today: " + ", ".join(parts) + "."


def _calendar_tomorrow_response(r: ToolResult) -> str:
    events = r.result
    if not events:
        return "Tomorrow's looking clear, nothing on the calendar."
    parts = [e["summary"] + (" at " + e["start"].split("T")[1][:5] if "T" in e["start"] else "") for e in events[:3]]
    if len(events) == 1:
        return f"Tomorrow you've got {parts[0]}."
    return f"Tomorrow you've got {len(events)} things: " + ", ".join(parts) + "."


def _calendar_list_response(r: ToolResult) -> str:
    events = r.result
    if not events:
        return "Nothing coming up on your calendar."
    parts = [e["summary"] for e in events[:4]]
    return f"Coming up you've got: " + ", ".join(parts) + "."


def _email_list_response(r: ToolResult) -> str:
    if not r.success:
        return r.result.get("error", "Couldn't check your email right now.")
    messages = r.result
    if isinstance(messages, dict):
        messages = messages.get("messages", [])
    if not messages:
        return "Inbox is empty, you're all caught up."
    count = len(messages)
    first = messages[0]
    subj = first.get("subject", "no subject")
    sender = first.get("from", "someone")
    if count == 1:
        return f"You've got one new email from {sender} about {subj}."
    return f"You've got {count} recent emails. The latest one's from {sender}, subject: {subj}."


def _home_command_response(r: ToolResult) -> str:
    if not r.success:
        return r.error or "That didn't work, the device might be offline."
    return f"Done, {r.result['device']} is {r.result['action']}."


def _home_list_response(r: ToolResult) -> str:
    devices = r.result
    if not devices:
        return "No smart devices set up yet."
    names = [d.get("name", "unknown") for d in devices[:5]]
    return f"Your devices: " + ", ".join(names) + "."


# ------------------------------------------------------------------
# Response function lookup table (used by IntentToolMapper)
# ------------------------------------------------------------------

_RESPONSE_FNS: dict[str, Callable] = {
    "todo_add": _todo_add_response,
    "todo_list": _todo_list_response,
    "todo_complete": _todo_complete_response,
    "timer_set": _timer_set_response,
    "timer_list": _timer_list_response,
    "timer_cancel": _timer_cancel_response,
    "alarm_set": _alarm_set_response,
    "alarm_list": _alarm_list_response,
    "alarm_cancel": _alarm_cancel_response,
    "reminder_set": _reminder_set_response,
    "reminder_list": _reminder_list_response,
    "weather_current": _weather_current_response,
    "weather_forecast": _weather_forecast_response,
    "weather_hourly": _weather_hourly_response,
    "weather_alerts": _weather_alerts_response,
    "weather_outfit": _weather_outfit_response,
    "spotify_play": _spotify_play_response,
    "spotify_pause": _spotify_pause_response,
    "spotify_skip": _spotify_skip_response,
    "spotify_queue": _spotify_queue_response,
    "spotify_now_playing": _spotify_now_playing_response,
    "spotify_volume": _spotify_volume_response,
    "spotify_playlist": _spotify_playlist_response,
    "spotify_search": _spotify_search_response,
    "study_start": _study_start_response,
    "study_end": _study_end_response,
    "study_stats": _study_stats_response,
    "calc_math": _calc_math_response,
    "calc_convert": _calc_convert_response,
    "flashcard_list": _flashcard_list_response,
    "calendar_today": _calendar_today_response,
    "calendar_tomorrow": _calendar_tomorrow_response,
    "calendar_list": _calendar_list_response,
    "email_list": _email_list_response,
    "home_command": _home_command_response,
    "home_list": _home_list_response,
}


# ------------------------------------------------------------------
# FastPath engine (JointBERT NLU)
# ------------------------------------------------------------------

class FastPath:
    """JointBERT NLU that bypasses all LLM calls for high-confidence intents."""

    def __init__(self, tool_registry: ToolRegistry, model_dir: str = "data/models/jointbert"):
        self._registry = tool_registry
        self._confidence_threshold = 0.7

        # Initialize NLU model
        self._nlu = NLUInference(model_dir)

        # Initialize intent→tool mapper and inject response functions
        self._mapper = IntentToolMapper()
        self._mapper.set_response_fns(_RESPONSE_FNS)

        logger.info(f"FastPath initialized (JointBERT NLU, threshold={self._confidence_threshold})")

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize STT output for more reliable matching."""
        s = text.strip()
        s = re.sub(r"\bwhats\b", "what's", s, flags=re.IGNORECASE)
        s = re.sub(r"\bhows\b", "how's", s, flags=re.IGNORECASE)
        s = re.sub(r"\btodays\b", "today's", s, flags=re.IGNORECASE)
        s = re.sub(r"\btomorrows\b", "tomorrow's", s, flags=re.IGNORECASE)
        s = re.sub(r"\bim\b", "i'm", s, flags=re.IGNORECASE)
        s = re.sub(r"\bdont\b", "don't", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwont\b", "won't", s, flags=re.IGNORECASE)
        s = re.sub(r"\bcant\b", "can't", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s)
        return s

    def try_match(self, text: str) -> Optional[FastPathResult]:
        """Run JointBERT inference and map to a tool. Returns None to fall through to LLM."""
        cleaned = self._normalize(text)

        if len(cleaned) < 3:
            return None

        t0 = time.perf_counter()
        nlu_result = self._nlu.predict(cleaned)
        infer_ms = (time.perf_counter() - t0) * 1000

        if nlu_result.confidence < self._confidence_threshold:
            logger.debug(
                f"FastPath skip: low confidence {nlu_result.intent}={nlu_result.confidence:.3f} "
                f"(threshold={self._confidence_threshold}) in {infer_ms:.1f}ms"
            )
            return None

        mapping = self._mapper.resolve(nlu_result.intent, cleaned, nlu_result.slots)
        if mapping is None:
            logger.debug(
                f"FastPath skip: no mapping for intent={nlu_result.intent} "
                f"(conf={nlu_result.confidence:.3f}) in {infer_ms:.1f}ms"
            )
            return None

        logger.info(
            f"FastPath [{mapping.tool_name}] intent={nlu_result.intent} "
            f"conf={nlu_result.confidence:.3f} slots={nlu_result.slots} in {infer_ms:.1f}ms"
        )

        return FastPathResult(
            tool_name=mapping.tool_name,
            kwargs=mapping.kwargs,
            response_fn=mapping.response_fn,
            intent=nlu_result.intent,
            confidence=nlu_result.confidence,
            slots=nlu_result.slots,
        )

    async def execute(self, result: FastPathResult) -> tuple[ToolResult, str]:
        """Execute the matched tool and format the spoken response."""
        tool = self._registry.get(result.tool_name)
        if not tool:
            return (
                ToolResult(tool_name=result.tool_name, success=False, error="Tool not found"),
                "Sorry, that tool isn't available right now.",
            )

        tool_result = await tool.safe_execute(**result.kwargs)

        try:
            spoken = result.response_fn(tool_result)
        except Exception as e:
            logger.warning(f"FastPath response template error: {e}")
            spoken = "Done." if tool_result.success else f"Error: {tool_result.error or 'something went wrong'}"

        # If tool failed and template didn't handle it, override
        if not tool_result.success and tool_result.error:
            spoken = tool_result.error

        return tool_result, spoken
