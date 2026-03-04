"""Deterministic mapping from MASSIVE intents → NOVA tools + slot→param conversion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from shared.schemas.tool_calls import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolMapping:
    """Result of resolving a MASSIVE intent to a NOVA tool."""
    tool_name: str
    kwargs: dict[str, Any]
    response_fn: Callable[[ToolResult], str]


class IntentToolMapper:
    """Maps MASSIVE intents to NOVA tool names with slot→parameter conversion."""

    def __init__(self):
        # Import response templates (they live in fastpath to avoid circular imports)
        # We'll set these via set_response_fns() after fastpath imports us
        self._response_fns: dict[str, Callable] = {}

    def set_response_fns(self, fns: dict[str, Callable]) -> None:
        """Inject response template functions from fastpath module."""
        self._response_fns = fns

    def _rfn(self, key: str) -> Callable:
        return self._response_fns.get(key, lambda r: "Done." if r.success else (r.error or "Something went wrong."))

    def resolve(self, intent: str, text: str, slots: dict[str, str]) -> Optional[ToolMapping]:
        """Resolve a MASSIVE intent + slots to a NOVA tool mapping.

        Returns None if the intent should fall through to LLM.
        """
        handler = self._HANDLERS.get(intent)
        if handler is None:
            return None
        try:
            return handler(self, text, slots)
        except Exception as e:
            logger.warning(f"IntentToolMapper error for {intent}: {e}")
            return None

    # ------------------------------------------------------------------
    # Handler methods (one per mapped MASSIVE intent)
    # ------------------------------------------------------------------

    def _alarm_set(self, text: str, slots: dict) -> Optional[ToolMapping]:
        if "timer" in text.lower():
            duration = _parse_duration_from_slot(slots.get("time", ""))
            if duration is None:
                return None  # can't parse → fall through
            return ToolMapping(
                tool_name="timer.set",
                kwargs={"duration_seconds": duration, "name": slots.get("alarm_type", "Timer")},
                response_fn=self._rfn("timer_set"),
            )
        time_str = _parse_alarm_time_from_slot(slots.get("time", ""))
        if time_str is None:
            return None
        return ToolMapping(
            tool_name="alarm.set",
            kwargs={"time": time_str},
            response_fn=self._rfn("alarm_set"),
        )

    def _alarm_query(self, text: str, slots: dict) -> Optional[ToolMapping]:
        if "timer" in text.lower():
            return ToolMapping("timer.list", {}, self._rfn("timer_list"))
        return ToolMapping("alarm.list", {}, self._rfn("alarm_list"))

    def _alarm_remove(self, text: str, slots: dict) -> Optional[ToolMapping]:
        if "timer" in text.lower():
            name = slots.get("alarm_type", "")
            return ToolMapping("timer.cancel", {"name": name} if name else {}, self._rfn("timer_cancel"))
        return ToolMapping("alarm.cancel", {}, self._rfn("alarm_cancel"))

    def _calendar_query(self, text: str, slots: dict) -> Optional[ToolMapping]:
        lower = text.lower()
        date_slot = slots.get("date", "").lower()
        if "today" in date_slot or "today" in lower:
            return ToolMapping("calendar.today", {}, self._rfn("calendar_today"))
        if "tomorrow" in date_slot or "tomorrow" in lower:
            return ToolMapping("calendar.tomorrow", {}, self._rfn("calendar_tomorrow"))
        return ToolMapping("calendar.list_events", {}, self._rfn("calendar_list"))

    def _email_query(self, text: str, slots: dict) -> Optional[ToolMapping]:
        return ToolMapping("email.gmail.list", {}, self._rfn("email_list"))

    def _iot_light_on(self, text: str, slots: dict) -> Optional[ToolMapping]:
        device = slots.get("device_type", slots.get("house_place", "lights"))
        return ToolMapping("home.command", {"device": device, "action": "on"}, self._rfn("home_command"))

    def _iot_light_off(self, text: str, slots: dict) -> Optional[ToolMapping]:
        device = slots.get("device_type", slots.get("house_place", "lights"))
        return ToolMapping("home.command", {"device": device, "action": "off"}, self._rfn("home_command"))

    def _iot_light_change(self, text: str, slots: dict) -> Optional[ToolMapping]:
        device = slots.get("device_type", slots.get("house_place", "lights"))
        return ToolMapping("home.command", {"device": device, "action": "toggle"}, self._rfn("home_command"))

    def _iot_light_dim(self, text: str, slots: dict) -> Optional[ToolMapping]:
        device = slots.get("device_type", slots.get("house_place", "lights"))
        return ToolMapping("home.command", {"device": device, "action": "dim"}, self._rfn("home_command"))

    def _iot_light_up(self, text: str, slots: dict) -> Optional[ToolMapping]:
        device = slots.get("device_type", slots.get("house_place", "lights"))
        return ToolMapping("home.command", {"device": device, "action": "brighten"}, self._rfn("home_command"))

    def _iot_wemo_on(self, text: str, slots: dict) -> Optional[ToolMapping]:
        device = slots.get("device_type", slots.get("house_place", "device"))
        return ToolMapping("home.command", {"device": device, "action": "on"}, self._rfn("home_command"))

    def _iot_wemo_off(self, text: str, slots: dict) -> Optional[ToolMapping]:
        device = slots.get("device_type", slots.get("house_place", "device"))
        return ToolMapping("home.command", {"device": device, "action": "off"}, self._rfn("home_command"))

    def _lists_add(self, text: str, slots: dict) -> Optional[ToolMapping]:
        item_text = slots.get("list_item", slots.get("todo", ""))
        if not item_text:
            # Try to extract from the text itself, after removing common prefixes
            item_text = _extract_list_item(text)
        if not item_text:
            return None
        return ToolMapping("todo.add", {"text": item_text}, self._rfn("todo_add"))

    def _lists_query(self, text: str, slots: dict) -> Optional[ToolMapping]:
        return ToolMapping("todo.list", {}, self._rfn("todo_list"))

    def _lists_remove(self, text: str, slots: dict) -> Optional[ToolMapping]:
        item_text = slots.get("list_item", slots.get("todo", ""))
        if not item_text:
            item_text = _extract_list_item(text)
        if not item_text:
            return None
        return ToolMapping("todo.complete", {"text": item_text}, self._rfn("todo_complete"))

    def _play_music(self, text: str, slots: dict) -> Optional[ToolMapping]:
        query = (
            slots.get("music_item", "")
            or slots.get("song_name", "")
            or slots.get("artist_name", "")
            or slots.get("playlist_name", "")
            or slots.get("music_genre", "")
        )
        if not query:
            # Extract anything after "play"
            m = re.search(r"\bplay\s+(.+)", text, re.IGNORECASE)
            query = m.group(1).strip() if m else ""
        if not query:
            return None
        return ToolMapping("spotify.play", {"query": query}, self._rfn("spotify_play"))

    def _music_query(self, text: str, slots: dict) -> Optional[ToolMapping]:
        lower = text.lower()
        if any(kw in lower for kw in ("playing", "current", "listening", "what song", "what's on")):
            return ToolMapping("spotify.now_playing", {}, self._rfn("spotify_now_playing"))
        query = slots.get("music_item", slots.get("song_name", slots.get("artist_name", "")))
        if query:
            return ToolMapping("spotify.search", {"query": query}, self._rfn("spotify_search"))
        return ToolMapping("spotify.now_playing", {}, self._rfn("spotify_now_playing"))

    def _music_settings(self, text: str, slots: dict) -> Optional[ToolMapping]:
        lower = text.lower()
        if any(kw in lower for kw in ("pause", "stop")):
            return ToolMapping("spotify.pause", {}, self._rfn("spotify_pause"))
        if any(kw in lower for kw in ("skip", "next")):
            return ToolMapping("spotify.skip", {}, self._rfn("spotify_skip"))
        if "resume" in lower or "unpause" in lower:
            return ToolMapping("spotify.play", {"query": ""}, self._rfn("spotify_play"))
        # Shuffle, repeat, etc. → fall through to LLM
        return None

    def _music_dislikeness(self, text: str, slots: dict) -> Optional[ToolMapping]:
        """'skip this song', 'I don't like this' → skip."""
        return ToolMapping("spotify.skip", {}, self._rfn("spotify_skip"))

    def _music_likeness(self, text: str, slots: dict) -> Optional[ToolMapping]:
        """'I like this song' → now_playing (acknowledge)."""
        return ToolMapping("spotify.now_playing", {}, self._rfn("spotify_now_playing"))

    def _volume_up(self, text: str, slots: dict) -> Optional[ToolMapping]:
        return ToolMapping("spotify.volume", {"volume": "+20"}, self._rfn("spotify_volume"))

    def _volume_down(self, text: str, slots: dict) -> Optional[ToolMapping]:
        return ToolMapping("spotify.volume", {"volume": "-20"}, self._rfn("spotify_volume"))

    def _volume_mute(self, text: str, slots: dict) -> Optional[ToolMapping]:
        # "pause the music" sometimes classified as volume_mute
        lower = text.lower()
        if any(kw in lower for kw in ("pause", "stop")):
            return ToolMapping("spotify.pause", {}, self._rfn("spotify_pause"))
        return ToolMapping("spotify.volume", {"volume": 0}, self._rfn("spotify_volume"))

    def _volume_other(self, text: str, slots: dict) -> Optional[ToolMapping]:
        # Try to extract a number from the text
        m = re.search(r"(\d+)", text)
        if m:
            return ToolMapping("spotify.volume", {"volume": int(m.group(1))}, self._rfn("spotify_volume"))
        return None

    def _weather_query(self, text: str, slots: dict) -> Optional[ToolMapping]:
        lower = text.lower()
        city = slots.get("place_name", "").strip() or "auto"
        if "forecast" in lower:
            return ToolMapping("weather.forecast", {"city": city}, self._rfn("weather_forecast"))
        if any(kw in lower for kw in ("wear", "outfit", "dress", "clothing")):
            return ToolMapping("weather.outfit", {"city": city}, self._rfn("weather_outfit"))
        if "hourly" in lower:
            return ToolMapping("weather.hourly", {"city": city}, self._rfn("weather_hourly"))
        if "alert" in lower or "warning" in lower:
            return ToolMapping("weather.alerts", {}, self._rfn("weather_alerts"))
        return ToolMapping("weather.current", {"city": city}, self._rfn("weather_current"))

    def _qa_maths(self, text: str, slots: dict) -> Optional[ToolMapping]:
        # Extract math expression from text
        expr = _extract_math_expression(text)
        if not expr:
            return None
        return ToolMapping("calc.math", {"expression": expr}, self._rfn("calc_math"))

    def _qa_currency(self, text: str, slots: dict) -> Optional[ToolMapping]:
        # Try to parse conversion from slots
        value_str = slots.get("currency_name", "") or slots.get("unit_name", "")
        m = re.search(r"(\d+\.?\d*)\s*(\w+)\s+(?:to|in)\s+(\w+)", text, re.IGNORECASE)
        if m:
            return ToolMapping("calc.convert", {
                "value": float(m.group(1)),
                "from_unit": m.group(2),
                "to_unit": m.group(3),
            }, self._rfn("calc_convert"))
        return None

    def _play_radio(self, text: str, slots: dict) -> Optional[ToolMapping]:
        query = slots.get("radio_name", slots.get("music_item", "radio"))
        return ToolMapping("spotify.play", {"query": query}, self._rfn("spotify_play"))

    def _play_podcasts(self, text: str, slots: dict) -> Optional[ToolMapping]:
        query = slots.get("podcast_name", slots.get("music_item", "podcast"))
        return ToolMapping("spotify.play", {"query": query}, self._rfn("spotify_play"))

    # ------------------------------------------------------------------
    # Intent → handler dispatch table
    # ------------------------------------------------------------------

    _HANDLERS: dict[str, Callable] = {
        "alarm_set": _alarm_set,
        "alarm_query": _alarm_query,
        "alarm_remove": _alarm_remove,
        "calendar_query": _calendar_query,
        "email_query": _email_query,
        "iot_hue_lighton": _iot_light_on,
        "iot_hue_lightoff": _iot_light_off,
        "iot_hue_lightchange": _iot_light_change,
        "iot_hue_lightdim": _iot_light_dim,
        "iot_hue_lightup": _iot_light_up,
        "iot_wemo_on": _iot_wemo_on,
        "iot_wemo_off": _iot_wemo_off,
        "lists_createoradd": _lists_add,
        "lists_query": _lists_query,
        "lists_remove": _lists_remove,
        "play_music": _play_music,
        "music_query": _music_query,
        "music_settings": _music_settings,
        "music_dislikeness": _music_dislikeness,
        "music_likeness": _music_likeness,
        "audio_volume_up": _volume_up,
        "audio_volume_down": _volume_down,
        "audio_volume_mute": _volume_mute,
        "audio_volume_other": _volume_other,
        "weather_query": _weather_query,
        "qa_maths": _qa_maths,
        "qa_currency": _qa_currency,
        "play_radio": _play_radio,
        "play_podcasts": _play_podcasts,
    }


# ------------------------------------------------------------------
# Slot parsing helpers
# ------------------------------------------------------------------

# Duration units
_DURATION_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
    "half an hour": 1800, "half hour": 1800,
}

# Number words
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty five": 25, "thirty": 30, "forty": 40, "forty five": 45,
    "fifty": 50, "sixty": 60, "ninety": 90,
}


def _parse_number(s: str) -> Optional[float]:
    """Parse a number from string, supporting both digits and words."""
    s = s.strip().lower()
    if not s:
        return None
    # Try direct float parse
    try:
        return float(s)
    except ValueError:
        pass
    # Try number words
    if s in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[s])
    # Try "a" as 1 (e.g., "a minute")
    if s in ("a", "an"):
        return 1.0
    return None


def _parse_duration_from_slot(time_slot: str) -> Optional[float]:
    """Parse a MASSIVE time slot like "5 minutes", "an hour", "thirty seconds" to seconds."""
    if not time_slot:
        return None

    s = time_slot.strip().lower()

    # Try pattern: <number> <unit>
    m = re.match(r"^(.+?)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?|half\s+(?:an?\s+)?hours?)$", s)
    if m:
        num = _parse_number(m.group(1))
        unit = m.group(2)
        if num is not None:
            multiplier = _DURATION_UNITS.get(unit, 60)
            return num * multiplier

    # Try "half an hour"
    if "half" in s and "hour" in s:
        return 1800.0

    # Try bare number (assume minutes)
    num = _parse_number(s)
    if num is not None:
        return num * 60

    return None


def _parse_alarm_time_from_slot(time_slot: str) -> Optional[str]:
    """Parse a MASSIVE time slot into HH:MM 24-hour format."""
    if not time_slot:
        return None

    s = time_slot.strip().lower()

    # Try HH:MM pattern
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?$", s)
    if m:
        return _format_alarm_time(int(m.group(1)), int(m.group(2)), m.group(3))

    # Try "<number> <ampm>" like "7 am", "seven pm"
    m = re.match(r"^(.+?)\s*(am|pm|a\.m\.|p\.m\.)$", s)
    if m:
        num = _parse_number(m.group(1))
        if num is not None:
            hour = int(num)
            return _format_alarm_time(hour, 0, m.group(2))

    # Try "<number> o'clock"
    m = re.match(r"^(.+?)\s*o'?clock$", s)
    if m:
        num = _parse_number(m.group(1))
        if num is not None:
            return _format_alarm_time(int(num), 0, None)

    # Try bare number
    num = _parse_number(s)
    if num is not None:
        return _format_alarm_time(int(num), 0, None)

    return None


def _format_alarm_time(hour: int, minute: int, ampm: Optional[str]) -> str:
    """Convert hour/minute/ampm to HH:MM 24-hour format."""
    if ampm:
        ampm = ampm.lower().replace(".", "")
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
    return f"{hour:02d}:{minute:02d}"


def _extract_list_item(text: str) -> str:
    """Extract the to-do item text from a natural language utterance."""
    lower = text.lower()
    # Remove common prefixes
    for prefix in [
        "add ", "put ", "create a task ", "create task ", "add a task ",
        "add to my to do list ", "add to my todo list ", "add to my list ",
        "put on my to do list ", "put on my list ",
        "remove ", "complete ", "check off ", "mark ", "delete ",
        "remove from my to do list ", "remove from my list ",
    ]:
        if lower.startswith(prefix):
            return text[len(prefix):].strip()
    return ""


def _extract_math_expression(text: str) -> str:
    """Extract a math expression from natural language."""
    lower = text.lower()
    # Remove common prefixes
    for prefix in [
        "what's ", "what is ", "calculate ", "compute ", "solve ",
        "how much is ", "what does ", "what do ",
    ]:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # Remove trailing "?" and common suffixes
    text = re.sub(r"\s*\??\s*$", "", text)
    text = re.sub(r"\s+equal(?:s)?\s*$", "", text, flags=re.IGNORECASE)

    return text.strip()
