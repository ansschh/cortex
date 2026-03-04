"""Policy gates — non-LLM logic that controls what actions are allowed."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolDefinition

logger = logging.getLogger(__name__)

CONFIRMATION_PHRASES = {
    "send it", "post it", "yes confirm", "yes, confirm",
    "confirm", "go ahead", "do it", "yes send", "yes post",
    "approved", "yes", "yeah send it", "yeah post it",
}


@dataclass
class GateResult:
    allowed: bool
    reason: str = ""
    requires_confirmation: bool = False
    requires_speaker_verification: bool = False


# ---------------------------------------------------------------------------
# Dynamic permissions — voice-configurable confirmation overrides
# ---------------------------------------------------------------------------

# Natural language → tool pattern mapping
ACTION_TO_TOOLS: dict[str, str] = {
    "sending emails": "email.*",
    "send email": "email.*",
    "send emails": "email.*",
    "emails": "email.*",
    "posting to slack": "slack.post_message",
    "slack messages": "slack.post_message",
    "slack": "slack.post_message",
    "setting timers": "timer.set",
    "timers": "timer.set",
    "creating alarms": "alarm.set",
    "alarms": "alarm.set",
    "setting reminders": "reminder.set",
    "reminders": "reminder.set",
    "adding todos": "todo.add",
    "todos": "todo.add",
    "deleting todos": "todo.delete",
    "controlling devices": "home.command",
    "devices": "home.command",
    "lights": "home.command",
    "playing music": "spotify.play",
    "music": "spotify.play",
    "deleting calendar events": "calendar.delete_event",
    "calendar events": "calendar.*",
    "creating calendar events": "calendar.create_event",
}


class DynamicPermissions:
    """Voice-configurable permission overrides stored in preferences."""

    PREFIX = "confirm_before:"

    # Actions that require confirmation by default (irreversible/outbound)
    AUTO_HIGH_RISK = {
        "email.gmail.send", "email.outlook.send",
        "slack.post_message",
        "calendar.delete_event",
        "todo.delete",
    }

    def __init__(self, memory_store):
        self._store = memory_store
        self._cache: dict[str, bool] = {}  # tool_name → requires_confirmation

    async def load(self):
        """Load all confirm_before:* preferences into cache."""
        try:
            prefs = await self._store.list_all_preferences()
            for p in prefs:
                if p.key.startswith(self.PREFIX):
                    tool = p.key[len(self.PREFIX):]
                    self._cache[tool] = p.value.lower() == "true"
        except Exception as e:
            logger.error(f"Failed to load dynamic permissions: {e}")

    def should_confirm(self, tool_name: str) -> bool | None:
        """Check if dynamic override exists. Returns None if no override."""
        if tool_name in self._cache:
            return self._cache[tool_name]
        # Check category-level (e.g., "email.*" matches "email.gmail.send")
        for pattern, val in self._cache.items():
            if pattern.endswith(".*") and tool_name.startswith(pattern[:-2]):
                return val
        return None  # no override → fall through to static sensitivity

    async def set_rule(self, tool_pattern: str, require: bool):
        """Set a dynamic confirmation rule."""
        from shared.schemas.memory import Preference
        key = f"{self.PREFIX}{tool_pattern}"
        await self._store.set_preference(Preference(
            key=key, value=str(require).lower(), source="voice",
        ))
        self._cache[tool_pattern] = require

    async def remove_rule(self, tool_pattern: str):
        """Remove a dynamic confirmation rule."""
        self._cache.pop(tool_pattern, None)

    def list_rules(self) -> dict[str, bool]:
        """List all active dynamic rules."""
        return dict(self._cache)

    def is_auto_high_risk(self, tool_name: str) -> bool:
        """Check if this tool is in the default high-risk set."""
        return tool_name in self.AUTO_HIGH_RISK


# ---------------------------------------------------------------------------
# Permission command parser — detects "ask me before X" voice commands
# ---------------------------------------------------------------------------

_ENABLE_PATTERNS = [
    re.compile(r"(?:always |from now on |start )?(?:ask|confirm|check with) (?:me )?before (.+)", re.IGNORECASE),
    re.compile(r"(?:require|need) confirmation (?:for|before) (.+)", re.IGNORECASE),
]

_DISABLE_PATTERNS = [
    re.compile(r"(?:don'?t|stop|no longer) (?:ask|confirm|check) (?:me )?before (.+)", re.IGNORECASE),
    re.compile(r"(?:don'?t|stop) (?:requiring|needing) confirmation (?:for|before) (.+)", re.IGNORECASE),
    re.compile(r"(?:skip|remove) confirmation (?:for|on) (.+)", re.IGNORECASE),
]

_LIST_PATTERNS = [
    re.compile(r"what (?:needs|requires) (?:my )?confirmation", re.IGNORECASE),
    re.compile(r"(?:list|show|what are) (?:my )?(?:confirmation|permission) (?:rules|settings)", re.IGNORECASE),
]


def parse_permission_command(text: str) -> dict | None:
    """Detect 'ask me before X' or 'don't ask before X' commands.

    Returns:
        {"action": "enable"|"disable"|"list", "tool_pattern": str, "description": str}
        or None if not a permission command.
    """
    text_clean = text.strip()

    for pat in _LIST_PATTERNS:
        if pat.search(text_clean):
            return {"action": "list"}

    for pat in _ENABLE_PATTERNS:
        m = pat.search(text_clean)
        if m:
            desc = m.group(1).strip().rstrip(".")
            tool_pattern = _resolve_action_to_tool(desc)
            if tool_pattern:
                return {"action": "enable", "tool_pattern": tool_pattern, "description": desc}

    for pat in _DISABLE_PATTERNS:
        m = pat.search(text_clean)
        if m:
            desc = m.group(1).strip().rstrip(".")
            tool_pattern = _resolve_action_to_tool(desc)
            if tool_pattern:
                return {"action": "disable", "tool_pattern": tool_pattern, "description": desc}

    return None


def _resolve_action_to_tool(description: str) -> str | None:
    """Map a natural language action description to a tool pattern."""
    desc_lower = description.lower().strip()
    # Direct match
    if desc_lower in ACTION_TO_TOOLS:
        return ACTION_TO_TOOLS[desc_lower]
    # Fuzzy: check if description contains any known key
    for key, pattern in ACTION_TO_TOOLS.items():
        if key in desc_lower or desc_lower in key:
            return pattern
    return None


# ---------------------------------------------------------------------------
# Policy gate
# ---------------------------------------------------------------------------

class PolicyGate:
    """Evaluates whether a tool call should proceed based on security policy."""

    def __init__(self, speaker_verify_threshold: float = 0.65):
        self._threshold = speaker_verify_threshold

    def evaluate(
        self,
        tool_def: ToolDefinition,
        speaker_verified: bool = False,
        speaker_confidence: float = 0.0,
        dynamic_perms: DynamicPermissions | None = None,
    ) -> GateResult:
        # 1. Check dynamic override first
        if dynamic_perms:
            override = dynamic_perms.should_confirm(tool_def.name)
            if override is True:
                return GateResult(
                    allowed=False,
                    reason="You asked me to confirm before doing this.",
                    requires_confirmation=True,
                )
            elif override is False:
                # User explicitly said no confirmation needed
                return GateResult(allowed=True)

        # 2. Check auto-high-risk list (default confirmation for irreversible actions)
        if dynamic_perms and dynamic_perms.is_auto_high_risk(tool_def.name):
            # Only enforce if not already handled by static sensitivity
            if tool_def.sensitivity == SensitivityLevel.LOW:
                return GateResult(
                    allowed=False,
                    reason="This is an irreversible action — confirming first.",
                    requires_confirmation=True,
                )

        # 3. Fall through to static sensitivity levels
        if tool_def.sensitivity == SensitivityLevel.LOW:
            return GateResult(allowed=True)

        if tool_def.sensitivity == SensitivityLevel.MEDIUM:
            if tool_def.requires_confirmation:
                return GateResult(
                    allowed=False,
                    reason="This action requires your explicit confirmation.",
                    requires_confirmation=True,
                )
            return GateResult(allowed=True)

        if tool_def.sensitivity == SensitivityLevel.HIGH:
            if not speaker_verified or speaker_confidence < self._threshold:
                return GateResult(
                    allowed=False,
                    reason="Speaker verification required for this action.",
                    requires_speaker_verification=True,
                    requires_confirmation=True,
                )
            return GateResult(
                allowed=False,
                reason="This action requires your explicit confirmation before proceeding.",
                requires_confirmation=True,
            )

        return GateResult(allowed=False, reason="Unknown sensitivity level.")

    @staticmethod
    def is_confirmation_phrase(text: str) -> bool:
        """Check if the user's utterance is a confirmation phrase."""
        normalized = text.strip().lower().rstrip("!.,?")
        return normalized in CONFIRMATION_PHRASES

    @staticmethod
    def is_memory_command(text: str) -> bool:
        """Check if the user's utterance is a deterministic memory command."""
        lower = text.strip().lower()
        prefixes = [
            "add to memories",
            "remember that",
            "save this",
            "remember this",
            "store this",
            "note that",
            "make a note",
        ]
        return any(lower.startswith(p) for p in prefixes)

    @staticmethod
    def extract_memory_text(text: str) -> str:
        """Extract the memory content from a memory command."""
        lower = text.strip().lower()
        prefixes = [
            "add to memories that ",
            "add to memories: ",
            "add to memories ",
            "remember that ",
            "save this: ",
            "save this ",
            "remember this: ",
            "remember this ",
            "store this: ",
            "store this ",
            "note that ",
            "make a note: ",
            "make a note that ",
            "make a note ",
        ]
        for p in prefixes:
            if lower.startswith(p):
                return text.strip()[len(p):]
        return text.strip()
