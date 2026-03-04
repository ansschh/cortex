"""Behavior rules — deterministic Python conditions, NOT LLM prompt injection.

Each rule defines:
  - A trigger event type
  - A Python condition function (evaluated deterministically)
  - An action to take when the condition is met
  - A cooldown to prevent spam

The LLM is ONLY called AFTER a rule fires, to generate natural speech.
Rules themselves never touch the LLM — this prevents hallucination.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from server.app.behaviors.events import Event


@dataclass
class BehaviorRule:
    """A single proactive behavior rule."""
    name: str                           # Unique identifier
    description: str                    # Human-readable description
    event_type: str                     # Event type to trigger on
    condition: Callable[[Event], bool]  # Deterministic Python condition
    action_type: str                    # "speak", "device_command", "notification"
    action_config: dict[str, Any] = field(default_factory=dict)
    cooldown_seconds: float = 300.0     # Min time between firings (5 min default)
    enabled: bool = True
    priority: int = 5                   # Lower = higher priority (1-10)


# ------------------------------------------------------------------
# Built-in rules
# ------------------------------------------------------------------

def _welcome_back_condition(event: Event) -> bool:
    """Fire when person returns after being absent for 5+ minutes."""
    return event.data.get("absence_duration", 0) > 300  # 5 minutes


def _stretch_reminder_condition(event: Event) -> bool:
    """Fire when person has been sitting still for 60+ minutes."""
    return event.data.get("duration", 0) > 3600  # 60 minutes


def _long_sitting_30min_condition(event: Event) -> bool:
    """Fire when person has been sitting still for 30+ minutes (gentler reminder)."""
    duration = event.data.get("duration", 0)
    return 1800 < duration <= 1860  # Only fire once around the 30-min mark


def _lights_on_empty_condition(event: Event) -> bool:
    """Fire when person leaves and it's late (lights probably still on)."""
    hour = time.localtime().tm_hour
    return hour >= 23 or hour < 5  # Between 11 PM and 5 AM


def _goodnight_condition(event: Event) -> bool:
    """Fire when person leaves room after midnight."""
    hour = time.localtime().tm_hour
    return 0 <= hour < 5


def _timer_fired_condition(event: Event) -> bool:
    """Always fire when a timer goes off."""
    return True


def _reminder_fired_condition(event: Event) -> bool:
    """Always fire when a reminder triggers."""
    return True


BUILT_IN_RULES: list[BehaviorRule] = [
    BehaviorRule(
        name="welcome_back",
        description="Greet the user when they return after being away for 5+ minutes",
        event_type="vision.person_appeared",
        condition=_welcome_back_condition,
        action_type="speak",
        action_config={
            "prompt": "The user just returned to the room after being away. Give a brief, warm welcome back. If you know about any pending notifications (new emails, reminders), mention them briefly. Keep it to 1-2 sentences max.",
            "max_tokens": 80,
        },
        cooldown_seconds=300,  # 5 min cooldown
        enabled=True,
        priority=3,
    ),
    BehaviorRule(
        name="stretch_reminder",
        description="Suggest a break when user has been sitting for 60+ minutes",
        event_type="vision.person_static",
        condition=_stretch_reminder_condition,
        action_type="speak",
        action_config={
            "prompt": "The user has been sitting still for over an hour. Gently suggest taking a short break or stretching. Be casual, not preachy. One sentence.",
            "max_tokens": 60,
        },
        cooldown_seconds=1800,  # 30 min cooldown (don't nag)
        enabled=True,
        priority=7,
    ),
    BehaviorRule(
        name="half_hour_check",
        description="Gentle posture/hydration check at 30 minutes of sitting",
        event_type="vision.person_static",
        condition=_long_sitting_30min_condition,
        action_type="speak",
        action_config={
            "prompt": "The user has been sitting for about 30 minutes. A very brief, casual reminder about hydration or posture. Keep it super short — just a few words, like a friend would say.",
            "max_tokens": 40,
        },
        cooldown_seconds=1800,
        enabled=False,  # Off by default, user can enable
        priority=8,
    ),
    BehaviorRule(
        name="lights_reminder",
        description="Remind about lights when leaving room late at night",
        event_type="vision.person_left",
        condition=_lights_on_empty_condition,
        action_type="speak",
        action_config={
            "prompt": "The user just left the room and it's late at night. The lights might still be on. Offer to turn them off. Keep it brief.",
            "max_tokens": 40,
        },
        cooldown_seconds=600,
        enabled=True,
        priority=4,
    ),
    BehaviorRule(
        name="goodnight",
        description="Say goodnight when user leaves very late",
        event_type="vision.person_left",
        condition=_goodnight_condition,
        action_type="speak",
        action_config={
            "prompt": "The user is leaving the room very late at night (after midnight). Wish them goodnight briefly and warmly. One short sentence.",
            "max_tokens": 40,
        },
        cooldown_seconds=3600,  # Once per hour max
        enabled=True,
        priority=5,
    ),
    BehaviorRule(
        name="timer_alert",
        description="Announce when a timer goes off",
        event_type="timer.fired",
        condition=_timer_fired_condition,
        action_type="speak",
        action_config={
            "prompt": "A timer just went off. The timer was named '{name}' and was set for {duration} seconds. Announce it casually, like 'Hey, your {name} timer is up!' Keep it to one sentence.",
            "max_tokens": 40,
        },
        cooldown_seconds=5,
        enabled=True,
        priority=1,  # High priority — timers are time-critical
    ),
    BehaviorRule(
        name="reminder_alert",
        description="Announce when a reminder fires",
        event_type="reminder.fired",
        condition=_reminder_fired_condition,
        action_type="speak",
        action_config={
            "prompt": "A reminder just fired. The reminder text is: '{text}'. Tell the user about it naturally, like 'Hey, just a heads up — {text}'. Keep it brief.",
            "max_tokens": 60,
        },
        cooldown_seconds=5,
        enabled=True,
        priority=1,
    ),
]


def get_default_rules() -> list[BehaviorRule]:
    """Return a copy of the built-in rules."""
    return list(BUILT_IN_RULES)
