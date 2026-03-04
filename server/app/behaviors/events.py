"""Simple event bus for NOVA's proactive behavior system.

Events are published by various sources (vision, timers, integrations) and
consumed by BehaviorRules. No external dependencies — just async pub/sub.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A structured event from any NOVA subsystem."""
    type: str           # e.g., "vision.motion_detected", "timer.fired"
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""    # e.g., "vision", "timer", "integration"


# Callback type: async or sync function taking an Event
EventCallback = Callable[[Event], Any]


class EventBus:
    """In-process pub/sub for NOVA events.

    Thread-safe: publish() can be called from any context.
    Subscribers are called in the event loop via create_task.
    """

    def __init__(self):
        self._subscribers: dict[str, list[EventCallback]] = defaultdict(list)
        self._wildcard_subscribers: list[EventCallback] = []
        self._event_log: list[Event] = []  # Rolling log for debugging
        self._max_log_size: int = 200
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Subscribe to a specific event type. Use '*' for all events."""
        if event_type == "*":
            self._wildcard_subscribers.append(callback)
        else:
            self._subscribers[event_type].append(callback)
        logger.debug(f"EventBus: subscribed to '{event_type}'")

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Remove a subscription."""
        if event_type == "*":
            self._wildcard_subscribers = [c for c in self._wildcard_subscribers if c != callback]
        elif event_type in self._subscribers:
            self._subscribers[event_type] = [c for c in self._subscribers[event_type] if c != callback]

    def publish(self, event: Event) -> None:
        """Publish an event. Dispatches to all matching subscribers.

        Safe to call from sync or async context.
        """
        # Log for debugging
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]

        # Collect all matching callbacks
        callbacks = list(self._wildcard_subscribers)
        if event.type in self._subscribers:
            callbacks.extend(self._subscribers[event.type])

        # Also match prefix patterns: "vision.*" matches "vision.motion_detected"
        prefix = event.type.rsplit(".", 1)[0] + ".*" if "." in event.type else ""
        if prefix and prefix in self._subscribers:
            callbacks.extend(self._subscribers[prefix])

        if not callbacks:
            return

        # Dispatch each callback
        for cb in callbacks:
            try:
                result = cb(event)
                # If the callback is a coroutine, schedule it
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        pass  # No running loop
            except Exception as e:
                logger.error(f"EventBus callback error for '{event.type}': {e}")

    def get_recent_events(self, event_type: str | None = None, limit: int = 50) -> list[Event]:
        """Get recent events, optionally filtered by type."""
        events = self._event_log
        if event_type:
            events = [e for e in events if e.type == event_type or e.type.startswith(event_type.rstrip("*"))]
        return events[-limit:]

    def clear_log(self) -> None:
        self._event_log.clear()
