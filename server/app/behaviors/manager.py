"""Behavior manager — evaluates rules and dispatches proactive actions.

This is the core of NOVA's proactive behavior system. It:
1. Subscribes to events from the EventBus
2. Evaluates deterministic Python rules (NO LLM involved in rule matching)
3. Queues actions with priority and cooldown management
4. Dispatches actions (speak, device_command, notification)

The LLM is only called AFTER a rule fires, to generate natural speech.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from server.app.behaviors.events import Event, EventBus
from server.app.behaviors.rules import BehaviorRule, get_default_rules

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PendingAction:
    """An action queued for execution."""
    priority: int
    timestamp: float = field(compare=False)
    rule_name: str = field(compare=False)
    action_type: str = field(compare=False)
    action_config: dict[str, Any] = field(compare=False, default_factory=dict)
    event: Event = field(compare=False, default=None)


@dataclass
class BehaviorLog:
    """Record of a behavior activation for debugging."""
    timestamp: float
    rule_name: str
    event_type: str
    action_type: str
    result: str  # "executed", "cooldown", "disabled", "error"
    details: str = ""


class BehaviorManager:
    """Manages proactive behavior rules and dispatches actions."""

    def __init__(
        self,
        event_bus: EventBus,
        rules: list[BehaviorRule] | None = None,
        speak_callback=None,
        device_callback=None,
        notify_callback=None,
    ):
        self._event_bus = event_bus
        self._rules: dict[str, BehaviorRule] = {}
        self._cooldowns: dict[str, float] = {}  # rule_name → last_fired timestamp
        self._action_queue: asyncio.PriorityQueue[PendingAction] = asyncio.PriorityQueue()
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False

        # Action callbacks (set by orchestrator)
        self._speak_callback = speak_callback      # async fn(prompt, max_tokens) → str
        self._device_callback = device_callback    # async fn(device, action) → dict
        self._notify_callback = notify_callback    # async fn(message) → None

        # Activity log
        self._log: list[BehaviorLog] = []
        self._max_log_size = 100

        # Load rules
        for rule in (rules or get_default_rules()):
            self._rules[rule.name] = rule

    @property
    def rules(self) -> dict[str, BehaviorRule]:
        return self._rules

    async def start(self) -> None:
        """Start the action consumer loop and subscribe to events."""
        if self._running:
            return
        self._running = True

        # Subscribe to all event types that rules care about
        event_types = set()
        for rule in self._rules.values():
            event_types.add(rule.event_type)

        for et in event_types:
            self._event_bus.subscribe(et, self._on_event)

        # Start consumer
        self._consumer_task = asyncio.create_task(self._action_consumer())
        logger.info(f"BehaviorManager started with {len(self._rules)} rules, "
                    f"listening to {len(event_types)} event types")

    async def stop(self) -> None:
        """Stop the behavior manager."""
        self._running = False
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        logger.info("BehaviorManager stopped")

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: BehaviorRule) -> None:
        """Add or replace a rule."""
        self._rules[rule.name] = rule
        self._event_bus.subscribe(rule.event_type, self._on_event)

    def remove_rule(self, name: str) -> bool:
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    def enable_rule(self, name: str) -> bool:
        if name in self._rules:
            self._rules[name].enabled = True
            return True
        return False

    def disable_rule(self, name: str) -> bool:
        if name in self._rules:
            self._rules[name].enabled = False
            return True
        return False

    def get_rule(self, name: str) -> Optional[BehaviorRule]:
        return self._rules.get(name)

    def list_rules(self) -> list[dict[str, Any]]:
        """Return rules as serializable dicts."""
        return [
            {
                "name": r.name,
                "description": r.description,
                "event_type": r.event_type,
                "action_type": r.action_type,
                "cooldown_seconds": r.cooldown_seconds,
                "enabled": r.enabled,
                "priority": r.priority,
            }
            for r in self._rules.values()
        ]

    def get_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent behavior activation log."""
        return [
            {
                "timestamp": entry.timestamp,
                "rule_name": entry.rule_name,
                "event_type": entry.event_type,
                "action_type": entry.action_type,
                "result": entry.result,
                "details": entry.details,
            }
            for entry in self._log[-limit:]
        ]

    # ------------------------------------------------------------------
    # Event handling (deterministic rule matching)
    # ------------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        """Called when an event is published. Matches rules deterministically."""
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.event_type != event.type:
                continue

            # Check cooldown
            last_fired = self._cooldowns.get(rule.name, 0)
            if time.time() - last_fired < rule.cooldown_seconds:
                self._add_log(rule.name, event.type, rule.action_type, "cooldown")
                continue

            # Evaluate condition (deterministic Python, no LLM)
            try:
                if not rule.condition(event):
                    continue
            except Exception as e:
                logger.error(f"Rule '{rule.name}' condition error: {e}")
                self._add_log(rule.name, event.type, rule.action_type, "error", str(e))
                continue

            # Condition met — queue the action
            action = PendingAction(
                priority=rule.priority,
                timestamp=time.time(),
                rule_name=rule.name,
                action_type=rule.action_type,
                action_config=dict(rule.action_config),
                event=event,
            )

            try:
                self._action_queue.put_nowait(action)
                self._cooldowns[rule.name] = time.time()
                logger.info(f"Behavior rule fired: {rule.name} (event: {event.type})")
            except asyncio.QueueFull:
                logger.warning(f"Action queue full, dropping action from rule '{rule.name}'")

    # ------------------------------------------------------------------
    # Action consumer
    # ------------------------------------------------------------------

    async def _action_consumer(self) -> None:
        """Consume and execute queued actions."""
        while self._running:
            try:
                action = await asyncio.wait_for(
                    self._action_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            try:
                await self._execute_action(action)
                self._add_log(
                    action.rule_name, action.event.type if action.event else "",
                    action.action_type, "executed"
                )
            except Exception as e:
                logger.error(f"Action execution error ({action.rule_name}): {e}")
                self._add_log(
                    action.rule_name, action.event.type if action.event else "",
                    action.action_type, "error", str(e)
                )

    async def _execute_action(self, action: PendingAction) -> None:
        """Execute a single action."""
        if action.action_type == "speak":
            if self._speak_callback:
                prompt = action.action_config.get("prompt", "")
                # Format prompt with event data (e.g., {name}, {text}, {duration})
                if action.event and action.event.data:
                    try:
                        prompt = prompt.format(**action.event.data)
                    except (KeyError, IndexError):
                        pass  # If formatting fails, use raw prompt
                max_tokens = action.action_config.get("max_tokens", 80)
                await self._speak_callback(prompt, max_tokens)
            else:
                logger.warning(f"No speak callback set, skipping action from '{action.rule_name}'")

        elif action.action_type == "device_command":
            if self._device_callback:
                device = action.action_config.get("device", "")
                device_action = action.action_config.get("action", "")
                await self._device_callback(device, device_action)
            else:
                logger.warning(f"No device callback set, skipping action from '{action.rule_name}'")

        elif action.action_type == "notification":
            if self._notify_callback:
                message = action.action_config.get("message", "")
                await self._notify_callback(message)
            else:
                logger.warning(f"No notify callback set, skipping action from '{action.rule_name}'")

        else:
            logger.warning(f"Unknown action type: {action.action_type}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _add_log(self, rule_name: str, event_type: str, action_type: str,
                 result: str, details: str = "") -> None:
        self._log.append(BehaviorLog(
            timestamp=time.time(),
            rule_name=rule_name,
            event_type=event_type,
            action_type=action_type,
            result=result,
            details=details,
        ))
        if len(self._log) > self._max_log_size:
            self._log = self._log[-self._max_log_size:]
