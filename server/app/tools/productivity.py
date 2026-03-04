"""Productivity tools — todo lists, timers, alarms, reminders."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any

import aiosqlite

from server.app.tools.base import BaseTool
from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolResult


class TodoAddTool(BaseTool):
    name = "todo.add"
    description = "Add a new task to the to-do list."
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The task description"},
            "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"], "description": "Task priority"},
            "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD format, optional)"},
        },
        "required": ["text"],
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        text = kwargs["text"]
        priority = kwargs.get("priority", "normal")
        due_date = kwargs.get("due_date")

        cursor = await self._db.execute(
            "INSERT INTO todos (text, priority, due_date) VALUES (?, ?, ?)",
            (text, priority, due_date),
        )
        await self._db.commit()
        return ToolResult(
            tool_name=self.name, success=True,
            result={"id": cursor.lastrowid, "text": text, "priority": priority, "due_date": due_date},
        )


class TodoListTool(BaseTool):
    name = "todo.list"
    description = "List tasks from the to-do list. Can filter by status."
    parameters_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["pending", "completed", "all"], "description": "Filter by status (default: pending)"},
            "limit": {"type": "integer", "description": "Max tasks to return (default: 20)"},
        },
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        status = kwargs.get("status", "pending")
        limit = int(kwargs.get("limit", 20))

        if status == "all":
            rows = await self._db.execute_fetchall(
                "SELECT id, text, status, priority, due_date, created_at FROM todos ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        else:
            rows = await self._db.execute_fetchall(
                "SELECT id, text, status, priority, due_date, created_at FROM todos WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )

        tasks = [{"id": r[0], "text": r[1], "status": r[2], "priority": r[3], "due_date": r[4], "created_at": r[5]} for r in rows]
        return ToolResult(tool_name=self.name, success=True, result={"tasks": tasks, "count": len(tasks)})


class TodoCompleteTool(BaseTool):
    name = "todo.complete"
    description = "Mark a task as completed. Accepts task ID or partial text match."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "Task ID to complete"},
            "text": {"type": "string", "description": "Partial text to match (if ID not known)"},
        },
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        task_id = kwargs.get("task_id")
        text = kwargs.get("text")

        if task_id:
            await self._db.execute(
                "UPDATE todos SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (task_id,),
            )
        elif text:
            await self._db.execute(
                "UPDATE todos SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE text LIKE ? AND status = 'pending'",
                (f"%{text}%",),
            )
        else:
            return ToolResult(tool_name=self.name, success=False, result={"error": "Provide task_id or text"})

        await self._db.commit()
        return ToolResult(tool_name=self.name, success=True, result={"status": "completed"})


class TodoDeleteTool(BaseTool):
    name = "todo.delete"
    description = "Delete a task from the to-do list."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "Task ID to delete"},
        },
        "required": ["task_id"],
    }
    requires_confirmation = True
    sensitivity = SensitivityLevel.MEDIUM

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        task_id = kwargs["task_id"]
        await self._db.execute("DELETE FROM todos WHERE id = ?", (task_id,))
        await self._db.commit()
        return ToolResult(tool_name=self.name, success=True, result={"deleted": task_id})


# ------------------------------------------------------------------
# Timers
# ------------------------------------------------------------------

class _TimerStore:
    """Shared timer state (in-memory active timers + DB persistence)."""
    _active_timers: dict[int, asyncio.Task] = {}
    _callbacks: list = []

    @classmethod
    def register_callback(cls, cb):
        cls._callbacks.append(cb)


class TimerSetTool(BaseTool):
    name = "timer.set"
    description = "Set a countdown timer. Specify duration in seconds or natural description like '5 minutes'."
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Timer label (e.g., 'pasta timer')"},
            "duration_seconds": {"type": "number", "description": "Duration in seconds"},
        },
        "required": ["duration_seconds"],
    }

    def __init__(self, db: aiosqlite.Connection, event_bus=None):
        self._db = db
        self._event_bus = event_bus

    async def execute(self, **kwargs: Any) -> ToolResult:
        name = kwargs.get("name", "Timer")
        duration = float(kwargs["duration_seconds"])
        now = time.time()

        cursor = await self._db.execute(
            "INSERT INTO timers (name, duration_seconds, started_at, ends_at) VALUES (?, ?, ?, ?)",
            (name, duration, now, now + duration),
        )
        await self._db.commit()
        timer_id = cursor.lastrowid

        # Schedule the timer callback
        async def _fire():
            await asyncio.sleep(duration)
            await self._db.execute("UPDATE timers SET fired = 1 WHERE id = ?", (timer_id,))
            await self._db.commit()
            if self._event_bus:
                from server.app.behaviors.events import Event
                self._event_bus.publish(Event(
                    type="timer.fired",
                    data={"timer_id": timer_id, "name": name, "duration": duration},
                    source="timer",
                ))

        task = asyncio.create_task(_fire())
        _TimerStore._active_timers[timer_id] = task

        minutes = int(duration // 60)
        seconds = int(duration % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        return ToolResult(
            tool_name=self.name, success=True,
            result={"timer_id": timer_id, "name": name, "duration": time_str, "ends_at": now + duration},
        )


class TimerListTool(BaseTool):
    name = "timer.list"
    description = "Show all active (unfired) timers."
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        now = time.time()
        rows = await self._db.execute_fetchall(
            "SELECT id, name, duration_seconds, started_at, ends_at FROM timers WHERE fired = 0 AND ends_at > ?",
            (now,),
        )
        timers = []
        for r in rows:
            remaining = max(0, r[4] - now)
            timers.append({
                "id": r[0], "name": r[1], "duration": r[2],
                "remaining_seconds": round(remaining, 1),
            })
        return ToolResult(tool_name=self.name, success=True, result={"timers": timers, "count": len(timers)})


class TimerCancelTool(BaseTool):
    name = "timer.cancel"
    description = "Cancel an active timer."
    parameters_schema = {
        "type": "object",
        "properties": {
            "timer_id": {"type": "integer", "description": "Timer ID to cancel"},
            "name": {"type": "string", "description": "Timer name to cancel (if ID not known)"},
        },
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        timer_id = kwargs.get("timer_id")
        name = kwargs.get("name")

        if timer_id:
            await self._db.execute("UPDATE timers SET fired = 1 WHERE id = ?", (timer_id,))
        elif name:
            row = await self._db.execute_fetchall(
                "SELECT id FROM timers WHERE name LIKE ? AND fired = 0 LIMIT 1", (f"%{name}%",)
            )
            if row:
                timer_id = row[0][0]
                await self._db.execute("UPDATE timers SET fired = 1 WHERE id = ?", (timer_id,))
            else:
                return ToolResult(tool_name=self.name, success=False, result={"error": f"No active timer matching '{name}'"})
        else:
            return ToolResult(tool_name=self.name, success=False, result={"error": "Provide timer_id or name"})

        # Cancel the asyncio task
        if timer_id and timer_id in _TimerStore._active_timers:
            _TimerStore._active_timers[timer_id].cancel()
            del _TimerStore._active_timers[timer_id]

        await self._db.commit()
        return ToolResult(tool_name=self.name, success=True, result={"cancelled": timer_id})


# ------------------------------------------------------------------
# Alarms
# ------------------------------------------------------------------

class AlarmSetTool(BaseTool):
    name = "alarm.set"
    description = "Set an alarm for a specific time (HH:MM format, 24-hour)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "time": {"type": "string", "description": "Alarm time in HH:MM format (24-hour)"},
            "name": {"type": "string", "description": "Alarm label"},
            "days_of_week": {"type": "string", "description": "Comma-separated days (mon,tue,wed,thu,fri,sat,sun) or empty for one-time"},
        },
        "required": ["time"],
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        alarm_time = kwargs["time"]
        name = kwargs.get("name", "Alarm")
        days = kwargs.get("days_of_week", "")

        cursor = await self._db.execute(
            "INSERT INTO alarms (name, time, days_of_week) VALUES (?, ?, ?)",
            (name, alarm_time, days),
        )
        await self._db.commit()
        return ToolResult(
            tool_name=self.name, success=True,
            result={"alarm_id": cursor.lastrowid, "time": alarm_time, "name": name, "days": days or "one-time"},
        )


class AlarmListTool(BaseTool):
    name = "alarm.list"
    description = "Show all alarms."
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        rows = await self._db.execute_fetchall(
            "SELECT id, name, time, days_of_week, enabled FROM alarms ORDER BY time",
        )
        alarms = [{"id": r[0], "name": r[1], "time": r[2], "days": r[3], "enabled": bool(r[4])} for r in rows]
        return ToolResult(tool_name=self.name, success=True, result={"alarms": alarms, "count": len(alarms)})


class AlarmCancelTool(BaseTool):
    name = "alarm.cancel"
    description = "Cancel/delete an alarm."
    parameters_schema = {
        "type": "object",
        "properties": {
            "alarm_id": {"type": "integer", "description": "Alarm ID to cancel"},
        },
        "required": ["alarm_id"],
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        alarm_id = kwargs["alarm_id"]
        await self._db.execute("DELETE FROM alarms WHERE id = ?", (alarm_id,))
        await self._db.commit()
        return ToolResult(tool_name=self.name, success=True, result={"cancelled": alarm_id})


# ------------------------------------------------------------------
# Reminders
# ------------------------------------------------------------------

class ReminderSetTool(BaseTool):
    name = "reminder.set"
    description = "Set a reminder. Specify time as Unix timestamp or natural language offset like 'in 30 minutes', 'tomorrow at 9am'."
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "What to remind about"},
            "remind_at": {"type": "number", "description": "Unix timestamp for when to remind"},
            "offset_minutes": {"type": "number", "description": "Minutes from now (alternative to remind_at)"},
        },
        "required": ["text"],
    }

    def __init__(self, db: aiosqlite.Connection, event_bus=None):
        self._db = db
        self._event_bus = event_bus

    async def execute(self, **kwargs: Any) -> ToolResult:
        text = kwargs["text"]
        remind_at = kwargs.get("remind_at")
        offset = kwargs.get("offset_minutes")

        if offset:
            remind_at = time.time() + (float(offset) * 60)
        elif remind_at:
            remind_at = float(remind_at)
        else:
            remind_at = time.time() + 3600  # Default: 1 hour from now

        cursor = await self._db.execute(
            "INSERT INTO reminders (text, remind_at) VALUES (?, ?)",
            (text, remind_at),
        )
        await self._db.commit()
        reminder_id = cursor.lastrowid

        # Schedule the reminder
        delay = max(0, remind_at - time.time())

        async def _fire():
            await asyncio.sleep(delay)
            await self._db.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
            await self._db.commit()
            if self._event_bus:
                from server.app.behaviors.events import Event
                self._event_bus.publish(Event(
                    type="reminder.fired",
                    data={"reminder_id": reminder_id, "text": text},
                    source="reminder",
                ))

        asyncio.create_task(_fire())

        dt = datetime.fromtimestamp(remind_at)
        return ToolResult(
            tool_name=self.name, success=True,
            result={"reminder_id": reminder_id, "text": text, "remind_at": dt.strftime("%Y-%m-%d %H:%M")},
        )


class ReminderListTool(BaseTool):
    name = "reminder.list"
    description = "Show upcoming (unfired) reminders."
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        now = time.time()
        rows = await self._db.execute_fetchall(
            "SELECT id, text, remind_at FROM reminders WHERE fired = 0 AND remind_at > ? ORDER BY remind_at",
            (now,),
        )
        reminders = []
        for r in rows:
            dt = datetime.fromtimestamp(r[2])
            reminders.append({"id": r[0], "text": r[1], "remind_at": dt.strftime("%Y-%m-%d %H:%M")})
        return ToolResult(tool_name=self.name, success=True, result={"reminders": reminders, "count": len(reminders)})
