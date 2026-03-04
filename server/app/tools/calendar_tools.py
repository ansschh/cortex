"""Google Calendar tools — list, create, delete events.

Uses the shared Google OAuth credentials from server.app.auth.google.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from server.app.tools.base import BaseTool
from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolResult

logger = logging.getLogger(__name__)

# Cached service
_cal_service = None


async def _get_calendar_service():
    """Get or create Calendar API service using shared Google credentials."""
    global _cal_service
    if _cal_service is not None:
        return _cal_service

    from server.app.auth.google import get_credentials
    from googleapiclient.discovery import build

    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. User needs to authenticate at /auth/google")

    _cal_service = build("calendar", "v3", credentials=creds)
    return _cal_service


def _reset_service():
    """Reset cached service (e.g., after re-auth)."""
    global _cal_service
    _cal_service = None


def _format_event(event: dict) -> dict:
    """Format a Google Calendar event into a clean dict."""
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id", ""),
        "summary": event.get("summary", "(No title)"),
        "start": start.get("dateTime", start.get("date", "")),
        "end": end.get("dateTime", end.get("date", "")),
        "location": event.get("location", ""),
        "description": event.get("description", "")[:200],
        "status": event.get("status", ""),
        "organizer": event.get("organizer", {}).get("email", ""),
    }


class CalendarListEventsTool(BaseTool):
    name = "calendar.list_events"
    description = "List upcoming Google Calendar events. Can specify number of days to look ahead."
    parameters_schema = {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "Number of days to look ahead (default 7)", "default": 7},
            "max_results": {"type": "integer", "description": "Max events to return (default 10)", "default": 10},
            "query": {"type": "string", "description": "Optional search query to filter events", "default": ""},
        },
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        service = await _get_calendar_service()
        days = min(int(kwargs.get("days", 7)), 30)
        max_results = min(int(kwargs.get("max_results", 10)), 50)
        query = kwargs.get("query", "")

        now = datetime.now(timezone.utc)
        time_max = now + timedelta(days=days)

        params = {
            "calendarId": "primary",
            "timeMin": now.isoformat(),
            "timeMax": time_max.isoformat(),
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if query:
            params["q"] = query

        result = service.events().list(**params).execute()
        events = [_format_event(e) for e in result.get("items", [])]

        return ToolResult(
            tool_name=self.name, success=True, result=events,
            display_card={
                "card_type": "CalendarCard",
                "title": f"Calendar — next {days} days ({len(events)} events)",
                "body": events,
            },
        )


class CalendarCreateEventTool(BaseTool):
    name = "calendar.create_event"
    description = "Create a new Google Calendar event."
    parameters_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Event title"},
            "start_time": {"type": "string", "description": "Start time in ISO format (e.g. '2024-03-15T14:00:00')"},
            "end_time": {"type": "string", "description": "End time in ISO format. If not provided, defaults to 1 hour after start."},
            "description": {"type": "string", "description": "Event description (optional)", "default": ""},
            "location": {"type": "string", "description": "Event location (optional)", "default": ""},
        },
        "required": ["summary", "start_time"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        service = await _get_calendar_service()

        start_str = kwargs["start_time"]
        end_str = kwargs.get("end_time", "")

        # Parse start time
        try:
            start_dt = datetime.fromisoformat(start_str)
        except ValueError:
            return ToolResult(tool_name=self.name, success=False,
                              error=f"Invalid start time format: {start_str}. Use ISO format like '2024-03-15T14:00:00'")

        # Default end = start + 1 hour
        if end_str:
            try:
                end_dt = datetime.fromisoformat(end_str)
            except ValueError:
                end_dt = start_dt + timedelta(hours=1)
        else:
            end_dt = start_dt + timedelta(hours=1)

        event_body = {
            "summary": kwargs["summary"],
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Los_Angeles"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Los_Angeles"},
        }
        if kwargs.get("description"):
            event_body["description"] = kwargs["description"]
        if kwargs.get("location"):
            event_body["location"] = kwargs["location"]

        created = service.events().insert(calendarId="primary", body=event_body).execute()

        return ToolResult(
            tool_name=self.name, success=True,
            result=_format_event(created),
            display_card={
                "card_type": "CalendarEventCreated",
                "title": f"Created: {created.get('summary', '')}",
                "body": _format_event(created),
            },
        )


class CalendarDeleteEventTool(BaseTool):
    name = "calendar.delete_event"
    description = "Delete a Google Calendar event by ID."
    parameters_schema = {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "description": "The event ID to delete"},
        },
        "required": ["event_id"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        service = await _get_calendar_service()
        event_id = kwargs["event_id"]

        try:
            service.events().delete(calendarId="primary", eventId=event_id).execute()
        except Exception as e:
            if "404" in str(e) or "notFound" in str(e):
                return ToolResult(tool_name=self.name, success=False,
                                  error=f"Event '{event_id}' not found")
            raise

        return ToolResult(tool_name=self.name, success=True,
                          result={"deleted": event_id})


class CalendarTodayTool(BaseTool):
    name = "calendar.today"
    description = "Show today's schedule — all events for today."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        service = await _get_calendar_service()

        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        result = service.events().list(
            calendarId="primary",
            timeMin=start_of_day.isoformat(),
            timeMax=end_of_day.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=20,
        ).execute()

        events = [_format_event(e) for e in result.get("items", [])]
        return ToolResult(
            tool_name=self.name, success=True, result=events,
            display_card={
                "card_type": "CalendarCard",
                "title": f"Today's Schedule ({len(events)} events)",
                "body": events,
            },
        )


class CalendarTomorrowTool(BaseTool):
    name = "calendar.tomorrow"
    description = "Show tomorrow's schedule — all events for tomorrow."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        service = await _get_calendar_service()

        now = datetime.now(timezone.utc)
        start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        result = service.events().list(
            calendarId="primary",
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=20,
        ).execute()

        events = [_format_event(e) for e in result.get("items", [])]
        return ToolResult(
            tool_name=self.name, success=True, result=events,
            display_card={
                "card_type": "CalendarCard",
                "title": f"Tomorrow's Schedule ({len(events)} events)",
                "body": events,
            },
        )
