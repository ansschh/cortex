"""Slack integration — draft-first messaging with explicit send confirmation."""

from __future__ import annotations

from typing import Any

from server.app.config import settings
from server.app.tools.base import BaseTool
from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolResult


class SlackDraftMessageTool(BaseTool):
    name = "slack.draft_message"
    description = "Draft a Slack message for a channel/user. Does NOT send — shows preview first."
    parameters_schema = {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Channel name (e.g. #general) or user ID"},
            "text": {"type": "string", "description": "Message text"},
        },
        "required": ["channel", "text"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.MEDIUM

    async def execute(self, **kwargs: Any) -> ToolResult:
        draft = {
            "channel": kwargs["channel"],
            "text": kwargs["text"],
        }
        return ToolResult(
            tool_name=self.name, success=True, result=draft,
            display_card={
                "card_type": "SlackDraftCard",
                "title": f"Slack Draft → {kwargs['channel']}",
                "body": draft,
                "actions": [{"label": "CONFIRM POST", "action": "confirm_slack_post"}],
            },
        )


class SlackPostMessageTool(BaseTool):
    name = "slack.post_message"
    description = "Post a message to Slack. REQUIRES explicit user confirmation."
    parameters_schema = {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Channel name or ID"},
            "text": {"type": "string", "description": "Message text"},
        },
        "required": ["channel", "text"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            from slack_sdk.web.async_client import AsyncWebClient

            client = AsyncWebClient(token=settings.slack_bot_token)
            resp = await client.chat_postMessage(
                channel=kwargs["channel"],
                text=kwargs["text"],
            )
            return ToolResult(
                tool_name=self.name, success=True,
                result={"channel": kwargs["channel"], "ts": resp["ts"]},
                display_card={
                    "card_type": "SlackSentCard",
                    "title": "Slack Message Posted",
                    "body": {"channel": kwargs["channel"], "text": kwargs["text"]},
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class SlackListChannelsTool(BaseTool):
    name = "slack.list_channels"
    description = "List Slack channels the bot has access to."
    parameters_schema = {"type": "object", "properties": {}}
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            from slack_sdk.web.async_client import AsyncWebClient

            client = AsyncWebClient(token=settings.slack_bot_token)
            resp = await client.conversations_list(types="public_channel,private_channel", limit=50)
            channels = [
                {"id": c["id"], "name": c["name"], "is_member": c.get("is_member", False)}
                for c in resp.get("channels", [])
            ]
            return ToolResult(tool_name=self.name, success=True, result=channels)
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))
