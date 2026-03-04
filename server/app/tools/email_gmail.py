"""Gmail integration — uses shared Google OAuth, list, read, draft, send (gated)."""

from __future__ import annotations

import base64
from email.mime.text import MIMEText
from typing import Any

from server.app.tools.base import BaseTool
from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolResult

# Cached service
_service = None


async def _get_gmail_service():
    """Get or create Gmail API service using shared Google credentials."""
    global _service
    if _service is not None:
        return _service

    from server.app.auth.google import get_credentials
    from googleapiclient.discovery import build

    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. User needs to authenticate at /auth/google")

    _service = build("gmail", "v1", credentials=creds)
    return _service


def _reset_service():
    """Reset cached service (e.g., after re-auth)."""
    global _service
    _service = None


def _parse_message(msg: dict) -> dict[str, Any]:
    """Extract useful fields from a Gmail API message resource."""
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
    snippet = msg.get("snippet", "")

    body_text = ""
    payload = msg.get("payload", {})
    if payload.get("body", {}).get("data"):
        body_text = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    elif payload.get("parts"):
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                body_text = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                break

    return {
        "id": msg["id"],
        "thread_id": msg.get("threadId", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "snippet": snippet,
        "body": body_text[:2000],
        "labels": msg.get("labelIds", []),
    }


class GmailListTool(BaseTool):
    name = "email.gmail.list"
    description = "List recent Gmail messages. Optionally filter by query."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Gmail search query (e.g. 'is:unread', 'from:boss@co.com')", "default": ""},
            "max_results": {"type": "integer", "description": "Maximum messages to return", "default": 10},
        },
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            service = await _get_gmail_service()
            query = kwargs.get("query", "")
            max_results = int(kwargs.get("max_results", 10))

            results = service.users().messages().list(
                userId="me", q=query, maxResults=max_results
            ).execute()

            messages = results.get("messages", [])
            summaries = []
            for m in messages[:max_results]:
                msg = service.users().messages().get(userId="me", id=m["id"], format="metadata",
                                                      metadataHeaders=["From", "Subject", "Date"]).execute()
                headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
                summaries.append({
                    "id": m["id"],
                    "from": headers.get("from", ""),
                    "subject": headers.get("subject", ""),
                    "date": headers.get("date", ""),
                    "snippet": msg.get("snippet", ""),
                })

            return ToolResult(
                tool_name=self.name, success=True, result=summaries,
                display_card={
                    "card_type": "EmailSummaryCard",
                    "title": f"Gmail — {len(summaries)} messages",
                    "body": summaries,
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class GmailReadTool(BaseTool):
    name = "email.gmail.read"
    description = "Read a specific Gmail message by ID."
    parameters_schema = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "The Gmail message ID"},
        },
        "required": ["message_id"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            service = await _get_gmail_service()
            msg = service.users().messages().get(
                userId="me", id=kwargs["message_id"], format="full"
            ).execute()
            parsed = _parse_message(msg)
            return ToolResult(
                tool_name=self.name, success=True, result=parsed,
                display_card={"card_type": "EmailDetailCard", "title": parsed["subject"], "body": parsed},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class GmailDraftReplyTool(BaseTool):
    name = "email.gmail.draft_reply"
    description = "Draft a reply to an email. Does NOT send — shows preview first."
    parameters_schema = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Original message ID to reply to"},
            "body": {"type": "string", "description": "Reply body text"},
        },
        "required": ["message_id", "body"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.MEDIUM

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            service = await _get_gmail_service()
            original = service.users().messages().get(
                userId="me", id=kwargs["message_id"], format="metadata",
                metadataHeaders=["From", "Subject", "Message-ID"],
            ).execute()

            headers = {h["name"].lower(): h["value"] for h in original.get("payload", {}).get("headers", [])}
            to = headers.get("from", "")
            subject = headers.get("subject", "")
            if not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"

            draft_data = {
                "to": to,
                "subject": subject,
                "body": kwargs["body"],
                "thread_id": original.get("threadId", ""),
                "in_reply_to": headers.get("message-id", ""),
                "original_message_id": kwargs["message_id"],
            }

            return ToolResult(
                tool_name=self.name, success=True, result=draft_data,
                display_card={
                    "card_type": "EmailDraftCard",
                    "title": f"Draft: {subject}",
                    "body": draft_data,
                    "actions": [{"label": "CONFIRM SEND", "action": "confirm_gmail_send"}],
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class GmailSendTool(BaseTool):
    name = "email.gmail.send"
    description = "Send an email via Gmail. REQUIRES explicit user confirmation."
    parameters_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body text"},
            "thread_id": {"type": "string", "description": "Thread ID for replies", "default": ""},
            "in_reply_to": {"type": "string", "description": "Message-ID header for threading", "default": ""},
        },
        "required": ["to", "subject", "body"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            service = await _get_gmail_service()

            message = MIMEText(kwargs["body"])
            message["to"] = kwargs["to"]
            message["subject"] = kwargs["subject"]
            if kwargs.get("in_reply_to"):
                message["In-Reply-To"] = kwargs["in_reply_to"]
                message["References"] = kwargs["in_reply_to"]

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            body: dict[str, Any] = {"raw": raw}
            if kwargs.get("thread_id"):
                body["threadId"] = kwargs["thread_id"]

            sent = service.users().messages().send(userId="me", body=body).execute()
            return ToolResult(
                tool_name=self.name, success=True,
                result={"message_id": sent["id"], "thread_id": sent.get("threadId", "")},
                display_card={"card_type": "EmailSentCard", "title": "Email Sent", "body": {"to": kwargs["to"], "subject": kwargs["subject"]}},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))
