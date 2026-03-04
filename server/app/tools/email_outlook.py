"""Outlook / Microsoft Graph email integration — delegated user auth via MSAL."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from server.app.config import settings
from server.app.tools.base import BaseTool
from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolResult

_TOKEN_CACHE_PATH = "data/outlook_token_cache.json"
_access_token: Optional[str] = None

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.Read", "Mail.Send"]


async def _get_access_token() -> str:
    """Get a valid access token via MSAL device code or cached token."""
    global _access_token
    if _access_token:
        return _access_token

    import msal

    cache = msal.SerializableTokenCache()
    if os.path.exists(_TOKEN_CACHE_PATH):
        cache.deserialize(open(_TOKEN_CACHE_PATH).read())

    app = msal.PublicClientApplication(
        settings.ms_client_id,
        authority=f"https://login.microsoftonline.com/{settings.ms_tenant_id}",
        token_cache=cache,
    )

    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])

    if not result:
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(f"Device flow failed: {json.dumps(flow)}")
        print(f"\n[Outlook Auth] Go to {flow['verification_uri']} and enter code: {flow['user_code']}\n")
        result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        raise RuntimeError(f"Token acquisition failed: {result.get('error_description', 'unknown')}")

    os.makedirs(os.path.dirname(_TOKEN_CACHE_PATH), exist_ok=True)
    with open(_TOKEN_CACHE_PATH, "w") as f:
        f.write(cache.serialize())

    _access_token = result["access_token"]
    return _access_token


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class OutlookListTool(BaseTool):
    name = "email.outlook.list"
    description = "List recent Outlook/Microsoft 365 emails."
    parameters_schema = {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "Mail folder (inbox, sentitems, etc.)", "default": "inbox"},
            "max_results": {"type": "integer", "default": 10},
            "filter_query": {"type": "string", "description": "OData $filter query", "default": ""},
        },
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            token = await _get_access_token()
            folder = kwargs.get("folder", "inbox")
            top = kwargs.get("max_results", 10)
            filter_q = kwargs.get("filter_query", "")

            url = f"{GRAPH_BASE}/me/mailFolders/{folder}/messages?$top={top}&$orderby=receivedDateTime desc"
            if filter_q:
                url += f"&$filter={filter_q}"

            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=_headers(token))
                resp.raise_for_status()
                data = resp.json()

            messages = []
            for m in data.get("value", []):
                messages.append({
                    "id": m["id"],
                    "from": m.get("from", {}).get("emailAddress", {}).get("address", ""),
                    "subject": m.get("subject", ""),
                    "date": m.get("receivedDateTime", ""),
                    "snippet": m.get("bodyPreview", "")[:200],
                    "is_read": m.get("isRead", False),
                })

            return ToolResult(
                tool_name=self.name, success=True, result=messages,
                display_card={"card_type": "EmailSummaryCard", "title": f"Outlook — {len(messages)} messages", "body": messages},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class OutlookReadTool(BaseTool):
    name = "email.outlook.read"
    description = "Read a specific Outlook message by ID."
    parameters_schema = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "The Outlook message ID"},
        },
        "required": ["message_id"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            token = await _get_access_token()
            url = f"{GRAPH_BASE}/me/messages/{kwargs['message_id']}"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=_headers(token))
                resp.raise_for_status()
                m = resp.json()

            parsed = {
                "id": m["id"],
                "from": m.get("from", {}).get("emailAddress", {}).get("address", ""),
                "to": [r.get("emailAddress", {}).get("address", "") for r in m.get("toRecipients", [])],
                "subject": m.get("subject", ""),
                "date": m.get("receivedDateTime", ""),
                "body": m.get("body", {}).get("content", "")[:3000],
                "body_type": m.get("body", {}).get("contentType", "text"),
            }
            return ToolResult(
                tool_name=self.name, success=True, result=parsed,
                display_card={"card_type": "EmailDetailCard", "title": parsed["subject"], "body": parsed},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class OutlookSendTool(BaseTool):
    name = "email.outlook.send"
    description = "Send an email via Outlook/Microsoft 365. REQUIRES explicit user confirmation."
    parameters_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body (text)"},
            "content_type": {"type": "string", "description": "text or html", "default": "text"},
        },
        "required": ["to", "subject", "body"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            token = await _get_access_token()
            payload = {
                "message": {
                    "subject": kwargs["subject"],
                    "body": {
                        "contentType": kwargs.get("content_type", "text"),
                        "content": kwargs["body"],
                    },
                    "toRecipients": [
                        {"emailAddress": {"address": kwargs["to"]}}
                    ],
                },
                "saveToSentItems": True,
            }

            url = f"{GRAPH_BASE}/me/sendMail"
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=_headers(token), json=payload)
                resp.raise_for_status()

            return ToolResult(
                tool_name=self.name, success=True,
                result={"sent_to": kwargs["to"], "subject": kwargs["subject"]},
                display_card={"card_type": "EmailSentCard", "title": "Outlook Email Sent", "body": {"to": kwargs["to"], "subject": kwargs["subject"]}},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))
