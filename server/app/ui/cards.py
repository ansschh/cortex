"""Card builders — helpers that produce UICard dicts from tool results."""

from __future__ import annotations

from typing import Any

from shared.schemas.events import UICard


def email_summary_card(title: str, messages: list[dict]) -> UICard:
    rows = []
    for m in messages:
        rows.append(f"• **{m.get('from', '?')}** — {m.get('subject', '(no subject)')}")
    return UICard(
        card_type="EmailSummaryCard",
        title=title,
        body={"messages": messages, "display_text": "\n".join(rows)},
        priority=5,
    )


def email_draft_card(draft: dict) -> UICard:
    return UICard(
        card_type="EmailDraftCard",
        card_id=f"draft-{draft.get('to', '')}",
        title=f"Draft → {draft.get('to', '?')}",
        body=draft,
        actions=[{"label": "CONFIRM SEND", "action": "confirm_send"}],
        priority=10,
    )


def slack_draft_card(channel: str, text: str) -> UICard:
    return UICard(
        card_type="SlackDraftCard",
        card_id=f"slack-draft-{channel}",
        title=f"Slack Draft → {channel}",
        body={"channel": channel, "text": text},
        actions=[{"label": "CONFIRM POST", "action": "confirm_post"}],
        priority=10,
    )


def memory_saved_card(text: str, mem_id: int = 0) -> UICard:
    return UICard(
        card_type="MemorySavedCard",
        card_id=f"memory-{mem_id}",
        title="Memory Saved",
        body=text,
        priority=3,
    )


def device_status_card(devices: list[dict]) -> UICard:
    return UICard(
        card_type="DeviceStatusCard",
        title="Home Devices",
        body=devices,
        priority=2,
    )


def toast_card(message: str, level: str = "info") -> UICard:
    return UICard(
        card_type="ToastCard",
        title=level.upper(),
        body=message,
        priority=1,
    )


def assistant_response_card(text: str) -> UICard:
    return UICard(
        card_type="AssistantResponseCard",
        title="NOVA",
        body=text,
        priority=4,
    )
