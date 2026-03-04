"""Confirmation manager — tracks pending actions awaiting user approval."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from shared.schemas.tool_calls import PendingAction, ToolResult


class ConfirmationManager:
    """Manages pending actions that need explicit user confirmation before execution."""

    EXPIRY_SECONDS = 120  # pending actions expire after 2 minutes

    def __init__(self):
        self._pending: dict[str, PendingAction] = {}

    def create_pending(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        preview_text: str = "",
        requires_speaker_verification: bool = True,
    ) -> PendingAction:
        action_id = str(uuid.uuid4())[:8]
        action = PendingAction(
            action_id=action_id,
            tool_name=tool_name,
            arguments=arguments,
            preview_text=preview_text,
            requires_speaker_verification=requires_speaker_verification,
            created_at=time.time(),
        )
        self._pending[action_id] = action
        return action

    def get_pending(self, action_id: Optional[str] = None) -> Optional[PendingAction]:
        if action_id:
            return self._pending.get(action_id)
        # Return the most recent pending action
        if not self._pending:
            return None
        return max(self._pending.values(), key=lambda a: a.created_at)

    def get_all_pending(self) -> list[PendingAction]:
        self._cleanup_expired()
        return list(self._pending.values())

    def resolve(self, action_id: str, confirmed: bool) -> Optional[PendingAction]:
        action = self._pending.pop(action_id, None)
        return action if confirmed else None

    def resolve_latest(self, confirmed: bool) -> Optional[PendingAction]:
        action = self.get_pending()
        if action:
            return self.resolve(action.action_id, confirmed)
        return None

    def clear_all(self) -> int:
        count = len(self._pending)
        self._pending.clear()
        return count

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            aid for aid, a in self._pending.items()
            if now - a.created_at > self.EXPIRY_SECONDS
        ]
        for aid in expired:
            del self._pending[aid]
