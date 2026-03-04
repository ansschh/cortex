"""Tool call schemas — defines the contract for tool execution."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from shared.schemas.events import SensitivityLevel


class ToolDefinition(BaseModel):
    """Schema describing a tool the LLM can call."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    sensitivity: SensitivityLevel = SensitivityLevel.LOW


class ToolCall(BaseModel):
    """A single tool invocation produced by the LLM."""
    call_id: str = ""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The result of executing a tool."""
    call_id: str = ""
    tool_name: str
    success: bool = True
    result: Any = None
    error: Optional[str] = None
    display_card: Optional[dict[str, Any]] = None


class PendingAction(BaseModel):
    """An action waiting for explicit user confirmation."""
    action_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    preview_text: str = ""
    requires_speaker_verification: bool = True
    created_at: float = 0.0
