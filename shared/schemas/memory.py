"""Memory schemas — defines structured memory types."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    id: Optional[int] = None
    name: str
    aliases: list[str] = Field(default_factory=list)
    relationship: str = ""
    notes: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Preference(BaseModel):
    key: str
    value: str
    confidence: float = 1.0
    source: str = "user"
    updated_at: Optional[str] = None


class Fact(BaseModel):
    id: Optional[int] = None
    subject: str
    predicate: str
    object: str
    provenance: str = ""
    created_at: Optional[str] = None


class Rule(BaseModel):
    id: Optional[int] = None
    rule_text: str
    scope: str = "global"
    priority: int = 0
    created_at: Optional[str] = None


class Episode(BaseModel):
    id: Optional[int] = None
    summary: str
    expires_at: Optional[str] = None
    created_at: Optional[str] = None


class MemoryEntry(BaseModel):
    """A freeform memory entry with optional embedding for semantic search."""
    id: Optional[int] = None
    text: str
    tags: list[str] = Field(default_factory=list)
    source: str = "user_command"
    embedding: Optional[list[float]] = None
    created_at: Optional[str] = None


class MemorySearchResult(BaseModel):
    entry: MemoryEntry
    score: float = 0.0
