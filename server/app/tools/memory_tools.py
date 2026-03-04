"""Memory tools — add, search, list memories and people."""

from __future__ import annotations

from typing import Any

from server.app.tools.base import BaseTool
from shared.schemas.events import SensitivityLevel
from shared.schemas.memory import Fact, MemoryEntry, Person, Preference
from shared.schemas.tool_calls import ToolResult


class MemoryAddTool(BaseTool):
    name = "memory.add"
    description = "Save a new memory or fact the user wants to remember."
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The memory text to store"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization",
            },
        },
        "required": ["text"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, memory_store):
        self._store = memory_store

    async def execute(self, **kwargs: Any) -> ToolResult:
        text = kwargs.get("text", "")
        tags = kwargs.get("tags", [])
        entry = MemoryEntry(text=text, tags=tags, source="user_command")
        mem_id = await self._store.add_memory(entry)
        return ToolResult(
            tool_name=self.name,
            success=True,
            result={"id": mem_id, "text": text},
            display_card={
                "card_type": "MemorySavedCard",
                "title": "Memory Saved",
                "body": text,
            },
        )


class MemorySearchTool(BaseTool):
    name = "memory.search"
    description = "Search through saved memories and facts."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results", "default": 10},
        },
        "required": ["query"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, memory_store):
        self._store = memory_store

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)
        results = await self._store.search_memories(query, limit)
        facts = await self._store.search_facts(query)
        person = await self._store.find_person(query)

        combined = []
        if person:
            combined.append({"type": "person", "data": person.model_dump()})
        for f in facts:
            combined.append({"type": "fact", "data": f.model_dump()})
        for r in results:
            combined.append({"type": "memory", "data": r.entry.model_dump(), "score": r.score})

        return ToolResult(
            tool_name=self.name,
            success=True,
            result=combined,
        )


class PersonAddTool(BaseTool):
    name = "memory.add_person"
    description = "Add a person to the known people database."
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's name"},
            "relationship": {"type": "string", "description": "Relationship to user"},
            "notes": {"type": "string", "description": "Any notes about this person"},
            "aliases": {
                "type": "array", "items": {"type": "string"},
                "description": "Alternative names or nicknames",
            },
        },
        "required": ["name"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, memory_store):
        self._store = memory_store

    async def execute(self, **kwargs: Any) -> ToolResult:
        person = Person(
            name=kwargs.get("name", ""),
            aliases=kwargs.get("aliases", []),
            relationship=kwargs.get("relationship", ""),
            notes=kwargs.get("notes", ""),
        )
        pid = await self._store.add_person(person)
        return ToolResult(
            tool_name=self.name, success=True,
            result={"id": pid, "name": person.name},
        )


class FactAddTool(BaseTool):
    name = "memory.add_fact"
    description = "Store a structured fact (subject-predicate-object triple)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
        },
        "required": ["subject", "predicate", "object"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, memory_store):
        self._store = memory_store

    async def execute(self, **kwargs: Any) -> ToolResult:
        fact = Fact(
            subject=kwargs.get("subject", ""),
            predicate=kwargs.get("predicate", ""),
            object=kwargs.get("object", ""),
            provenance="assistant",
        )
        fid = await self._store.add_fact(fact)
        return ToolResult(
            tool_name=self.name, success=True,
            result={"id": fid, "subject": fact.subject, "predicate": fact.predicate, "object": fact.object},
        )


class PreferenceSetTool(BaseTool):
    name = "memory.set_preference"
    description = "Store or update a user preference."
    parameters_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Preference key"},
            "value": {"type": "string", "description": "Preference value"},
        },
        "required": ["key", "value"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, memory_store):
        self._store = memory_store

    async def execute(self, **kwargs: Any) -> ToolResult:
        pref = Preference(key=kwargs["key"], value=kwargs["value"])
        await self._store.set_preference(pref)
        return ToolResult(
            tool_name=self.name, success=True,
            result={"key": pref.key, "value": pref.value},
        )
