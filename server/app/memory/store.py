"""SQLite-backed memory store with structured tables, vector indexing, and conversation persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import numpy as np

from server.app.config import settings
from shared.schemas.memory import (
    Episode,
    Fact,
    MemoryEntry,
    MemorySearchResult,
    Person,
    Preference,
    Rule,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    aliases TEXT DEFAULT '[]',
    relationship TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'user',
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    provenance TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_text TEXT NOT NULL,
    scope TEXT DEFAULT 'global',
    priority INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary TEXT NOT NULL,
    expires_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    source TEXT DEFAULT 'user_command',
    embedding BLOB,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    intent TEXT DEFAULT 'conversation',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS vector_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


class MemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or settings.database_path
        self._db: Optional[aiosqlite.Connection] = None
        self._embeddings = None  # Set via set_embedding_service()
        self._vector_index = None  # Set via set_vector_index()

    def set_embedding_service(self, service) -> None:
        """Set the embedding service (deferred init after model loads)."""
        self._embeddings = service

    def set_vector_index(self, index) -> None:
        """Set the vector index (deferred init)."""
        self._vector_index = index

    async def initialize(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ------------------------------------------------------------------
    # People
    # ------------------------------------------------------------------

    async def add_person(self, person: Person) -> int:
        cur = await self._db.execute(
            "INSERT INTO people (name, aliases, relationship, notes) VALUES (?, ?, ?, ?)",
            (person.name, json.dumps(person.aliases), person.relationship, person.notes),
        )
        await self._db.commit()
        pid = cur.lastrowid
        text = f"{person.name}"
        if person.relationship:
            text += f" ({person.relationship})"
        if person.notes:
            text += f": {person.notes}"
        if person.aliases:
            text += f" (aka {', '.join(person.aliases)})"
        await self._index_item("person", pid, text)
        return pid

    async def find_person(self, name: str) -> Optional[Person]:
        cur = await self._db.execute(
            "SELECT * FROM people WHERE name LIKE ? OR aliases LIKE ?",
            (f"%{name}%", f"%{name}%"),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return Person(
            id=row["id"],
            name=row["name"],
            aliases=json.loads(row["aliases"]),
            relationship=row["relationship"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def list_people(self) -> list[Person]:
        cur = await self._db.execute("SELECT * FROM people ORDER BY name")
        rows = await cur.fetchall()
        return [
            Person(
                id=r["id"], name=r["name"], aliases=json.loads(r["aliases"]),
                relationship=r["relationship"], notes=r["notes"],
                created_at=r["created_at"], updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    async def set_preference(self, pref: Preference) -> None:
        await self._db.execute(
            """INSERT INTO preferences (key, value, confidence, source, updated_at)
               VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET value=?, confidence=?, source=?, updated_at=datetime('now')""",
            (pref.key, pref.value, pref.confidence, pref.source,
             pref.value, pref.confidence, pref.source),
        )
        await self._db.commit()
        await self._index_item("preference", 0, f"{pref.key}: {pref.value}")

    async def get_preference(self, key: str) -> Optional[Preference]:
        cur = await self._db.execute("SELECT * FROM preferences WHERE key = ?", (key,))
        row = await cur.fetchone()
        if not row:
            return None
        return Preference(
            key=row["key"], value=row["value"], confidence=row["confidence"],
            source=row["source"], updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    async def add_fact(self, fact: Fact) -> int:
        cur = await self._db.execute(
            "INSERT INTO facts (subject, predicate, object, provenance) VALUES (?, ?, ?, ?)",
            (fact.subject, fact.predicate, fact.object, fact.provenance),
        )
        await self._db.commit()
        fid = cur.lastrowid
        await self._index_item("fact", fid, f"{fact.subject} {fact.predicate} {fact.object}")
        return fid

    async def search_facts(self, query: str) -> list[Fact]:
        cur = await self._db.execute(
            "SELECT * FROM facts WHERE subject LIKE ? OR object LIKE ? OR predicate LIKE ?",
            (f"%{query}%", f"%{query}%", f"%{query}%"),
        )
        rows = await cur.fetchall()
        return [
            Fact(id=r["id"], subject=r["subject"], predicate=r["predicate"],
                 object=r["object"], provenance=r["provenance"], created_at=r["created_at"])
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    async def add_rule(self, rule: Rule) -> int:
        cur = await self._db.execute(
            "INSERT INTO rules (rule_text, scope, priority) VALUES (?, ?, ?)",
            (rule.rule_text, rule.scope, rule.priority),
        )
        await self._db.commit()
        return cur.lastrowid

    async def list_rules(self) -> list[Rule]:
        cur = await self._db.execute("SELECT * FROM rules ORDER BY priority DESC")
        rows = await cur.fetchall()
        return [
            Rule(id=r["id"], rule_text=r["rule_text"], scope=r["scope"],
                 priority=r["priority"], created_at=r["created_at"])
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Freeform memories (Layer B + C)
    # ------------------------------------------------------------------

    async def add_memory(self, entry: MemoryEntry) -> int:
        embedding_blob = None
        if entry.embedding:
            embedding_blob = np.array(entry.embedding, dtype=np.float32).tobytes()
        cur = await self._db.execute(
            "INSERT INTO memories (text, tags, source, embedding) VALUES (?, ?, ?, ?)",
            (entry.text, json.dumps(entry.tags), entry.source, embedding_blob),
        )
        await self._db.commit()
        mem_id = cur.lastrowid
        await self._index_item("memory", mem_id, entry.text)
        return mem_id

    async def search_memories(self, query: str, limit: int = 10) -> list[MemorySearchResult]:
        """Simple text search — swap to vector similarity when embeddings are added."""
        cur = await self._db.execute(
            "SELECT * FROM memories WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cur.fetchall()
        return [
            MemorySearchResult(
                entry=MemoryEntry(
                    id=r["id"], text=r["text"], tags=json.loads(r["tags"]),
                    source=r["source"], created_at=r["created_at"],
                ),
                score=1.0,
            )
            for r in rows
        ]

    async def list_memories(self, limit: int = 50) -> list[MemoryEntry]:
        cur = await self._db.execute(
            "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [
            MemoryEntry(
                id=r["id"], text=r["text"], tags=json.loads(r["tags"]),
                source=r["source"], created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    async def log_audit(self, event_type: str, payload: dict) -> None:
        await self._db.execute(
            "INSERT INTO audit_log (event_type, payload) VALUES (?, ?)",
            (event_type, json.dumps(payload, default=str)),
        )
        await self._db.commit()

    async def get_audit_log(self, limit: int = 100) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [
            {"id": r["id"], "event_type": r["event_type"],
             "payload": json.loads(r["payload"]), "created_at": r["created_at"]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Vector indexing (unified across all memory types)
    # ------------------------------------------------------------------

    async def _index_item(self, source_type: str, source_id: int, text: str) -> Optional[int]:
        """Insert into vector_items table and add to the vector index."""
        cur = await self._db.execute(
            "INSERT INTO vector_items (source_type, source_id, text) VALUES (?, ?, ?)",
            (source_type, source_id, text),
        )
        await self._db.commit()
        item_id = cur.lastrowid

        # Generate embedding and add to vector index
        if self._embeddings and self._vector_index and self._embeddings.is_available:
            vec = self._embeddings.embed_text(text)
            if vec is not None:
                self._vector_index.add(vec, item_id)

        return item_id

    async def get_vector_item(self, item_id: int) -> Optional[dict]:
        """Look up a vector_items record by ID."""
        cur = await self._db.execute("SELECT * FROM vector_items WHERE id = ?", (item_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return {
            "id": row["id"], "source_type": row["source_type"],
            "source_id": row["source_id"], "text": row["text"],
            "created_at": row["created_at"],
        }

    async def list_all_preferences(self) -> list[Preference]:
        """Return all stored preferences."""
        cur = await self._db.execute("SELECT * FROM preferences")
        rows = await cur.fetchall()
        return [
            Preference(key=r["key"], value=r["value"], confidence=r["confidence"],
                       source=r["source"], updated_at=r["updated_at"])
            for r in rows
        ]

    async def rebuild_vector_index(self, embeddings, vector_index) -> int:
        """Rebuild the entire vector index from the vector_items table.

        Called once at startup if the index file is missing/empty but the DB has data.
        """
        cur = await self._db.execute("SELECT id, text FROM vector_items ORDER BY id")
        rows = await cur.fetchall()
        if not rows:
            return 0

        texts = [r["text"] for r in rows]
        ids = np.array([r["id"] for r in rows], dtype=np.int64)

        vectors = embeddings.embed_batch(texts)
        if vectors is None:
            return 0

        vector_index.add_batch(vectors, ids)
        vector_index.save()
        return len(rows)

    # ------------------------------------------------------------------
    # Conversation turns (persistent conversation history)
    # ------------------------------------------------------------------

    async def add_conversation_turn(
        self, session_id: str, role: str, content: str, intent: str = "conversation",
    ) -> int:
        """Store a conversation turn and index it for vector search."""
        cur = await self._db.execute(
            "INSERT INTO conversation_turns (session_id, role, content, intent) VALUES (?, ?, ?, ?)",
            (session_id, role, content, intent),
        )
        await self._db.commit()
        turn_id = cur.lastrowid

        # Index in vector search (prefix with role for better retrieval)
        index_text = f"{role}: {content}"
        await self._index_item("conversation", turn_id, index_text)
        return turn_id

    async def get_recent_turns(self, limit: int = 50) -> list[dict]:
        """Get most recent conversation turns across all sessions."""
        cur = await self._db.execute(
            "SELECT * FROM conversation_turns ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [
            {"id": r["id"], "session_id": r["session_id"], "role": r["role"],
             "content": r["content"], "intent": r["intent"], "created_at": r["created_at"]}
            for r in rows
        ]

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        """List conversation sessions with summary stats."""
        cur = await self._db.execute(
            """SELECT session_id,
                      MIN(created_at) as started_at,
                      MAX(created_at) as ended_at,
                      COUNT(*) as turn_count,
                      GROUP_CONCAT(DISTINCT intent) as intents
               FROM conversation_turns
               GROUP BY session_id
               ORDER BY MAX(created_at) DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cur.fetchall()
        return [
            {
                "session_id": r["session_id"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "turn_count": r["turn_count"],
                "intents": r["intents"].split(",") if r["intents"] else [],
            }
            for r in rows
        ]

    async def get_session_turns(self, session_id: str) -> list[dict]:
        """Get all turns for a specific session, ordered chronologically."""
        cur = await self._db.execute(
            "SELECT * FROM conversation_turns WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = await cur.fetchall()
        return [
            {"id": r["id"], "session_id": r["session_id"], "role": r["role"],
             "content": r["content"], "intent": r["intent"], "created_at": r["created_at"]}
            for r in rows
        ]

    async def search_conversations(self, query: str, limit: int = 20) -> list[dict]:
        """Text search across conversation turns."""
        cur = await self._db.execute(
            """SELECT * FROM conversation_turns
               WHERE content LIKE ?
               ORDER BY id DESC LIMIT ?""",
            (f"%{query}%", limit),
        )
        rows = await cur.fetchall()
        return [
            {"id": r["id"], "session_id": r["session_id"], "role": r["role"],
             "content": r["content"], "intent": r["intent"], "created_at": r["created_at"]}
            for r in rows
        ]

    async def list_all_facts(self, limit: int = 100) -> list[Fact]:
        """Return all stored facts."""
        cur = await self._db.execute(
            "SELECT * FROM facts ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [
            Fact(id=r["id"], subject=r["subject"], predicate=r["predicate"],
                 object=r["object"], provenance=r["provenance"], created_at=r["created_at"])
            for r in rows
        ]

    async def list_episodes(self, limit: int = 50) -> list[Episode]:
        """Return all stored episodes (conversation summaries)."""
        cur = await self._db.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [
            Episode(id=r["id"], summary=r["summary"], expires_at=r["expires_at"],
                    created_at=r["created_at"])
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Episodes (conversation summaries for long-term memory)
    # ------------------------------------------------------------------

    async def add_episode(self, episode: Episode) -> int:
        """Store a conversation summary episode and index it."""
        cur = await self._db.execute(
            "INSERT INTO episodes (summary, expires_at) VALUES (?, ?)",
            (episode.summary, episode.expires_at),
        )
        await self._db.commit()
        eid = cur.lastrowid
        await self._index_item("episode", eid, episode.summary)
        return eid

    async def get_unsummarized_sessions(self, older_than_hours: int = 24, limit: int = 5) -> list[dict]:
        """Find conversation sessions that haven't been summarized into episodes yet."""
        cur = await self._db.execute(
            """SELECT session_id, GROUP_CONCAT(role || ': ' || content, '\n') as transcript,
                      COUNT(*) as turn_count
               FROM conversation_turns
               WHERE created_at < datetime('now', ? || ' hours')
               GROUP BY session_id
               HAVING turn_count > 4
               ORDER BY MIN(created_at) ASC
               LIMIT ?""",
            (f"-{older_than_hours}", limit),
        )
        rows = await cur.fetchall()
        return [
            {"session_id": r["session_id"], "transcript": r["transcript"],
             "turn_count": r["turn_count"]}
            for r in rows
        ]
