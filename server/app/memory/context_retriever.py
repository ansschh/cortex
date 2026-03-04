"""Context retriever — searches across all memory types and builds context for the LLM.

Queries the unified vector index, fetches full records from SQLite,
categorizes by type, and formats into structured prompt sections.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from server.app.memory.embeddings import EmbeddingService
from server.app.memory.vector_index import VectorIndex

logger = logging.getLogger(__name__)

MEMORY_TYPE_CONVERSATION = "conversation"
MEMORY_TYPE_MEMORY = "memory"
MEMORY_TYPE_FACT = "fact"
MEMORY_TYPE_PERSON = "person"
MEMORY_TYPE_EPISODE = "episode"
MEMORY_TYPE_PREFERENCE = "preference"

# Maximum characters for formatted context (leave room for LLM response)
MAX_CONTEXT_CHARS = 6000

# Minimum similarity score to include a result
MIN_SCORE_THRESHOLD = 0.25


@dataclass
class RetrievedContext:
    """A single piece of retrieved context with metadata."""
    text: str
    source_type: str
    source_id: int
    score: float
    created_at: str = ""


@dataclass
class ContextBundle:
    """The assembled context bundle ready for system prompt injection."""
    memories: list[RetrievedContext] = field(default_factory=list)
    facts: list[RetrievedContext] = field(default_factory=list)
    people: list[RetrievedContext] = field(default_factory=list)
    conversations: list[RetrievedContext] = field(default_factory=list)
    episodes: list[RetrievedContext] = field(default_factory=list)
    preferences: list[RetrievedContext] = field(default_factory=list)
    retrieval_time_ms: float = 0.0

    def format_for_prompt(self) -> str:
        """Format all retrieved context into a string for the system prompt.

        Caps output at MAX_CONTEXT_CHARS to avoid overwhelming the LLM.
        """
        sections = []

        if self.preferences:
            lines = [f"- {c.text}" for c in self.preferences]
            sections.append("USER PREFERENCES:\n" + "\n".join(lines))

        if self.people:
            lines = [f"- {c.text}" for c in self.people[:5]]
            sections.append("KNOWN PEOPLE:\n" + "\n".join(lines))

        if self.memories:
            lines = [f"- {c.text}" for c in self.memories[:5]]
            sections.append("RELEVANT MEMORIES:\n" + "\n".join(lines))

        if self.facts:
            lines = [f"- {c.text}" for c in self.facts[:5]]
            sections.append("RELEVANT FACTS:\n" + "\n".join(lines))

        if self.episodes:
            lines = [f"- {c.text}" for c in self.episodes[:3]]
            sections.append("PAST CONVERSATIONS:\n" + "\n".join(lines))

        if self.conversations:
            lines = [f"- {c.text}" for c in self.conversations[:5]]
            sections.append("EARLIER IN THIS TOPIC:\n" + "\n".join(lines))

        if not sections:
            return ""

        result = "\n\n".join(sections)
        if len(result) > MAX_CONTEXT_CHARS:
            result = result[:MAX_CONTEXT_CHARS] + "\n..."
        return result

    @property
    def total_items(self) -> int:
        return (len(self.memories) + len(self.facts) + len(self.people)
                + len(self.conversations) + len(self.episodes) + len(self.preferences))


class ContextRetriever:
    """Searches across all memory types using vector similarity and assembles context."""

    def __init__(self, embeddings: EmbeddingService, vector_index: VectorIndex, memory_store):
        self._embeddings = embeddings
        self._index = vector_index
        self._store = memory_store

    async def retrieve(self, query: str, top_k: int = 15) -> ContextBundle:
        """Retrieve relevant context for a user query.

        Searches the unified vector index, then fetches full records from SQLite.
        """
        start = time.monotonic()
        bundle = ContextBundle()

        if not self._embeddings.is_available or not self._index.is_available:
            bundle.retrieval_time_ms = (time.monotonic() - start) * 1000
            return bundle

        # 1. Embed the query
        query_vec = self._embeddings.embed_text(query)
        if query_vec is None:
            bundle.retrieval_time_ms = (time.monotonic() - start) * 1000
            return bundle

        # 2. Search the unified vector index
        results = self._index.search(query_vec, top_k=top_k)

        # 3. Fetch full records from the database
        for item_id, score in results:
            if score < MIN_SCORE_THRESHOLD:
                continue
            record = await self._store.get_vector_item(item_id)
            if record is None:
                continue

            ctx = RetrievedContext(
                text=record["text"],
                source_type=record["source_type"],
                source_id=record["source_id"],
                score=score,
                created_at=record.get("created_at", ""),
            )

            if record["source_type"] == MEMORY_TYPE_MEMORY:
                bundle.memories.append(ctx)
            elif record["source_type"] == MEMORY_TYPE_FACT:
                bundle.facts.append(ctx)
            elif record["source_type"] == MEMORY_TYPE_PERSON:
                bundle.people.append(ctx)
            elif record["source_type"] == MEMORY_TYPE_CONVERSATION:
                bundle.conversations.append(ctx)
            elif record["source_type"] == MEMORY_TYPE_EPISODE:
                bundle.episodes.append(ctx)
            elif record["source_type"] == MEMORY_TYPE_PREFERENCE:
                bundle.preferences.append(ctx)

        # 4. Always include all preferences (small, always relevant)
        all_prefs = await self._store.list_all_preferences()
        existing_pref_keys = {c.text.split(":")[0].strip() for c in bundle.preferences}
        for pref in all_prefs:
            if pref.key not in existing_pref_keys:
                bundle.preferences.append(RetrievedContext(
                    text=f"{pref.key}: {pref.value}",
                    source_type=MEMORY_TYPE_PREFERENCE,
                    source_id=0,
                    score=1.0,
                    created_at=pref.updated_at or "",
                ))

        bundle.retrieval_time_ms = (time.monotonic() - start) * 1000
        logger.info(
            f"Context retrieved in {bundle.retrieval_time_ms:.1f}ms: "
            f"{len(bundle.memories)}mem {len(bundle.facts)}fact "
            f"{len(bundle.people)}ppl {len(bundle.conversations)}conv "
            f"{len(bundle.episodes)}ep {len(bundle.preferences)}pref"
        )
        return bundle
