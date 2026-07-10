"""MyelinProvider — dependency-light Mem0-compatible API backed by Myelin.

Provides an add/search/update/delete/get_all surface analogous to Mem0's
``Memory`` class, implemented on top of Myelin's episodic/semantic/procedural
memory stores. All operations are local (SQLite) and require no network.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from myelin.core.database import Database
from myelin.core.models import ActionType, NodeType, SemanticNode, SourceType
from myelin.memory.embedding import EmbeddingProvider, NoOpEmbedding
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory
from myelin.tools.handlers import ToolHandlers

# ──────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────


MemoryEntry = dict[str, Any]
"""Shape returned by provider methods. Always includes at least ``id``."""


@dataclass
class ProviderConfig:
    """Lightweight configuration for :class:`MyelinProvider`.

    Parameters
    ----------
    agent_id:
        Default agent identifier used when none is passed to a method call.
    session_id:
        Default session identifier used when none is passed.
    domain:
        Default domain for memories.
    auto_sleep:
        If True, run ``trigger_sleep`` after every ``add`` call to promote
        procedures in real-time. Off by default — batch-oriented users
        should call ``trigger_sleep`` manually.
    """

    agent_id: str = "default-agent"
    session_id: str = "default-session"
    domain: str | None = None
    auto_sleep: bool = False


# ──────────────────────────────────────────────────────────────────────
# Async helper — works in both sync and pytest-asyncio contexts
# ──────────────────────────────────────────────────────────────────────


def _run_async(coro: asyncio.coroutine) -> Any:
    """Run a coroutine, handling both sync and already-running-loop contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)

    # A loop is already running (e.g. pytest-asyncio).  Create a new task
    # and run until it completes.
    import concurrent.futures

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=60)
    except concurrent.futures.TimeoutError:
        raise RuntimeError("Async operation timed out (60s)")


# ──────────────────────────────────────────────────────────────────────
# MyelinProvider
# ──────────────────────────────────────────────────────────────────────


class MyelinProvider:
    """Mem0-compatible memory provider powered by Myelin's local learning layer.

    Every memory operation delegates to Myelin's SQLite-backed stores and
    optionally triggers consolidation (sleep) cycles.  No external
    dependencies beyond Myelin itself.
    """

    def __init__(
        self,
        db: Database,
        embedder: EmbeddingProvider | None = None,
        config: ProviderConfig | dict[str, Any] | None = None,
    ) -> None:
        self.db = db

        if isinstance(config, dict):
            self.config = ProviderConfig(**config)
        elif config is None:
            self.config = ProviderConfig()
        else:
            self.config = config

        self.episodic = EpisodicMemory(db)
        self.semantic = SemanticMemory(db)
        self.procedural = ProceduralMemory(db)
        self.embedder = embedder or NoOpEmbedding()

        self._handlers = ToolHandlers(
            episodic=self.episodic,
            semantic=self.semantic,
            procedural=self.procedural,
            embedder=self.embedder,
        )

    # ── add ──────────────────────────────────────────────────────────

    def add(
        self,
        messages: str | list[dict[str, str]] | dict[str, Any],
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        """Store one or more memory entries.

        Analogous to ``mem0.add()``. Accepts a plain text string, a
        message dict (``{"role": …, "content": …}``), or a list of
        message dicts.  Each entry is recorded as a Myelin episode plus a
        semantic node.

        Parameters
        ----------
        messages:
            Content to remember.
        user_id:
            Override the default agent_id (maps to user context).
        agent_id:
            Agent identifier (falls back to ``config.agent_id``).
        metadata:
            Extra fields merged into the episode (e.g. domain, tags).
        session_id:
            Override the default session identifier.
        **kwargs:
            Ignored — kept for Mem0 API compatibility.

        Returns
        -------
        list[MemoryEntry]
            List of created memory entries with at least ``id``.
        """
        aid = agent_id or self.config.agent_id
        sid = session_id or self.config.session_id
        meta = metadata or {}

        # Normalise input to a list of content strings
        texts = self._normalise_messages(messages)
        entries: list[MemoryEntry] = []

        for text in texts:
            domain = meta.get("domain") or self.config.domain

            entry = _run_async(
                self._handlers.observe(
                    agent_id=aid,
                    session_id=sid,
                    action=meta.get("action", "add"),
                    action_type=meta.get("action_type", "user_input"),
                    content_text=text,
                    domain=domain,
                    tags=meta.get("tags"),
                    success=True,
                )
            )

            mem_id = entry["episode_id"]

            # Also store a semantic node so search picks it up via
            # semantic memory in addition to episodic FTS.
            node = SemanticNode(
                node_type=NodeType.FACT,
                content=text,
                source_type=SourceType.OBSERVATION,
                source_ids=[mem_id],
                domain=domain,
                confidence=0.5,
            )
            node_id = self.semantic.store(node)

            entries.append(
                {
                    "id": mem_id,
                    "semantic_id": node_id,
                    "content": text,
                    "agent_id": aid,
                    "session_id": sid,
                    "metadata": meta,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )

        if self.config.auto_sleep:
            _run_async(self._handlers.trigger_sleep(agent_id=aid))

        return entries

    # ── search ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        """Search memories by text query.

        Uses Myelin's multi-signal retriever (FTS5 + entity + temporal
        + activation) for ranked results.

        Parameters
        ----------
        query:
            Search text.
        agent_id:
            Agent identifier (falls back to ``config.agent_id``).
        limit:
            Maximum results.
        **kwargs:
            Ignored — Mem0 API compatibility.

        Returns
        -------
        list[MemoryEntry]
            Ranked memory entries.
        """
        aid = agent_id or self.config.agent_id

        result = _run_async(
            self._handlers.query(
                query=query,
                limit=limit,
                agent_ids=[aid],
            )
        )

        return [
            {
                "id": r.get("id", ""),
                "content": r.get("content", ""),
                "score": r.get("composite_score", 0),
                "scores": r.get("scores", {}),
                "source_type": r.get("source_type", "episode"),
                "agent_id": r.get("source_agent", aid),
            }
            for r in result.get("results", [])
        ]

    # ── update ───────────────────────────────────────────────────────

    def update(
        self,
        memory_id: str,
        data: str | dict[str, Any],
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> MemoryEntry:
        """Update the content of an existing memory entry.

        Myelin is append-only by design, so this creates a new episode
        referencing the original and stores the update as a semantic
        node with the old ID superseded.

        Parameters
        ----------
        memory_id:
            The episode ID to update.
        data:
            New content string or dict with ``content`` key.
        agent_id:
            Agent identifier.
        **kwargs:
            Ignored.

        Returns
        -------
        MemoryEntry
            The updated/created entry.
        """
        aid = agent_id or self.config.agent_id

        content = data if isinstance(data, str) else data.get("content", str(data))

        # Record the update as a new episode that references the old one
        entry = _run_async(
            self._handlers.observe(
                agent_id=aid,
                session_id=self.config.session_id,
                action="update",
                action_type="user_input",
                content_text=content,
                domain=self.config.domain,
                tags=["update", f"supersedes:{memory_id}"],
                success=True,
            )
        )

        new_id = entry["episode_id"]
        return {
            "id": new_id,
            "previous_id": memory_id,
            "content": content,
            "agent_id": aid,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # ── delete ───────────────────────────────────────────────────────

    def delete(
        self,
        memory_id: str,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Mark a memory entry as removed (soft-delete via archival).

        Myelin does not support hard deletes of episodes.  This creates a
        tombstone semantic node so the entry is excluded from active
        results on the provider level.  The underlying episode remains in
        the database.

        Parameters
        ----------
        memory_id:
            Episode ID to delete.
        **kwargs:
            Ignored.

        Returns
        -------
        bool
            True if the memory was found and marked.
        """
        episode = self.episodic.get(memory_id)
        if not episode:
            return False

        # Record a deletion event as a semantic node (episodes are
        # append-only — we mark via metadata / tombstone).
        node = SemanticNode(
            node_type=NodeType.REFLECTION,
            content=f"DELETED:{memory_id}",
            source_type=SourceType.OBSERVATION,
            source_ids=[memory_id],
            domain=episode.get("domain"),
            confidence=0.0,
        )
        self.semantic.store(node)
        return True

    # ── get_all ──────────────────────────────────────────────────────

    def get_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        """Retrieve all memory entries for an agent.

        Returns episodes in reverse chronological order, excluding entries
        that have been deleted (marked with a tombstone).

        Parameters
        ----------
        agent_id:
            Agent identifier (falls back to ``config.agent_id``).
        limit:
            Maximum entries.
        **kwargs:
            Ignored.

        Returns
        -------
        list[MemoryEntry]
            List of memory entries.
        """
        aid = agent_id or self.config.agent_id

        # Fetch episodes + check tombstones
        episodes = self.episodic.get_recent(limit=limit * 2)
        deleted_ids = self._get_deleted_ids()

        result: list[MemoryEntry] = []
        for ep in episodes:
            if ep["agent_id"] != aid:
                continue
            if ep["id"] in deleted_ids:
                continue
            result.append(
                {
                    "id": ep["id"],
                    "content": ep.get("content_text", ""),
                    "action": ep.get("action", ""),
                    "agent_id": ep.get("agent_id", aid),
                    "session_id": ep.get("session_id", ""),
                    "domain": ep.get("domain"),
                    "success": ep.get("success", True),
                    "created_at": ep.get("created_at", ""),
                }
            )
            if len(result) >= limit:
                break

        return result

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalise_messages(
        messages: str | list[dict[str, str]] | dict[str, Any],
    ) -> list[str]:
        """Convert various input shapes to a list of content strings."""
        if isinstance(messages, str):
            return [messages]
        if isinstance(messages, dict):
            return [messages.get("content", str(messages))]
        if isinstance(messages, list):
            return [
                m.get("content", str(m)) if isinstance(m, dict) else str(m)
                for m in messages
            ]
        return [str(messages)]

    def _get_deleted_ids(self) -> set[str]:
        """Return the set of episode IDs that have deletion tombstones."""
        result = _run_async(
            self._handlers.recall(
                query="DELETED:",
                memory_types=["semantic"],
                limit=1000,
            )
        )
        deleted: set[str] = set()
        for node in result.get("results", {}).get("semantic", []):
            content = node.get("content", "")
            if content.startswith("DELETED:"):
                deleted.add(content[len("DELETED:"):])
        return deleted
