"""MyelinProvider adapter with reversible Mem0 dual-write/shadow-read mode.

Myelin is authoritative by default.  Mem0 integration is opt-in,
non-blocking, and safe when unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .protocol import MemoryProvider, SearchResult, compare_search_results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MyelinProvider — wraps Myelin's own MCP tools behind the MemoryProvider ABC
# ---------------------------------------------------------------------------


class MyelinProvider(MemoryProvider):
    """Pure-Myelin memory provider.

    Translates the 4-method ``MemoryProvider`` surface into Myelin MCP
    tool calls using an inline-myelin handler pattern (or, in production,
    an MCP client).  **Myelin is authoritative.**
    """

    def __init__(
        self,
        *,
        tool_handlers: Any = None,
        agent_id: str = "hermes",
        user_id: str = "default",
        timeout_ms: int = 5000,
    ) -> None:
        self._handlers = tool_handlers
        self._agent_id = agent_id
        self._user_id = user_id
        self._timeout_s = timeout_ms / 1000

    # ── MemoryProvider ABC ─────────────────────────────────────

    async def add(
        self,
        content: str,
        *,
        agent_id: str = "hermes",
        user_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a fact via myelin_memorize (profile_facts upsert)."""
        if self._handlers is None:
            return {"result": "Myelin not available.", "event_id": ""}
        result = await self._handlers.memorize(
            agent_id=agent_id,
            fact=content,
            category=(metadata or {}).get("category", "fact"),
            confidence=float((metadata or {}).get("confidence", 0.6)),
        )
        return {
            "result": "Fact stored.",
            "event_id": result.get("fact_id", ""),
        }

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        agent_id: str = "hermes",
        user_id: str = "default",
    ) -> list[SearchResult]:
        """Search via myelin_context, formatted into SearchResults."""
        if self._handlers is None:
            return []
        ctx = await self._handlers.context(
            query=query,
            agent_id=agent_id,
            max_memories=top_k,
            max_procedures=0,
        )
        memories = ctx.get("relevant_memories", ctx.get("results", {}).get("episodes", []))
        results: list[SearchResult] = []
        for m in memories[:top_k]:
            results.append(
                SearchResult(
                    id=m.get("id", ""),
                    memory=m.get("content_text", m.get("content", m.get("name", ""))),
                    score=m.get("relevance", m.get("composite_score", m.get("score", 0.5))),
                    metadata={
                        k: m[k]
                        for k in ("source_type", "domain", "action", "source_agent")
                        if k in m
                    },
                )
            )
        # Supplement with profile facts
        try:
            profile = await self._handlers.profile(agent_id=agent_id)
            for f in profile.get("static_facts", []):
                results.append(
                    SearchResult(
                        id=f.get("id", ""),
                        memory=f.get("fact", ""),
                        score=f.get("confidence", 0.5),
                        metadata={"source_type": "profile_fact", "category": f.get("category", "")},
                    )
                )
        except Exception:
            pass

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def update(
        self,
        memory_id: str,
        text: str,
        *,
        agent_id: str = "hermes",
    ) -> dict[str, Any]:
        """Update via myelin_update."""
        if self._handlers is None:
            return {"memory_id": memory_id}
        await self._handlers.update_memory(
            memory_id=memory_id,
            new_text=text,
        )
        return {"memory_id": memory_id}

    async def delete(
        self,
        memory_id: str,
        *,
        agent_id: str = "hermes",
    ) -> dict[str, Any]:
        """Delete via myelin_forget."""
        if self._handlers is None:
            return {"memory_id": memory_id}
        await self._handlers.forget(fact_id=memory_id)
        return {"memory_id": memory_id}

    async def system_prompt_block(
        self,
        *,
        agent_id: str = "hermes",
        user_id: str = "default",
    ) -> str:
        """Return a system-prompt injection block."""
        if self._handlers is None:
            return "# Myelin Memory\nNot available."
        try:
            profile = await self._handlers.profile(agent_id=agent_id)
        except Exception:
            return "# Myelin Memory\nActive."
        fact_count = profile.get("fact_count", 0)
        cats = profile.get("category_breakdown", {})
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items())) or "none"
        return (
            f"# Myelin Memory\n"
            f"Active. {fact_count} facts across categories: {cat_str}.\n"
            "Call myelin_context for rich context before answering questions.\n"
        )

    async def health(self) -> dict[str, Any]:
        """Return health status."""
        ok = self._handlers is not None
        return {"ok": ok, "detail": "myelin handlers available" if ok else "myelin not available"}


# ---------------------------------------------------------------------------
# Mem0DualWriteAdapter — wraps a MyelinProvider with optional Mem0 dual-write
# ---------------------------------------------------------------------------


class Mem0DualWriteAdapter(MemoryProvider):
    """Opt-in, reversible adapter that wraps ``MyelinProvider`` with Mem0.

    **Myelin is authoritative.**  By default Mem0 integration is OFF.
    Call ``enable_mem0(provider)`` to activate dual-write.

    Architecture
    ------------
    - *Default mode* (no Mem0): passes everything through to the inner
      MyelinProvider unchanged.  Zero overhead.
    - *Dual-write mode* (``enable_mem0``): every ``add()`` writes to both
      Myelin and Mem0.  ``search()`` returns Myelin results but *also*
      fires a comparison log if ``shadow_read=True``.
    - *Safe fallback*: any Mem0 failure is caught, logged, and ignored.
      The calling agent never sees Mem0 errors.
    """

    def __init__(self, myelin_provider: MyelinProvider) -> None:
        self._myelin = myelin_provider
        self._mem0: MemoryProvider | None = None
        self._shadow_read: bool = False
        self._dual_write: bool = False
        self._comparison_log: list[dict[str, Any]] = []

    # ── Configuration ─────────────────────────────────────────

    def enable_mem0(self, mem0_provider: MemoryProvider, *, shadow_read: bool = False) -> None:
        """Activate Mem0 dual-write mode.

        Parameters
        ----------
        mem0_provider:
            A ``MemoryProvider`` that talks to the live Mem0 backend.
        shadow_read:
            When ``True``, every ``search()`` also queries Mem0 and logs
            a comparison.  **Myelin results are still returned to the agent.**
        """
        self._mem0 = mem0_provider
        self._dual_write = True
        self._shadow_read = shadow_read
        logger.info("Mem0 dual-write enabled (shadow_read=%s)", shadow_read)

    def disable_mem0(self) -> None:
        """Deactivate Mem0 integration.  Fully reversible — no restart needed."""
        self._mem0 = None
        self._dual_write = False
        self._shadow_read = False
        logger.info("Mem0 dual-write disabled")

    @property
    def is_dual_write_active(self) -> bool:
        return self._dual_write

    @property
    def is_shadow_read_active(self) -> bool:
        return self._shadow_read

    @property
    def comparison_log(self) -> list[dict[str, Any]]:
        return list(self._comparison_log)

    def clear_comparison_log(self) -> None:
        self._comparison_log.clear()

    # ── MemoryProvider ABC ─────────────────────────────────────

    async def add(
        self,
        content: str,
        *,
        agent_id: str = "hermes",
        user_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Myelin write always succeeds.  Mem0 write is fire-and-forget."""
        result = await self._myelin.add(
            content, agent_id=agent_id, user_id=user_id, metadata=metadata
        )

        if self._dual_write and self._mem0 is not None:
            try:
                await self._mem0.add(content, agent_id=agent_id, user_id=user_id, metadata=metadata)
            except Exception:
                logger.warning("Mem0 dual-write add failed (ignored)", exc_info=True)

        return result

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        agent_id: str = "hermes",
        user_id: str = "default",
    ) -> list[SearchResult]:
        """Return Myelin results.  Optionally compare with Mem0 (shadow-read)."""
        myelin_results = await self._myelin.search(
            query, top_k=top_k, agent_id=agent_id, user_id=user_id
        )

        if self._shadow_read and self._mem0 is not None:
            try:
                mem0_results = await self._mem0.search(
                    query, top_k=top_k, agent_id=agent_id, user_id=user_id
                )
                comparison = compare_search_results(myelin_results, mem0_results)
                self._comparison_log.append(
                    {
                        "query": query,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "mem0_count": len(mem0_results),
                        "myelin_count": len(myelin_results),
                        "overlap": comparison.overlap,
                        "precision": comparison.precision,
                        "quality": comparison.quality,
                        "extra_in_mem0": [
                            {"id": r.id, "memory": r.memory[:100]}
                            for r in comparison.extra_in_primary
                        ],
                        "extra_in_myelin": [
                            {"id": r.id, "memory": r.memory[:100]}
                            for r in comparison.extra_in_shadow
                        ],
                    }
                )
            except Exception:
                logger.warning("Mem0 shadow-read failed (ignored)", exc_info=True)

        return myelin_results

    async def update(
        self,
        memory_id: str,
        text: str,
        *,
        agent_id: str = "hermes",
    ) -> dict[str, Any]:
        result = await self._myelin.update(memory_id, text, agent_id=agent_id)

        if self._dual_write and self._mem0 is not None:
            try:
                await self._mem0.update(memory_id, text, agent_id=agent_id)
            except Exception:
                logger.warning("Mem0 dual-write update failed (ignored)", exc_info=True)

        return result

    async def delete(
        self,
        memory_id: str,
        *,
        agent_id: str = "hermes",
    ) -> dict[str, Any]:
        result = await self._myelin.delete(memory_id, agent_id=agent_id)

        if self._dual_write and self._mem0 is not None:
            try:
                await self._mem0.delete(memory_id, agent_id=agent_id)
            except Exception:
                logger.warning("Mem0 dual-write delete failed (ignored)", exc_info=True)

        return result

    async def system_prompt_block(
        self,
        *,
        agent_id: str = "hermes",
        user_id: str = "default",
    ) -> str:
        block = await self._myelin.system_prompt_block(agent_id=agent_id, user_id=user_id)

        if self._dual_write and self._mem0 is not None:
            try:
                mem0_block = await self._mem0.system_prompt_block(
                    agent_id=agent_id, user_id=user_id
                )
                block += f"\n{mem0_block}"
            except Exception:
                pass

        return block

    async def health(self) -> dict[str, Any]:
        myelin_health = await self._myelin.health()
        mem0_ok = False
        if self._mem0 is not None:
            try:
                mem0_health = await self._mem0.health()
                mem0_ok = mem0_health.get("ok", False)
            except Exception:
                pass
        return {
            "ok": myelin_health.get("ok", False),
            "detail": f"myelin={myelin_health.get('ok', False)} mem0={mem0_ok} mode={'dual_write' if self._dual_write else 'myelin_only'}",
            "myelin": myelin_health,
            "mem0_available": mem0_ok,
            "dual_write_active": self._dual_write,
            "shadow_read_active": self._shadow_read,
        }
