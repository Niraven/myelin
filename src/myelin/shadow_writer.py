"""ShadowDualWriteWrapper — reversible Mem0 dual-write/shadow-read mode.

Wraps a :class:`MyelinProvider` and optionally mirrors writes to a Mem0
``Memory`` instance.  Mem0 failures are isolated (logged, never propagated),
results are compared, and discrepancies are recorded for audit.  Supports
disable (``mode="myelin_only"``) and rollback (remove shadow data from Mem0
without touching Mem0's own data) without deleting Mem0's original data.

Key design decisions
--------------------
* Mem0 is *additive* — this wrapper never deletes or modifies data that
  Mem0 owned before the wrapper started writing to it.
* Disable/rollback toggles the shadow-write path; ``mem0`` is always
  available for reads when ``mode="dual"`` (shadow-read), but isolation
  means a crashed or unavailable Mem0 never blocks the Myelin path.
* Discrepancies are stored locally in a SQLite table (``shadow_discrepancies``)
  so they survive restarts and can be queried later.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from myelin.core.database import Database


class DualWriteMode(str, Enum):
    """Operational mode for the shadow wrapper."""

    MYELIN_ONLY = "myelin_only"
    """Only write to Myelin.  Mem0 is never touched (safe default)."""

    DUAL = "dual"
    """Write to both Myelin and Mem0.  Mem0 errors are isolated."""

    SHADOW_READ = "shadow_read"
    """Write to both, *and* prefer Mem0 results for ``search()`` when
    available.  Discrepancies are still recorded."""


@dataclass
class DualWriteConfig:
    """Configuration for :class:`ShadowDualWriteWrapper`.

    Attributes
    ----------
    mode:
        Initial operational mode.
    record_discrepancies:
        If True, compare Myelin and Mem0 results on every dual-write
        ``search()`` and log differences to ``shadow_discrepancies``.
    mem0_import_path:
        Dotted path for importing the Mem0 ``Memory`` class.
        Default ``mem0.memory.main.Memory``.
    mem0_kwargs:
        Keyword arguments forwarded to the Mem0 ``Memory`` constructor.
        E.g. ``{"config": {"version": "v1.1"}}``.
    """

    mode: DualWriteMode = DualWriteMode.MYELIN_ONLY
    record_discrepancies: bool = True
    mem0_import_path: str = "mem0.memory.main.Memory"
    mem0_kwargs: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# ShadowDualWriteWrapper
# ──────────────────────────────────────────────────────────────────────


class ShadowDualWriteWrapper:
    """Wraps a :class:`MyelinProvider` with optional Mem0 shadow/dual-write.

    Usage
    -----
    >>> from myelin.provider import MyelinProvider
    >>> from myelin.shadow_writer import ShadowDualWriteWrapper
    >>> provider = MyelinProvider(db)
    >>> wrapper = ShadowDualWriteWrapper(
    ...     provider, db,
    ...     config=DualWriteConfig(mode=DualWriteMode.MYELIN_ONLY)
    ... )
    >>> wrapper.add("hello world")
    """

    def __init__(
        self,
        myelin_provider: Any,  # MyelinProvider, but duck-typed for simplicity
        db: Database,
        config: DualWriteConfig | None = None,
    ) -> None:
        self._myelin = myelin_provider
        self.db = db
        self.config = config or DualWriteConfig()

        # Mem0 instance — created lazily on first dual access
        self._mem0: Any = None
        self._mem0_available: bool = False
        self._mem0_init_error: str | None = None

        # Ensure shadow_discrepancies table exists
        self._ensure_tables()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def mode(self) -> DualWriteMode:
        return self.config.mode

    @mode.setter
    def mode(self, value: DualWriteMode | str) -> None:
        if isinstance(value, str):
            value = DualWriteMode(value)
        self.config.mode = value

    @property
    def mem0_available(self) -> bool:
        """Whether Mem0 was successfully initialised and is reachable."""
        if self._mem0 is None and not self._mem0_init_error:
            self._init_mem0()
        return self._mem0_available

    # ── Public API (delegates to MyelinProvider, optionally shadows) ──

    def add(
        self,
        messages: str | list[dict[str, str]] | dict[str, Any],
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Store memories in Myelin (+ Mem0 when in dual mode)."""
        myelin_entries = self._myelin.add(
            messages=messages,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata,
            **kwargs,
        )

        if self.config.mode != DualWriteMode.MYELIN_ONLY:
            self._shadow_add(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                myelin_ids=[e["id"] for e in myelin_entries],
                **kwargs,
            )

        return myelin_entries

    def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search memories.

        In ``SHADOW_READ`` mode, prefers Mem0 results when available
        and records discrepancies.
        """
        myelin_results = self._myelin.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            **kwargs,
        )

        if self.config.mode == DualWriteMode.SHADOW_READ and self.mem0_available:
            try:
                mem0_results = self._mem0.search(
                    query=query,
                    user_id=user_id,
                    agent_id=agent_id,
                    limit=limit,
                    **kwargs,
                )
                if self.config.record_discrepancies:
                    self._record_discrepancies(
                        query, myelin_results, mem0_results
                    )
                return mem0_results
            except Exception as exc:
                self._log_mem0_error("search", exc)

        if self.config.mode == DualWriteMode.DUAL and self.mem0_available:
            try:
                mem0_results = self._mem0.search(
                    query=query,
                    user_id=user_id,
                    agent_id=agent_id,
                    limit=limit,
                    **kwargs,
                )
                if self.config.record_discrepancies:
                    self._record_discrepancies(
                        query, myelin_results, mem0_results
                    )
            except Exception as exc:
                self._log_mem0_error("search", exc)

        return myelin_results

    def update(
        self,
        memory_id: str,
        data: str | dict[str, Any],
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update a memory entry in Myelin (+ Mem0 in dual mode)."""
        myelin_entry = self._myelin.update(
            memory_id=memory_id,
            data=data,
            agent_id=agent_id,
            **kwargs,
        )

        if self.config.mode != DualWriteMode.MYELIN_ONLY:
            self._shadow_update(
                memory_id=memory_id,
                data=data,
                agent_id=agent_id,
                **kwargs,
            )

        return myelin_entry

    def delete(
        self,
        memory_id: str,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Delete a memory in Myelin (+ Mem0 in dual mode)."""
        myelin_result = self._myelin.delete(
            memory_id=memory_id,
            agent_id=agent_id,
            **kwargs,
        )

        if self.config.mode != DualWriteMode.MYELIN_ONLY:
            self._shadow_delete(
                memory_id=memory_id,
                agent_id=agent_id,
                **kwargs,
            )

        return myelin_result

    def get_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Get all memories from Myelin (+ Mem0 comparison in dual mode)."""
        return self._myelin.get_all(
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            **kwargs,
        )

    # ── Rollback support ─────────────────────────────────────────────

    def rollback_shadow(self, agent_id: str | None = None) -> dict[str, Any]:
        """Remove shadow-written data from Mem0.

        This *only* removes data that the wrapper wrote to Mem0 (tracked
        via provenance table).  It leaves any pre-existing Mem0 data
        untouched.

        Returns a summary of what was removed.
        """
        if not self.mem0_available:
            return {"status": "skipped", "reason": "Mem0 not available"}

        provenance = self._get_provenance(agent_id=agent_id)
        removed = 0
        errors: list[str] = []

        for prov in provenance:
            mem0_id = prov.get("mem0_id")
            if not mem0_id:
                continue
            try:
                self._mem0.delete(memory_id=mem0_id)
                self._clear_provenance(mem0_id)
                removed += 1
            except Exception as exc:
                errors.append(f"{mem0_id}: {exc}")

        return {
            "status": "completed",
            "removed": removed,
            "errors": errors,
        }

    def disable_shadow(self) -> dict[str, Any]:
        """Switch to MYELIN_ONLY mode without touching Mem0 data.

        The wrapper stops writing to Mem0.  Mem0 data (including
        shadow-written entries) remains in place for future reference.
        """
        previous = self.config.mode
        self.config.mode = DualWriteMode.MYELIN_ONLY
        return {
            "status": "disabled",
            "previous_mode": previous.value,
            "note": "Mem0 data preserved in place. Call enable_shadow() to resume.",
        }

    def enable_shadow(
        self,
        mode: DualWriteMode = DualWriteMode.DUAL,
    ) -> dict[str, Any]:
        """Re-enable shadow/dual-write mode."""
        self.config.mode = mode
        return {
            "status": "enabled",
            "mode": mode.value,
        }

    # ── Mem0 initialisation ──────────────────────────────────────────

    def _init_mem0(self) -> None:
        """Lazy import and instantiate Mem0 ``Memory``."""
        if self._mem0 is not None or self._mem0_init_error:
            return
        try:
            import importlib

            module_path, _, class_name = self.config.mem0_import_path.rpartition(".")
            module = importlib.import_module(module_path)
            mem0_cls = getattr(module, class_name)
            self._mem0 = mem0_cls(**self.config.mem0_kwargs)
            self._mem0_available = True
        except ImportError as exc:
            self._mem0_init_error = f"ImportError: {exc}"
            self._mem0_available = False
        except Exception as exc:
            self._mem0_init_error = f"{type(exc).__name__}: {exc}"
            self._mem0_available = False

    # ── Shadow write helpers ─────────────────────────────────────────

    def _shadow_add(
        self,
        messages: str | list[dict[str, str]] | dict[str, Any],
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        myelin_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Mirror an ``add`` to Mem0.  Failure is isolated.

        Provenance is always recorded for Myelin IDs so the audit trail
        is complete even when Mem0 is unreachable.
        """
        effective_agent = agent_id or self._myelin.config.agent_id

        if not self.mem0_available:
            self._record_provenance(
                action="add",
                myelin_ids=myelin_ids or [],
                mem0_ids=[],
                agent_id=effective_agent,
            )
            self._log_mem0_error(
                "add",
                RuntimeError(self._mem0_init_error or "Mem0 not available"),
            )
            return

        try:
            mem0_result = self._mem0.add(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                **kwargs,
            )
            self._record_provenance(
                action="add",
                myelin_ids=myelin_ids or [],
                mem0_ids=self._extract_mem0_ids(mem0_result),
                agent_id=effective_agent,
            )
        except Exception as exc:
            self._record_provenance(
                action="add",
                myelin_ids=myelin_ids or [],
                mem0_ids=[],
                agent_id=effective_agent,
            )
            self._log_mem0_error("add", exc)

    def _shadow_update(
        self,
        memory_id: str,
        data: str | dict[str, Any],
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Mirror an ``update`` to Mem0.  Failure is isolated."""
        if not self.mem0_available:
            return
        try:
            self._mem0.update(
                memory_id=memory_id,
                data=data,
                agent_id=agent_id,
                **kwargs,
            )
        except Exception as exc:
            self._log_mem0_error("update", exc)

    def _shadow_delete(
        self,
        memory_id: str,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Mirror a ``delete`` to Mem0.  Failure is isolated."""
        if not self.mem0_available:
            return
        try:
            self._mem0.delete(
                memory_id=memory_id,
                agent_id=agent_id,
                **kwargs,
            )
        except Exception as exc:
            self._log_mem0_error("delete", exc)

    # ── Discrepancy recording ────────────────────────────────────────

    def _record_discrepancies(
        self,
        query: str,
        myelin_results: list[dict[str, Any]],
        mem0_results: list[dict[str, Any]],
    ) -> None:
        """Compare Myelin and Mem0 results for a query and log diffs."""
        myelin_ids = {r.get("id", "") for r in myelin_results}
        mem0_ids = {r.get("id", "") for r in mem0_results}

        only_myelin = myelin_ids - mem0_ids
        only_mem0 = mem0_ids - myelin_ids
        common = myelin_ids & mem0_ids

        # Compare content for common IDs
        content_diffs: list[dict[str, Any]] = []
        myelin_by_id = {r.get("id", ""): r for r in myelin_results}
        mem0_by_id = {r.get("id", ""): r for r in mem0_results}
        for cid in common:
            m1 = myelin_by_id[cid].get("content", "")
            m2 = mem0_by_id[cid].get("content", "")
            if m1 != m2:
                content_diffs.append({"id": cid, "myelin": m1, "mem0": m2})

        if only_myelin or only_mem0 or content_diffs:
            self.db.insert(
                "shadow_discrepancies",
                {
                    "query": query,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "only_myelin": json.dumps(list(only_myelin)),
                    "only_mem0": json.dumps(list(only_mem0)),
                    "content_diffs": json.dumps(content_diffs),
                    "myelin_count": len(myelin_results),
                    "mem0_count": len(mem0_results),
                },
            )

    # ── Provenance tracking ────────────────────────────────────────

    def _record_provenance(
        self,
        action: str,
        myelin_ids: list[str],
        mem0_ids: list[str],
        agent_id: str,
    ) -> None:
        """Track which Mem0 IDs the wrapper wrote for rollback support."""
        self.db.insert(
            "shadow_provenance",
            {
                "action": action,
                "myelin_ids": json.dumps(myelin_ids),
                "mem0_ids": json.dumps(mem0_ids),
                "agent_id": agent_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def _get_provenance(
        self,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get provenance records for rollback."""
        if agent_id:
            rows = self.db.fetchall(
                "SELECT * FROM shadow_provenance WHERE agent_id = ?"
                " ORDER BY timestamp DESC LIMIT ?",
                (agent_id, limit),
            )
        else:
            rows = self.db.fetchall(
                "SELECT * FROM shadow_provenance ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        result: list[dict[str, Any]] = []
        for r in rows:
            entry = dict(r)
            entry["myelin_ids"] = json.loads(entry.get("myelin_ids", "[]"))
            entry["mem0_ids"] = json.loads(entry.get("mem0_ids", "[]"))
            result.append(entry)
        return result

    def _clear_provenance(self, mem0_id: str) -> None:
        """Remove a provenance record after rollback."""
        self.db.execute(
            "DELETE FROM shadow_provenance WHERE mem0_ids LIKE ?",
            (f"%{mem0_id}%",),
        )

    # ── Error handling ──────────────────────────────────────────────

    def _log_mem0_error(self, operation: str, exc: Exception) -> None:
        """Log a non-fatal Mem0 error."""
        self.db.insert(
            "shadow_errors",
            {
                "operation": operation,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    # ── Schema management ──────────────────────────────────────────

    def _ensure_tables(self) -> None:
        """Create tracking tables if they don't exist."""
        conn = self.db.conn

        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_discrepancies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                only_myelin TEXT DEFAULT '[]',
                only_mem0 TEXT DEFAULT '[]',
                content_diffs TEXT DEFAULT '[]',
                myelin_count INTEGER DEFAULT 0,
                mem0_count INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                myelin_ids TEXT DEFAULT '[]',
                mem0_ids TEXT DEFAULT '[]',
                agent_id TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                error TEXT NOT NULL,
                traceback TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        conn.commit()

    # ── Static helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_mem0_ids(result: Any) -> list[str]:
        """Extract memory IDs from a Mem0 result (flexible shape)."""
        if isinstance(result, list):
            return [
                (r.get("id") or r.get("memory_id") or str(r)) for r in result
            ]
        if isinstance(result, dict):
            entries = result.get("results") or result.get("data") or [result]
            if isinstance(entries, list):
                return [
                    e.get("id") or e.get("memory_id") or str(e)
                    for e in entries
                ]
        return [str(result)]
