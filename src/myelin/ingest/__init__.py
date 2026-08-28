"""Observation Queue — production-grade ingest for multi-agent observation streams.

SQLite WAL-backed async queue with:
- Decoupled producers (agents) from consumers (learners)
- Sensitivity ACL (public/internal/restricted)
- Batch flush with configurable thresholds
- Idempotency via dedup key
- Backpressure: non-blocking write, periodic flush

Architecture:
    Agent → observe() → enqueue() → [in-memory buffer]
        → flush() → SQLite observation_queue →[bg poller]
        → batch_insert() → episodes table

This implements the Kappa streaming pattern for observations.
Availability over consistency for ingest (CAP: AP path).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from queue import Empty, Full, Queue
from typing import Any, Literal

from myelin.core.database import Database

Sensitivity = Literal["public", "internal", "restricted"]

DEFAULT_FLUSH_INTERVAL_S = 2.0
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_QUEUE_SIZE = 10_000


def _is_sqlite_lock_or_busy(exc: BaseException) -> bool:
    """Return True for SQLite lock/busy contention, not other OperationalErrors."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return "locked" in message or "busy" in message


class ObservationQueueError(Exception):
    """Base exception for ObservationQueue operations."""


class SensitivityViolation(ObservationQueueError):  # noqa: N818
    """Raised when an agent tries to observe at a higher sensitivity than allowed."""


@dataclass
class Observation:
    """A single observation from any agent."""

    agent_id: str
    agent_profile: str
    action: str
    action_type: str
    content_text: str
    session_id: str

    sensitivity: Sensitivity = "public"
    tenant: str | None = None
    input_context: dict[str, Any] | None = None
    output_result: dict[str, Any] | None = None
    success: bool = True

    tags: list[str] | None = None
    domain: str | None = None
    idempotency_key: str | None = None

    # Auto-set
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_row(self) -> dict[str, Any]:
        """Serialize to a database row dict."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "agent_profile": self.agent_profile,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "action_type": self.action_type,
            "content_text": self.content_text,
            "input_context": json.dumps(self.input_context) if self.input_context else None,
            "output_result": json.dumps(self.output_result) if self.output_result else None,
            "success": int(self.success),
            "sensitivity": self.sensitivity,
            "tenant": self.tenant,
            "tags": json.dumps(self.tags) if self.tags else None,
            "domain": self.domain,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_observe_call(
        cls,
        agent_id: str,
        agent_profile: str,
        action: str,
        action_type: str,
        content_text: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Create an Observation from an agent's observe() call."""
        return cls(
            agent_id=agent_id,
            agent_profile=agent_profile,
            session_id=session_id or f"session-{agent_id}",
            action=action,
            action_type=action_type,
            content_text=content_text,
            **kwargs,
        )


class AgentPermissions:
    """Access control for observation sensitivity levels.

    Maps agent profiles to their maximum allowed sensitivity.
    Default: all profiles can observe at 'public' level.
    """

    def __init__(self) -> None:
        self._profile_max: dict[str, Sensitivity] = {
            "default": "public",
        }

    def set_profile_sensitivity(self, profile: str, max_sensitivity: Sensitivity) -> None:
        """Set the maximum sensitivity an agent profile can observe at."""
        self._profile_max[profile] = max_sensitivity

    def get_max_sensitivity(self, profile: str) -> Sensitivity:
        """Get the maximum sensitivity level for a profile."""
        return self._profile_max.get(profile, "public")

    def check_allowed(self, observation: Observation) -> bool:
        """Check if an agent is allowed to observe at this sensitivity level."""
        allowed = self.get_max_sensitivity(observation.agent_profile)
        levels: list[Sensitivity] = ["public", "internal", "restricted"]
        return levels.index(observation.sensitivity) <= levels.index(allowed)


class ObservationQueue:
    """Non-blocking observation ingest queue with batch flush.

    2-phase write:
        1. enqueue() — O(1), never blocks agents
        2. flush() — batch INSERT to SQLite, called by bg thread or explicitly

    Usage:
        queue = ObservationQueue(db)
        queue.enqueue(observation)
        # ... agent continues immediately ...
        # Background poller calls flush() every ~2s
    """

    def __init__(
        self,
        db: Database,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self.db = db
        self.flush_interval_s = flush_interval_s
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        self._queue: Queue[Observation] = Queue(maxsize=max_queue_size)
        self._pending_retry: list[Observation] = []
        self._permissions = AgentPermissions()
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "enqueued": 0,
            "flushed": 0,
            "dropped_backpressure": 0,
            "rejected_sensitivity": 0,
        }

        self._ensure_table()

    # ── Public API ─────────────────────────────────────────

    def enqueue(self, observation: Observation) -> None:
        """Enqueue an observation for batch processing.

        Non-blocking — raises if queue is full (backpressure signal).
        Raises SensitivityViolation if the agent's profile isn't
        permitted at the observation's sensitivity level.
        """
        if not self._permissions.check_allowed(observation):
            self._stats["rejected_sensitivity"] += 1
            raise SensitivityViolation(
                f"Agent '{observation.agent_id}' (profile '{observation.agent_profile}') "
                f"cannot observe at sensitivity '{observation.sensitivity}'"
            )

        try:
            self._queue.put_nowait(observation)
            self._stats["enqueued"] += 1
        except Full:
            self._stats["dropped_backpressure"] += 1
            raise ObservationQueueError(
                f"Observation queue full ({self.max_queue_size}). "
                f"Agent '{observation.agent_id}' observation dropped."
            ) from None

    def flush(self) -> int:
        """Flush one batch to the database, serialized across callers."""
        with self._flush_lock:
            return self._flush_batch()

    def _flush_batch(self) -> int:
        """Flush one batch while the caller holds ``_flush_lock``."""
        batch: list[Observation] = []
        while self._pending_retry and len(batch) < self.batch_size:
            batch.append(self._pending_retry.pop(0))
        while not self._queue.empty() and len(batch) < self.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break

        if not batch:
            try:
                drained = self._drain_staged(self.batch_size)
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_lock_or_busy(exc):
                    raise
                return 0
            with self._lock:
                self._stats["flushed"] += drained
            return drained

        rows = [obs.to_row() for obs in batch]
        deduped = self._deduplicate(rows)
        try:
            with self.db.transaction():
                for row in deduped:
                    self.db.insert("observation_queue", row)
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_lock_or_busy(exc):
                raise
            self._pending_retry.extend(batch)
            return 0

        try:
            self._drain_staged(len(deduped))
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_lock_or_busy(exc):
                raise
            return 0

        flushed = len(batch)
        with self._lock:
            self._stats["flushed"] += flushed
        return flushed

    def flush_all(self) -> int:
        """Flush remaining observations, including any lock-retry buffer.

        A lock/busy flush returns 0 while work may still sit in ``_pending_retry``.
        Stop on no progress so shutdown stays bounded; a later flush delivers
        the preserved batch once the lock clears.
        """
        total = 0
        while True:
            count = self.flush()
            total += count
            if count == 0:
                break
        return total

    def queue_size(self) -> int:
        """Number of observations waiting to be flushed, including lock retries."""
        return self._queue.qsize() + len(self._pending_retry)

    def stats(self) -> dict[str, int]:
        """Return cumulative queue statistics."""
        with self._lock:
            return dict(self._stats)

    def permissions(self) -> AgentPermissions:
        """Access the permission system for configuration."""
        return self._permissions

    # ── Background poller (for threaded use) ────────────────

    def run_poller(self, stop_event: threading.Event | None = None) -> None:
        """Run a blocking poll loop. Call from a background thread.

        Args:
            stop_event: Set this event to stop the poller gracefully.
        """
        while True:
            if stop_event and stop_event.is_set():
                self.flush_all()
                return
            try:
                self.flush()
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_lock_or_busy(exc):
                    raise
            time.sleep(self.flush_interval_s)

    # ── Internal ───────────────────────────────────────────

    def _ensure_table(self) -> None:
        """Create the staging table if it doesn't exist."""
        self.db.conn.execute("""
            CREATE TABLE IF NOT EXISTS observation_queue (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                agent_profile TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                action_type TEXT NOT NULL,
                content_text TEXT NOT NULL,
                input_context TEXT,
                output_result TEXT,
                success INTEGER NOT NULL DEFAULT 1,
                sensitivity TEXT NOT NULL DEFAULT 'public',
                tenant TEXT,
                tags TEXT,
                domain TEXT,
                idempotency_key TEXT,
                processed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.db.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_queue_processed
            ON observation_queue(processed, timestamp)
        """)
        self.db.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_queue_idempotency
            ON observation_queue(idempotency_key)
        """)

    def _drain_staged(self, limit: int) -> int:
        """Transactionally promote up to ``limit`` persisted observations to episodes."""
        if limit <= 0:
            return 0

        with self.db.transaction():
            # Select the exact staged IDs this transaction will drain. Never
            # infer them from globally newest episodes: a concurrent or
            # future-dated episode can otherwise leave a successfully ingested
            # queue row permanently pending.
            staged = self.db.fetchall(
                """SELECT id FROM observation_queue
                   WHERE processed = 0
                   ORDER BY timestamp ASC
                   LIMIT ?""",
                (limit,),
            )
            staged_ids = [row["id"] for row in staged]
            if not staged_ids:
                return 0

            placeholders = ",".join("?" for _ in staged_ids)
            self.db.conn.execute(
                f"""INSERT OR IGNORE INTO episodes (
                    id, agent_id, session_id, timestamp,
                    action, action_type, input_context, output_result, success,
                    content_text, tags, domain
                ) SELECT
                    id, agent_id, session_id, timestamp,
                    action, action_type, input_context, output_result, success,
                    content_text, tags, domain
                FROM observation_queue
                WHERE id IN ({placeholders})""",
                tuple(staged_ids),
            )
            result = self.db.conn.execute(
                f"""UPDATE observation_queue
                SET processed = 1
                WHERE processed = 0
                  AND id IN ({placeholders})
                  AND id IN (SELECT id FROM episodes)""",
                tuple(staged_ids),
            )
            return result.rowcount

    def _deduplicate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicates by idempotency_key within the batch.

        If two observations in the same batch have the same idempotency_key,
        only the last one is kept (last-writer-wins).
        """
        seen: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row.get("idempotency_key") or row["id"]
            seen[key] = row  # last writer wins
        return list(seen.values())
