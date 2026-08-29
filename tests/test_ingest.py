"""Tests for ObservationQueue — multi-agent ingest with backpressure and ACL."""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from myelin.core.database import Database
from myelin.ingest import (
    AgentPermissions,
    Observation,
    ObservationQueue,
    ObservationQueueError,
    Sensitivity,
    SensitivityViolation,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    _db = Database(tmp_path / "test.db", enable_vec=False)
    return _db


@pytest.fixture
def queue(db: Database) -> ObservationQueue:
    q = ObservationQueue(db, flush_interval_s=0.1, batch_size=10, max_queue_size=100)
    return q


def make_obs(
    agent_id: str = "test-agent",
    profile: str = "default",
    sensitivity: Sensitivity = "public",
    **kwargs,
) -> Observation:
    return Observation(
        agent_id=agent_id,
        agent_profile=profile,
        action="test_action",
        action_type="tool_call",
        content_text="test observation",
        session_id="test-session",
        sensitivity=sensitivity,
        **kwargs,
    )


class TestObservation:
    def test_to_row_serialization(self):
        obs = make_obs(tags=["test", "demo"], domain="testing")
        row = obs.to_row()
        assert row["id"] == obs.id
        assert row["agent_id"] == "test-agent"
        assert row["sensitivity"] == "public"
        assert json.loads(row["tags"]) == ["test", "demo"]
        assert row["domain"] == "testing"

    def test_from_observe_call(self):
        obs = Observation.from_observe_call(
            agent_id="agent-1",
            agent_profile="builder",
            action="deploy",
            action_type="tool_call",
            content_text="Deployed to production",
            session_id="session-123",
            sensitivity="internal",
            domain="devops",
        )
        assert obs.agent_id == "agent-1"
        assert obs.agent_profile == "builder"
        assert obs.sensitivity == "internal"

    def test_sensitivity_violation(self, queue):
        obs = make_obs(profile="default", sensitivity="restricted")
        with pytest.raises(SensitivityViolation):
            queue.enqueue(obs)
        assert queue.stats()["rejected_sensitivity"] == 1

    def test_enqueue_and_flush(self, queue):
        obs = make_obs(idempotency_key="key-1")
        queue.enqueue(obs)
        assert queue.queue_size() == 1

        count = queue.flush()
        assert count == 1
        assert queue.stats()["flushed"] == 1
        assert queue.queue_size() == 0

    def test_backpressure_raises(self, db):
        tiny_queue = ObservationQueue(db, max_queue_size=2)
        tiny_queue.enqueue(make_obs(idempotency_key="a"))
        tiny_queue.enqueue(make_obs(idempotency_key="b"))
        with pytest.raises(ObservationQueueError):
            tiny_queue.enqueue(make_obs(idempotency_key="c"))
        assert tiny_queue.stats()["dropped_backpressure"] == 1

    def test_dedup_by_idempotency_key(self, queue):
        queue.enqueue(make_obs(idempotency_key="dup-1"))
        queue.enqueue(make_obs(idempotency_key="dup-1"))
        count = queue.flush_all()
        # Both should flush to staging, dedup is on INSERT to episodes
        assert count == 2

    def test_flush_all(self, queue):
        for i in range(5):
            queue.enqueue(make_obs(idempotency_key=f"key-{i}"))
        total = queue.flush_all()
        assert total == 5
        assert queue.queue_size() == 0

    def test_concurrent_flushes_are_serialized(self, db, monkeypatch):
        queue = ObservationQueue(db, batch_size=1)
        queue.enqueue(make_obs(idempotency_key="concurrent-1"))
        queue.enqueue(make_obs(idempotency_key="concurrent-2"))

        original_transaction = db.transaction
        state_lock = threading.Lock()
        active = 0
        max_active = 0

        @contextmanager
        def slow_transaction():
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.05)
                with original_transaction():
                    yield
            finally:
                with state_lock:
                    active -= 1

        monkeypatch.setattr(db, "transaction", slow_transaction)
        start = threading.Barrier(3)
        counts = []
        errors = []

        def flush_once():
            start.wait()
            try:
                counts.append(queue.flush())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=flush_once) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()

        assert errors == []
        assert sorted(counts) == [1, 1]
        assert max_active == 1

    def test_stats(self, queue):
        assert queue.stats()["enqueued"] == 0
        assert queue.stats()["flushed"] == 0
        assert queue.stats()["dropped_backpressure"] == 0
        assert queue.stats()["rejected_sensitivity"] == 0

    def test_queue_persistence(self, db, queue):
        """Observations should persist in the staging table after flush."""
        obs = make_obs(idempotency_key="persist-1", tags=["persist"])
        queue.enqueue(obs)
        queue.flush_all()

        rows = db.fetchall(
            "SELECT id, agent_id, processed FROM observation_queue WHERE id = ?",
            (obs.id,),
        )
        assert len(rows) == 1
        assert rows[0]["processed"] == 1

    def test_flush_marks_the_exact_staged_records_it_ingests(self, db, queue):
        """A historical backlog row must be marked processed even if a newer episode exists."""
        backlog = make_obs(idempotency_key="backlog")
        db.insert("observation_queue", backlog.to_row())

        db.execute(
            """INSERT INTO episodes (
                id, agent_id, session_id, timestamp, action, action_type,
                success, content_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "unrelated-future-episode",
                "other-agent",
                "other-session",
                "2099-01-01T00:00:00+00:00",
                "unrelated_action",
                "tool_call",
                1,
                "An unrelated, future-dated episode.",
            ),
        )
        db.commit()

        queue.enqueue(make_obs(idempotency_key="current"))
        queue.flush()

        row = db.fetchone("SELECT processed FROM observation_queue WHERE id = ?", (backlog.id,))
        assert row["processed"] == 1
        assert db.fetchone("SELECT id FROM episodes WHERE id = ?", (backlog.id,))

    def test_idle_flush_drains_persisted_staging_backlog(self, db, queue):
        """The background poller must recover staged rows without a new producer event."""
        backlog = make_obs(idempotency_key="idle-backlog")
        db.insert("observation_queue", backlog.to_row())

        assert queue.flush() == 1
        row = db.fetchone("SELECT processed FROM observation_queue WHERE id = ?", (backlog.id,))
        assert row["processed"] == 1
        assert db.fetchone("SELECT id FROM episodes WHERE id = ?", (backlog.id,))


class TestAgentPermissions:
    def test_default_is_public(self):
        perms = AgentPermissions()
        assert perms.get_max_sensitivity("unknown-profile") == "public"

    def test_set_profile_sensitivity(self):
        perms = AgentPermissions()
        perms.set_profile_sensitivity("builder", "restricted")
        assert perms.get_max_sensitivity("builder") == "restricted"

    def test_check_allowed_public(self):
        perms = AgentPermissions()
        obs = make_obs(profile="default", sensitivity="public")
        assert perms.check_allowed(obs) is True

    def test_check_allowed_restricted_blocked(self):
        perms = AgentPermissions()
        obs = make_obs(profile="default", sensitivity="restricted")
        assert perms.check_allowed(obs) is False

    def test_check_allowed_restricted_allowed(self):
        perms = AgentPermissions()
        perms.set_profile_sensitivity("builder", "restricted")
        obs = make_obs(profile="builder", sensitivity="restricted")
        assert perms.check_allowed(obs) is True


def _episode_ids(db: Database) -> list[str]:
    return [row["id"] for row in db.fetchall("SELECT id FROM episodes")]


class TestIngestLockRetry:
    def test_staging_lock_preserves_every_original_observation(self, db, queue, monkeypatch):
        observations = [make_obs(idempotency_key=f"stage-{i}") for i in range(3)]
        for obs in observations:
            queue.enqueue(obs)

        @contextmanager
        def locked_transaction():
            raise sqlite3.OperationalError("database is locked")
            yield

        monkeypatch.setattr(db, "transaction", locked_transaction)
        assert queue.flush() == 0
        assert queue.queue_size() == 3
        assert db.fetchall("SELECT id FROM observation_queue") == []

        monkeypatch.undo()
        assert queue.flush_all() == 3
        assert queue.queue_size() == 0
        assert sorted(_episode_ids(db)) == sorted(obs.id for obs in observations)

    def test_refilled_bounded_queue_cannot_drop_on_lock_restore(self, db, monkeypatch):
        queue = ObservationQueue(db, batch_size=2, max_queue_size=2)
        original = [make_obs(idempotency_key="orig-1"), make_obs(idempotency_key="orig-2")]
        refill = [make_obs(idempotency_key="new-1"), make_obs(idempotency_key="new-2")]
        for obs in original:
            queue.enqueue(obs)

        @contextmanager
        def refill_then_lock():
            for obs in refill:
                queue.enqueue(obs)
            raise sqlite3.OperationalError("database is locked")
            yield

        monkeypatch.setattr(db, "transaction", refill_then_lock)
        assert queue.flush() == 0
        assert queue.queue_size() == 4

        monkeypatch.undo()
        assert queue.flush_all() == 4
        assert queue.queue_size() == 0
        assert sorted(_episode_ids(db)) == sorted(obs.id for obs in original + refill)

    def test_drain_lock_leaves_staged_pending_then_idle_flush_drains_once(
        self, db, queue, monkeypatch
    ):
        obs = make_obs(idempotency_key="drain-lock")
        queue.enqueue(obs)
        original_drain = queue._drain_staged
        calls = {"n": 0}

        def flaky_drain(limit: int) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is busy")
            return original_drain(limit)

        monkeypatch.setattr(queue, "_drain_staged", flaky_drain)
        assert queue.flush() == 0
        row = db.fetchone("SELECT processed FROM observation_queue WHERE id = ?", (obs.id,))
        assert row is not None
        assert row["processed"] == 0
        assert queue.queue_size() == 0

        monkeypatch.setattr(queue, "_drain_staged", original_drain)
        assert queue.flush() == 1
        row = db.fetchone("SELECT processed FROM observation_queue WHERE id = ?", (obs.id,))
        assert row["processed"] == 1
        assert db.fetchall("SELECT id FROM episodes WHERE id = ?", (obs.id,)) == [{"id": obs.id}]
        assert queue.flush() == 0

    def test_run_poller_survives_locked_busy_and_delivers_once(self, db, monkeypatch):
        queue = ObservationQueue(db, flush_interval_s=0.01, batch_size=10)
        observations = [make_obs(idempotency_key=f"p-{i}") for i in range(3)]
        for obs in observations:
            queue.enqueue(obs)

        original_transaction = db.transaction
        remaining = {"n": 4}

        @contextmanager
        def flaky_transaction():
            if remaining["n"] > 0:
                remaining["n"] -= 1
                message = "database is locked" if remaining["n"] % 2 else "database is busy"
                raise sqlite3.OperationalError(message)
            with original_transaction():
                yield

        monkeypatch.setattr(db, "transaction", flaky_transaction)
        stop = threading.Event()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                queue.run_poller(stop)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(_episode_ids(db)) < 3:
            time.sleep(0.02)
        stop.set()
        thread.join(timeout=2.0)
        assert errors == []
        assert thread.is_alive() is False
        assert sorted(_episode_ids(db)) == sorted(obs.id for obs in observations)
        assert queue.queue_size() == 0

    def test_non_lock_operational_error_is_not_swallowed(self, db, queue, monkeypatch):
        obs = make_obs(idempotency_key="non-lock")
        queue.enqueue(obs)

        @contextmanager
        def other_operational_error():
            raise sqlite3.OperationalError("no such table: observation_queue")
            yield

        monkeypatch.setattr(db, "transaction", other_operational_error)
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            queue.flush()
        assert queue.queue_size() == 1
        assert queue._queue.empty()
        assert [item.id for item in queue._pending_retry] == [obs.id]

    def test_enqueue_rejects_when_pending_retry_fills_capacity(self, db, monkeypatch):
        queue = ObservationQueue(db, batch_size=2, max_queue_size=2)
        original = [make_obs(idempotency_key="pending-a"), make_obs(idempotency_key="pending-b")]
        for obs in original:
            queue.enqueue(obs)

        @contextmanager
        def locked_transaction():
            raise sqlite3.OperationalError("database is locked")
            yield

        monkeypatch.setattr(db, "transaction", locked_transaction)
        assert queue.flush() == 0
        assert queue._queue.empty()
        assert queue.queue_size() == 2

        with pytest.raises(ObservationQueueError, match="Observation queue full"):
            queue.enqueue(make_obs(idempotency_key="pending-c"))
        assert queue.stats()["dropped_backpressure"] == 1
        assert queue._queue.empty()
        assert queue.queue_size() == 2

        monkeypatch.undo()
        assert queue.flush_all() == 2
        assert queue.queue_size() == 0
        assert sorted(_episode_ids(db)) == sorted(obs.id for obs in original)

    def test_non_lock_operational_error_from_post_insert_drain_propagates(
        self, db, queue, monkeypatch
    ):
        obs = make_obs(idempotency_key="drain-non-lock")
        queue.enqueue(obs)

        def boom(_limit: int) -> int:
            raise sqlite3.OperationalError("no such table: episodes")

        monkeypatch.setattr(queue, "_drain_staged", boom)
        with pytest.raises(sqlite3.OperationalError, match="no such table: episodes"):
            queue.flush()
        row = db.fetchone("SELECT id FROM observation_queue WHERE id = ?", (obs.id,))
        assert row is not None

    def test_non_lock_operational_error_from_idle_drain_propagates(self, db, queue, monkeypatch):
        def boom(_limit: int) -> int:
            raise sqlite3.OperationalError("no such table: observation_queue")

        monkeypatch.setattr(queue, "_drain_staged", boom)
        with pytest.raises(sqlite3.OperationalError, match="no such table: observation_queue"):
            queue.flush()
        assert queue.queue_size() == 0

    def test_run_poller_propagates_non_lock_operational_error(self, db, monkeypatch):
        queue = ObservationQueue(db, flush_interval_s=0.01)
        queue.enqueue(make_obs(idempotency_key="poller-non-lock"))

        @contextmanager
        def other_operational_error():
            raise sqlite3.OperationalError("disk I/O error")
            yield

        monkeypatch.setattr(db, "transaction", other_operational_error)
        stop = threading.Event()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                queue.run_poller(stop)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=2.0)
        assert thread.is_alive() is False
        assert errors
        assert isinstance(errors[0], sqlite3.OperationalError)
        assert "disk I/O error" in str(errors[0])
        assert queue.queue_size() == 1

    def test_run_poller_stop_while_lock_remains_is_bounded_and_preserves_batch(
        self, db, monkeypatch
    ):
        queue = ObservationQueue(db, flush_interval_s=0.01)
        obs = make_obs(idempotency_key="stop-while-locked")
        queue.enqueue(obs)

        original_transaction = db.transaction
        locked = {"on": True}

        @contextmanager
        def maybe_locked():
            if locked["on"]:
                raise sqlite3.OperationalError("database is locked")
            with original_transaction():
                yield

        monkeypatch.setattr(db, "transaction", maybe_locked)
        stop = threading.Event()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                queue.run_poller(stop)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and queue.queue_size() == 0:
            time.sleep(0.01)
        started = time.monotonic()
        stop.set()
        thread.join(timeout=1.0)
        assert time.monotonic() - started < 1.0
        assert thread.is_alive() is False
        assert errors == []
        assert queue.queue_size() == 1
        assert _episode_ids(db) == []

        locked["on"] = False
        assert queue.flush_all() == 1
        assert queue.queue_size() == 0
        assert db.fetchall("SELECT id FROM episodes WHERE id = ?", (obs.id,)) == [{"id": obs.id}]

    def test_shutdown_flush_bounded_and_delivers_retry_when_lock_clears(self, db, monkeypatch):
        queue = ObservationQueue(db, flush_interval_s=0.01)
        obs = make_obs(idempotency_key="shutdown-1")
        queue.enqueue(obs)

        original_transaction = db.transaction
        locked = {"on": True}

        @contextmanager
        def maybe_locked():
            if locked["on"]:
                raise sqlite3.OperationalError("database is locked")
            with original_transaction():
                yield

        monkeypatch.setattr(db, "transaction", maybe_locked)
        started = time.monotonic()
        assert queue.flush_all() == 0
        assert time.monotonic() - started < 1.0
        assert queue.queue_size() == 1

        locked["on"] = False
        assert queue.flush_all() == 1
        assert queue.queue_size() == 0
        assert db.fetchone("SELECT id FROM episodes WHERE id = ?", (obs.id,)) is not None
