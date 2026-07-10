"""Tests for ObservationQueue — multi-agent ingest with backpressure and ACL."""

import json
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

        row = db.fetchone(
            "SELECT processed FROM observation_queue WHERE id = ?", (backlog.id,)
        )
        assert row["processed"] == 1
        assert db.fetchone("SELECT id FROM episodes WHERE id = ?", (backlog.id,))

    def test_idle_flush_drains_persisted_staging_backlog(self, db, queue):
        """The background poller must recover staged rows without a new producer event."""
        backlog = make_obs(idempotency_key="idle-backlog")
        db.insert("observation_queue", backlog.to_row())

        assert queue.flush() == 1
        row = db.fetchone(
            "SELECT processed FROM observation_queue WHERE id = ?", (backlog.id,)
        )
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
