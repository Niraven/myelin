"""Test temporal reasoning index."""

import pytest

from myelin.core.database import Database
from myelin.knowledge.entities import EntityStore
from myelin.knowledge.temporal import TemporalIndex


@pytest.fixture
def db(tmp_path):
    d = Database(path=tmp_path / "test.db", enable_vec=False)
    _ = d.conn
    yield d
    d.close()


@pytest.fixture
def store(db):
    return EntityStore(db)


@pytest.fixture
def temporal(db):
    return TemporalIndex(db)


class TestTemporalIndex:
    def test_record_state(self, temporal):
        sid = temporal.record_state("Service is healthy", domain="deployment")
        assert sid is not None

    def test_get_current_state(self, temporal, store):
        eid = store.upsert_entity("redis", "service", "redis")
        temporal.record_state("Redis is running", entity_id=eid)
        state = temporal.get_current_state(eid)
        assert state is not None
        assert state["state_description"] == "Redis is running"
        assert state["valid_until"] is None

    def test_state_supersedes_previous(self, temporal, store):
        eid = store.upsert_entity("redis", "service", "redis")
        temporal.record_state("Redis is running", entity_id=eid)
        temporal.record_state("Redis is down", entity_id=eid)

        current = temporal.get_current_state(eid)
        assert current["state_description"] == "Redis is down"

        history = temporal.get_state_history(eid)
        assert len(history) == 2
        closed = [h for h in history if h["valid_until"] is not None]
        assert len(closed) == 1

    def test_get_state_history(self, temporal, store):
        eid = store.upsert_entity("api", "service", "api")
        temporal.record_state("API v1 deployed", entity_id=eid)
        temporal.record_state("API v2 deployed", entity_id=eid)
        temporal.record_state("API v2.1 hotfix", entity_id=eid)
        history = temporal.get_state_history(eid)
        assert len(history) == 3
        assert history[0]["state_description"] == "API v2.1 hotfix"

    def test_get_state_transitions(self, temporal, store):
        eid = store.upsert_entity("db", "service", "db")
        temporal.record_state("DB on v14", entity_id=eid)
        temporal.record_state("DB on v15", entity_id=eid)
        temporal.record_state("DB on v16", entity_id=eid)
        transitions = temporal.get_state_transitions(eid)
        assert len(transitions) == 2
        assert transitions[0]["from_state"] == "DB on v15"
        assert transitions[0]["to_state"] == "DB on v16"

    def test_temporal_score_current(self, temporal, store):
        eid = store.upsert_entity("svc", "service", "svc")
        temporal.record_state("Running", entity_id=eid)
        state = temporal.get_current_state(eid)
        score = temporal.temporal_score(state)
        assert score > 0.5  # Current + recent

    def test_count(self, temporal):
        temporal.record_state("State A")
        temporal.record_state("State B")
        assert temporal.count() == 2

    def test_current_states_for_domain(self, temporal, store):
        e1 = store.upsert_entity("svc1", "service", "svc1")
        e2 = store.upsert_entity("svc2", "service", "svc2")
        temporal.record_state("Healthy", entity_id=e1, domain="prod")
        temporal.record_state("Degraded", entity_id=e2, domain="prod")
        states = temporal.get_current_states_for_domain("prod")
        assert len(states) == 2
