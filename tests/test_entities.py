"""Test entity extraction and storage."""

import pytest

from myelin.core.database import Database
from myelin.knowledge.entities import (
    EntityStore,
    extract_entities_from_text,
    extract_relations_from_sequence,
)


@pytest.fixture
def db(tmp_path):
    d = Database(path=tmp_path / "test.db", enable_vec=False)
    _ = d.conn
    yield d
    d.close()


@pytest.fixture
def store(db):
    return EntityStore(db)


class TestExtractEntities:
    def test_extracts_git_commands(self):
        entities = extract_entities_from_text("Pulled latest changes", "git pull origin main")
        names = [e["canonical_name"] for e in entities]
        assert any("git pull" in n for n in names)

    def test_extracts_npm_commands(self):
        entities = extract_entities_from_text("Running tests", "npm test")
        names = [e["canonical_name"] for e in entities]
        assert any("npm test" in n for n in names)

    def test_extracts_file_paths(self):
        entities = extract_entities_from_text("Editing src/main.py for the fix")
        names = [e["canonical_name"] for e in entities]
        assert "src/main.py" in names

    def test_extracts_services(self):
        entities = extract_entities_from_text("Deployed to AWS Lambda and updated S3 bucket")
        types = {e["entity_type"] for e in entities}
        assert "service" in types

    def test_extracts_errors(self):
        entities = extract_entities_from_text("Got a TypeError when parsing the response")
        names = [e["canonical_name"] for e in entities]
        assert "typeerror" in names

    def test_deduplicates_within_text(self):
        entities = extract_entities_from_text("Run npm test then npm test again", "npm test")
        npm_test = [e for e in entities if "npm test" in e["canonical_name"]]
        assert len(npm_test) == 1

    def test_empty_text(self):
        entities = extract_entities_from_text("")
        assert entities == []

    def test_multiple_types(self):
        entities = extract_entities_from_text("Fix TypeError in deploy.py using docker build")
        types = {e["entity_type"] for e in entities}
        assert len(types) >= 2


class TestExtractRelations:
    def test_infers_sequence_relations(self):
        episodes = [
            {
                "session_id": "s1",
                "timestamp": "2024-01-01T00:00:00",
                "content_text": "git pull",
                "action": "git pull",
            },
            {
                "session_id": "s1",
                "timestamp": "2024-01-01T00:01:00",
                "content_text": "npm test",
                "action": "npm test",
            },
            {
                "session_id": "s2",
                "timestamp": "2024-01-01T01:00:00",
                "content_text": "git pull",
                "action": "git pull",
            },
            {
                "session_id": "s2",
                "timestamp": "2024-01-01T01:01:00",
                "content_text": "npm test",
                "action": "npm test",
            },
        ]
        relations = extract_relations_from_sequence(episodes)
        trigger_rels = [r for r in relations if r["relation_type"] == "triggers"]
        assert len(trigger_rels) > 0

    def test_empty_episodes(self):
        assert extract_relations_from_sequence([]) == []


class TestEntityStore:
    def test_upsert_creates_entity(self, store):
        eid = store.upsert_entity("git pull", "tool", "git pull")
        assert eid is not None
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity["canonical_name"] == "git pull"

    def test_upsert_increments_mention_count(self, store):
        eid1 = store.upsert_entity("git pull", "tool", "git pull")
        eid2 = store.upsert_entity("git pull", "tool", "git pull")
        assert eid1 == eid2
        entity = store.get_entity(eid1)
        assert entity["mention_count"] == 2

    def test_add_mention(self, store):
        eid = store.upsert_entity("npm test", "tool", "npm test")
        mid = store.add_mention(eid, "episode", "ep_001", context_snippet="Running tests")
        assert mid is not None

    def test_count(self, store):
        store.upsert_entity("git pull", "tool", "git pull")
        store.upsert_entity("npm test", "tool", "npm test")
        assert store.count() == 2

    def test_process_episode(self, store):
        ids = store.process_episode(
            episode_id="ep_001",
            content_text="Running npm test after git pull",
            action="npm test",
            domain="testing",
        )
        assert len(ids) >= 1
        assert store.count() >= 1

    def test_get_top_entities(self, store):
        for _ in range(5):
            store.upsert_entity("git pull", "tool", "git pull")
        store.upsert_entity("npm test", "tool", "npm test")
        top = store.get_top_entities(limit=10)
        assert top[0]["canonical_name"] == "git pull"
        assert top[0]["mention_count"] == 5

    def test_search(self, store):
        store.upsert_entity(
            "docker compose", "tool", "docker compose", description="Container orchestration"
        )
        results = store.search("docker")
        assert len(results) >= 1
