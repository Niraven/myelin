"""Test knowledge graph operations."""

import pytest

from myelin.core.database import Database
from myelin.knowledge.entities import EntityStore
from myelin.knowledge.graph import KnowledgeGraph


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
def graph(db):
    return KnowledgeGraph(db)


def _make_entities(store):
    e1 = store.upsert_entity("git pull", "tool", "git pull")
    e2 = store.upsert_entity("npm test", "tool", "npm test")
    e3 = store.upsert_entity("npm build", "tool", "npm build")
    e4 = store.upsert_entity("deploy.py", "file", "deploy.py")
    return e1, e2, e3, e4


class TestKnowledgeGraph:
    def test_add_relationship(self, store, graph):
        e1, e2, _, _ = _make_entities(store)
        rid = graph.add_relationship(e1, e2, "triggers")
        assert rid is not None

    def test_strengthen_relationship(self, store, graph):
        e1, e2, _, _ = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        graph.add_relationship(e1, e2, "triggers")
        rel = db_fetch_rel(graph.db, e1, e2, "triggers")
        assert rel["evidence_count"] == 2
        assert rel["strength"] > 1.0

    def test_get_neighbors_outgoing(self, store, graph):
        e1, e2, e3, _ = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        graph.add_relationship(e1, e3, "triggers")
        neighbors = graph.get_neighbors(e1, direction="out")
        assert len(neighbors) == 2

    def test_get_neighbors_incoming(self, store, graph):
        e1, e2, _, _ = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        neighbors = graph.get_neighbors(e2, direction="in")
        assert len(neighbors) == 1

    def test_get_neighbors_filtered_by_type(self, store, graph):
        e1, e2, e3, e4 = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        graph.add_relationship(e1, e4, "uses")
        neighbors = graph.get_neighbors(e1, relation_types=["triggers"], direction="out")
        assert len(neighbors) == 1

    def test_bfs_subgraph(self, store, graph):
        e1, e2, e3, _ = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        graph.add_relationship(e2, e3, "triggers")
        subgraph = graph.bfs_subgraph(e1, max_depth=2)
        assert len(subgraph["nodes"]) == 3
        assert len(subgraph["edges"]) >= 2

    def test_bfs_respects_max_depth(self, store, graph):
        e1, e2, e3, _ = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        graph.add_relationship(e2, e3, "triggers")
        subgraph = graph.bfs_subgraph(e1, max_depth=1)
        assert len(subgraph["nodes"]) == 2

    def test_find_paths(self, store, graph):
        e1, e2, e3, _ = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        graph.add_relationship(e2, e3, "triggers")
        paths = graph.find_paths(e1, e3, max_depth=3)
        assert len(paths) >= 1
        assert len(paths[0]) == 2

    def test_domain_subgraph(self, store, graph):
        e1 = store.upsert_entity("git pull", "tool", "git pull", domain="deployment")
        e2 = store.upsert_entity("npm test", "tool", "npm test", domain="deployment")
        graph.add_relationship(e1, e2, "triggers")
        subgraph = graph.get_domain_subgraph("deployment")
        assert len(subgraph["nodes"]) == 2

    def test_relationship_stats(self, store, graph):
        e1, e2, _, e4 = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        graph.add_relationship(e1, e4, "uses")
        stats = graph.get_relationship_stats()
        assert stats["triggers"] == 1
        assert stats["uses"] == 1

    def test_count(self, store, graph):
        e1, e2, _, _ = _make_entities(store)
        graph.add_relationship(e1, e2, "triggers")
        assert graph.count_relationships() == 1


def db_fetch_rel(db, source_id, target_id, rel_type):
    return db.fetchone(
        "SELECT * FROM relationships "
        "WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?",
        (source_id, target_id, rel_type),
    )
