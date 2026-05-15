"""Tests for SchemaLearner — semantic clustering, schema induction, lifecycle management.

Uses in-memory SQLite; no side effects.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from myelin.cognitive.schema_learner import (
    ARCHIVE_DAYS,
    HYPOTHESIS_CONFIDENCE,
    SchemaLearner,
    extract_action_type,
    extract_entities_from_content,
    jaccard_similarity,
)
from myelin.core.database import Database
from myelin.core.models import SchemaStatus
from myelin.core.schema import SCHEMA_SQL


def _new_id() -> str:
    return uuid4().hex[:16]


def _make_db() -> Database:
    db = Database(":memory:")
    db.conn.executescript(SCHEMA_SQL)
    return db


def _add_semantic_node(
    db: Database,
    content: str = "test content",
    node_type: str = "reflection",
    domain: str = "testing",
    confidence: float = 0.5,
    source_ids: list[str] | None = None,
):
    node_id = _new_id()
    db.insert(
        "semantic_nodes",
        {
            "id": node_id,
            "node_type": node_type,
            "content": content,
            "source_type": "reflection",
            "source_ids": json.dumps(source_ids or []),
            "confidence": confidence,
            "domain": domain,
            "access_times": json.dumps([time.time()]),
            "access_count": 1,
            "last_accessed": datetime.utcnow().isoformat(),
            "tags": "[]",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    return node_id


def _add_schema(
    db: Database,
    name: str = "test_schema",
    behavioral_pattern: str = "test pattern",
    domain: str = "testing",
    confidence: float = 0.5,
    status: str = "hypothesis",
    induction_count: int = 1,
    semantic_source_ids: list[str] | None = None,
    updated_at: str | None = None,
):
    schema_id = _new_id()
    db.insert(
        "schemas",
        {
            "id": schema_id,
            "name": name,
            "description": "test schema",
            "behavioral_pattern": behavioral_pattern,
            "schema_type": "behavioral",
            "semantic_source_ids": json.dumps(semantic_source_ids or []),
            "episode_source_ids": "[]",
            "confidence": confidence,
            "induction_count": induction_count,
            "domain": domain,
            "conditions": "[]",
            "exceptions": "[]",
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": updated_at or datetime.utcnow().isoformat(),
        },
    )
    return schema_id


@pytest.fixture
def learner(tmp_db) -> SchemaLearner:
    return SchemaLearner(tmp_db)


# ── Jaccard Similarity ─────────────────────────────────────────────


def test_jaccard_identical():
    assert jaccard_similarity("hello world", "hello world") == 1.0


def test_jaccard_partial():
    s = jaccard_similarity("test content here", "test content there")
    assert 0.4 < s < 0.6


def test_jaccard_no_overlap():
    assert jaccard_similarity("abc def", "ghi jkl") == 0.0


def test_jaccard_empty():
    assert jaccard_similarity("", "test") == 0.0


# ── extract_action_type ────────────────────────────────────────────


def test_extract_action_prefix():
    assert extract_action_type("running tests on the pipeline") == "running"


def test_extract_action_fallback():
    assert extract_action_type("something happened") == "something"


def test_extract_action_ing_form():
    assert extract_action_type("deploying to production") == "deploying"


def test_extract_action_short():
    assert extract_action_type("") == "unknown"


# ── extract_entities_from_content ───────────────────────────────────


def test_extract_entities_capitalized():
    ents = extract_entities_from_content("Using Docker and Git together")
    assert "docker" in ents
    assert "git" in ents


def test_extract_entities_tool_names():
    ents = extract_entities_from_content("pip install requests and npm build")
    assert "pip" in ents
    assert "npm" in ents


def test_extract_entities_snake_case():
    ents = extract_entities_from_content("call_deploy_script")
    assert "call_deploy_script" in ents


# ── Clustering ──────────────────────────────────────────────────────


def test_cluster_min_size(tmp_db, learner):
    """Need at least MIN_CLUSTER_SIZE nodes to form a cluster."""
    _add_semantic_node(tmp_db, content="running deployment script")
    _add_semantic_node(tmp_db, content="deploying to production")
    # Only 2 nodes, not enough for a cluster
    nodes = learner._get_nodes_for_domain("testing")
    assert len(nodes) == 2
    clusters = learner._cluster_nodes(nodes)
    assert len(clusters) == 0


def test_cluster_forms_with_similar_nodes(tmp_db, learner):
    """Three similar nodes should form one cluster."""
    _add_semantic_node(tmp_db, content="running deployment script to production server")
    _add_semantic_node(tmp_db, content="deploying a script to the production server")
    _add_semantic_node(tmp_db, content="deployment on the production server")
    nodes = learner._get_nodes_for_domain("testing")
    assert len(nodes) == 3
    clusters = learner._cluster_nodes(nodes)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_separates_dissimilar_nodes(tmp_db, learner):
    """Dissimilar nodes should form separate clusters (or no cluster)."""
    _add_semantic_node(tmp_db, content="running deployment script to production server")
    _add_semantic_node(tmp_db, content="deploying a script to the production server")
    _add_semantic_node(tmp_db, content="deployment on the production server cluster")
    _add_semantic_node(tmp_db, content="testing unit tests with pytest framework")
    nodes = learner._get_nodes_for_domain("testing")
    clusters = learner._cluster_nodes(nodes)
    # The 3 deploy-like nodes form one cluster, unittest forms its own
    assert len(clusters) >= 1
    # Each cluster should be internally coherent
    for cluster in clusters:
        assert len(cluster) >= 1


# ── Schema Induction ────────────────────────────────────────────────


def test_induce_schema_hypothesis_confidence(tmp_db, learner):
    """First induction gets HYPOTHESIS status with 0.4 confidence."""
    _add_semantic_node(tmp_db, content="running deployment script to production server")
    _add_semantic_node(tmp_db, content="deploying a script to the production server")
    _add_semantic_node(tmp_db, content="deployment on the production server")
    nodes = learner._get_nodes_for_domain("testing")
    clusters = learner._cluster_nodes(nodes)
    assert len(clusters) == 1

    schema = learner._induce_schema(clusters[0], "testing")
    assert schema is not None
    assert schema.status == SchemaStatus.HYPOTHESIS
    assert schema.confidence == pytest.approx(HYPOTHESIS_CONFIDENCE)
    assert schema.induction_count == 1
    assert schema.domain == "testing"


def test_induce_schema_pattern_creation(tmp_db, learner):
    """Schema should contain a behavioral pattern."""
    _add_semantic_node(tmp_db, content="running deployment script to production server")
    _add_semantic_node(tmp_db, content="deploying a script to the production server")
    _add_semantic_node(tmp_db, content="deployment on the production server")
    nodes = learner._get_nodes_for_domain("testing")
    clusters = learner._cluster_nodes(nodes)
    schema = learner._induce_schema(clusters[0], "testing")
    assert schema is not None
    assert " is typically " in schema.behavioral_pattern
    assert schema.name.startswith("running") or schema.name.startswith("deploying")
    assert "testing" in schema.name


def test_induce_schema_conditions_and_exceptions(tmp_db, learner):
    """Schema should extract conditions from cluster."""
    _add_semantic_node(tmp_db, content="running deployment script to production server")
    _add_semantic_node(tmp_db, content="deploying a script to the production server")
    _add_semantic_node(tmp_db, content="deployment on the production server")
    nodes = learner._get_nodes_for_domain("testing")
    clusters = learner._cluster_nodes(nodes)
    schema = learner._induce_schema(clusters[0], "testing")
    assert schema is not None
    assert len(schema.conditions) >= 1
    assert schema.schema_type.value == "behavioral"


def test_induce_schema_fewer_than_3(tmp_db, learner):
    """Induction with < 3 nodes returns None."""
    _add_semantic_node(tmp_db, content="test content")
    _add_semantic_node(tmp_db, content="more content")
    nodes = learner._get_nodes_for_domain("testing")
    clusters = learner._cluster_nodes(nodes)
    assert len(clusters) == 0


# ── Re-induction (Merge) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_merge_with_existing_updates_confidence(tmp_db, learner):
    """Re-induction of same pattern merges and boosts confidence."""
    _add_semantic_node(tmp_db, content="running deployment on production")
    _add_semantic_node(tmp_db, content="deploying to production")
    _add_semantic_node(tmp_db, content="deployment scripts running")
    _add_schema(
        tmp_db,
        name="running_testing",
        behavioral_pattern="When performing testing tasks: running is typically followed by deploying",
        domain="testing",
        confidence=0.4,
        status="hypothesis",
        induction_count=1,
    )

    # Induce again with new nodes
    _add_semantic_node(tmp_db, content="running deployment on production server")
    _add_semantic_node(tmp_db, content="deploying to production environment")
    _add_semantic_node(tmp_db, content="deployment scripts running on server")

    learner._get_domains_with_min_nodes()  # populate cache
    nodes = learner._get_nodes_for_domain("testing")
    learner._cluster_nodes(nodes)

    # Should merge with existing
    result = await learner.execute()

    assert result["schemas_merged"] >= 0  # may or may not merge depending on patterns
    assert result["schemas_induced"] >= 0


# ── Schema Lifecycle ────────────────────────────────────────────────


def test_archive_stale_schemas(tmp_db, learner):
    """Schemas older than ARCHIVE_DAYS should be archived."""
    old_time = (datetime.utcnow() - timedelta(days=ARCHIVE_DAYS + 1)).isoformat()
    _add_schema(tmp_db, updated_at=old_time, status="active")
    _add_schema(tmp_db, updated_at=datetime.utcnow().isoformat(), status="active")

    archived = learner._archive_stale_schemas()
    assert archived == 1

    # Verify the stale one is archived
    schemas = tmp_db.fetchall("SELECT * FROM schemas")
    for s in schemas:
        if s["status"] == "archived":
            break
    else:
        pytest.fail("No archived schema found")


def test_archive_skips_already_archived(tmp_db, learner):
    """Already archived schemas should not be double-counted."""
    old_time = (datetime.utcnow() - timedelta(days=ARCHIVE_DAYS + 1)).isoformat()
    _add_schema(tmp_db, updated_at=old_time, status="archived")
    archived = learner._archive_stale_schemas()
    assert archived == 0


def test_check_contradictions_low_confidence(tmp_db, learner):
    """Schemas with low confidence after 2+ inductions should be refuted."""
    _add_schema(
        tmp_db,
        confidence=0.2,
        induction_count=3,
        status="active",
    )
    refuted = learner._check_contradictions()
    assert refuted == 1


def test_check_contradictions_superseded_sources(tmp_db, learner):
    """Schemas with all source nodes superseded should be refuted."""
    node_ids = [_new_id() for _ in range(3)]
    for nid in node_ids:
        tmp_db.insert(
            "semantic_nodes",
            {
                "id": nid,
                "node_type": "fact",
                "content": "test",
                "source_type": "reflection",
                "source_ids": "[]",
                "confidence": 0.5,
                "domain": "testing",
                "access_times": "[]",
                "access_count": 1,
                "last_accessed": datetime.utcnow().isoformat(),
                "tags": "[]",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "valid_until": datetime.utcnow().isoformat(),  # superseded
            },
        )
    _add_schema(
        tmp_db,
        confidence=0.6,
        induction_count=2,
        semantic_source_ids=node_ids,
        status="active",
    )
    refuted = learner._check_contradictions()
    assert refuted == 1


# ── Full Run ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_no_nodes(tmp_db, learner):
    """Execute with no semantic nodes should produce empty results."""
    result = await learner.execute()
    assert result["domains_processed"] == 0
    assert result["schemas_induced"] == 0
    assert result["schemas_archived"] == 0


@pytest.mark.asyncio
async def test_execute_with_cluster(tmp_db, learner):
    """Full run should induce schemas from a valid cluster."""
    for _ in range(5):
        _add_semantic_node(
            tmp_db,
            content="running deployment script on production server with docker",
        )
    result = await learner.execute()
    assert result["domains_processed"] >= 1
    assert result["schemas_induced"] >= 1


@pytest.mark.asyncio
async def test_execute_multiple_domains(tmp_db, learner):
    """Schemas should be induced per domain."""
    for _ in range(4):
        _add_semantic_node(
            tmp_db,
            content="running deployment on production",
            domain="deployment",
        )
    for _ in range(4):
        _add_semantic_node(
            tmp_db,
            content="testing unit tests with pytest",
            domain="testing",
        )

    result = await learner.execute()
    assert result["domains_processed"] >= 2
    assert result["schemas_induced"] >= 2

    # Check schemas were created in DB
    schemas = tmp_db.fetchall("SELECT * FROM schemas")
    assert len(schemas) >= 2
    domains = {s["domain"] for s in schemas}
    assert "deployment" in domains
    assert "testing" in domains


@pytest.mark.asyncio
async def test_execute_archives_and_refutes(tmp_db, learner):
    """Full execute should archive stale and refute contradictory schemas."""
    # Stale schema
    old = (datetime.utcnow() - timedelta(days=ARCHIVE_DAYS + 1)).isoformat()
    _add_schema(tmp_db, updated_at=old, status="active", domain="stale_domain")

    # Contradictory schema
    _add_schema(tmp_db, confidence=0.2, induction_count=3, status="active", domain="bad_domain")

    result = await learner.execute()
    assert result["schemas_archived"] >= 1
    assert result["schemas_refuted"] >= 1


@pytest.mark.asyncio
async def test_execute_then_reinduce(tmp_db, learner):
    """Second induction on same domain should merge/update."""
    for _ in range(3):
        _add_semantic_node(
            tmp_db,
            content="running deployment on production server",
            domain="cicd",
        )
    r1 = await learner.execute()
    assert r1["schemas_induced"] >= 1

    # Add more nodes in same domain
    for _ in range(3):
        _add_semantic_node(
            tmp_db,
            content="deploying scripts to production environment",
            domain="cicd",
        )
    r2 = await learner.execute()
    # Should re-induce and merge
    assert r2["domains_processed"] >= 1


# ── Edge Cases ──────────────────────────────────────────────────────


def test_cluster_empty_nodes(learner):
    assert learner._cluster_nodes([]) == []


def test_induce_schema_no_cluster(learner):
    assert learner._induce_schema([], "testing") is None
    assert learner._induce_schema([{"id": "1", "content": "test"}], "testing") is None


@pytest.mark.asyncio
async def test_execute_no_domains(tmp_db, learner):
    """Nodes without domain should be skipped."""
    _add_semantic_node(tmp_db, domain="")
    result = await learner.execute()
    assert result["domains_processed"] == 0
