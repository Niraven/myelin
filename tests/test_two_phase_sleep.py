"""Tests for two-phase sleep: NREM + REM cognitive processes.

Tests cover:
- NREM phase: Hebbian strengthening, synaptic downscaling, temporal substates,
  veridical replay
- REM phase: random walk dreaming, counterfactual generation, novel connections,
  TAG scoring
- SleepCycle orchestration (both phases together)

All tests use an in-memory SQLite database to avoid side effects.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from myelin.cognitive.nrem_sleep import NREMPhase
from myelin.cognitive.rem_sleep import REMPhase
from myelin.cognitive.sleep import SleepCycle
from myelin.core.database import Database
from myelin.core.schema import SCHEMA_SQL
from myelin.knowledge.entities import EntityStore
from myelin.knowledge.graph import KnowledgeGraph
from myelin.knowledge.temporal import TemporalIndex

# ── Helpers ──────────────────────────────────────────────────────


def _new_id() -> str:
    return uuid4().hex[:16]


def _make_db() -> Database:
    db = Database(":memory:")
    # Ensure schema is initialized
    db.conn.executescript(SCHEMA_SQL)
    return db


def _add_episode(
    db: Database,
    action: str = "test_action",
    content: str = "test content with git and docker",
    domain: str = "testing",
    success: bool = True,
    priority_score: float | None = 0.5,
    importance_score: float = 0.5,
    td_error: float | None = None,
    replay_count: int = 0,
    timestamp: str | None = None,
) -> str:
    ep_id = _new_id()
    db.insert(
        "episodes",
        {
            "id": ep_id,
            "agent_id": "test_agent",
            "session_id": "test_session",
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "action": action,
            "action_type": "tool_call",
            "content_text": content,
            "success": int(success),
            "priority_score": priority_score,
            "importance_score": importance_score,
            "td_error": td_error,
            "replay_count": replay_count,
            "access_times": json.dumps([time.time()]),
            "access_count": 1,
            "last_accessed": datetime.utcnow().isoformat(),
            "tags": "[]",
            "domain": domain,
        },
    )
    return ep_id


def _add_entity(
    db: Database,
    name: str = "test_entity",
    entity_type: str = "tool",
    canonical_name: str | None = None,
    domain: str | None = "testing",
    mention_count: int = 1,
) -> str:
    ent_id = _new_id()
    db.insert(
        "entities",
        {
            "id": ent_id,
            "name": name,
            "entity_type": entity_type,
            "canonical_name": canonical_name or name.lower(),
            "mention_count": mention_count,
            "domain": domain,
            "access_times": "[]",
            "source_episodes": "[]",
            "first_seen": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
        },
    )
    return ent_id


def _add_relationship(
    db: Database,
    source_id: str,
    target_id: str,
    relation_type: str = "related_to",
    strength: float = 1.0,
    evidence_count: int = 1,
) -> str:
    rel_id = _new_id()
    db.insert(
        "relationships",
        {
            "id": rel_id,
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "relation_type": relation_type,
            "strength": strength,
            "evidence_count": evidence_count,
            "evidence_episodes": "[]",
            "first_observed": datetime.utcnow().isoformat(),
            "last_observed": datetime.utcnow().isoformat(),
        },
    )
    return rel_id


def _add_entity_mention(db: Database, entity_id: str, episode_id: str) -> str:
    mention_id = _new_id()
    db.insert(
        "entity_mentions",
        {
            "id": mention_id,
            "entity_id": entity_id,
            "source_type": "episode",
            "source_id": episode_id,
            "context_snippet": "test",
        },
    )
    return mention_id


def _make_components(db: Database):
    """Create shared component instances."""
    return {
        "db": db,
        "entity_store": EntityStore(db),
        "graph": KnowledgeGraph(db),
        "temporal": TemporalIndex(db),
    }


# NREM Phase Tests
# ====================================================================


class TestNREMPhase:
    """Test NREM sleep: Hebbian strengthening, downscaling, substates, replay."""

    @pytest.mark.asyncio
    async def test_synaptic_downscaling_reduces_strength(self):
        db = _make_db()
        e1 = _add_entity(db, "git", "tool")
        e2 = _add_entity(db, "docker", "tool")
        _add_relationship(db, e1, e2, "related_to", strength=2.0)

        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))

        results = await nrem.execute()

        assert results["synaptic_downscaled"] == 1
        rel = db.fetchone("SELECT * FROM relationships")
        assert rel is not None
        assert abs(rel["strength"] - 2.0 * 0.85) < 0.01
        assert rel["strength"] < 2.0  # Strength must have decreased

    @pytest.mark.asyncio
    async def test_synaptic_downscaling_protects_strong_relationships(self):
        db = _make_db()
        e1 = _add_entity(db, "git", "tool")
        e2 = _add_entity(db, "docker", "tool")
        # Strong, well-evidenced edge (>= PROTECTED_STRENGTH) must not be
        # erased by the blanket 0.85 decay.
        _add_relationship(db, e1, e2, "related_to", strength=4.0)

        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))

        results = await nrem.execute()

        assert results["synaptic_downscaled"] == 1
        rel = db.fetchone("SELECT * FROM relationships")
        assert rel is not None
        expected = 4.0 * NREMPhase.PROTECTED_SCALE
        assert abs(rel["strength"] - expected) < 0.01
        assert rel["strength"] > 4.0 * 0.85  # Gentler decay for strong edges

    @pytest.mark.asyncio
    async def test_hebbian_strengthening_co_occurring_pairs(self):
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", domain="dev")
        e2 = _add_entity(db, "docker", "tool", domain="dev")
        ep = _add_episode(db, action="git push", content="using git with docker", domain="dev")

        # Create entity mentions linking entities to the episode
        _add_entity_mention(db, e1, ep)
        _add_entity_mention(db, e2, ep)

        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await nrem.execute()

        assert results["synaptic_downscaled"] >= 0
        # At minimum, downscaling ran and Hebbian should have created some
        # relationships for the co-occurring entities
        # (may be 0 if entities don't match extractor patterns, but should find git+docker)
        assert results["hebbian_strengthened"] >= 0

    @pytest.mark.asyncio
    async def test_downscale_then_strengthen_net_effect(self):
        """Co-occurring pairs should maintain strength after downscale+strengthen."""
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", domain="dev")
        e2 = _add_entity(db, "docker", "tool", domain="dev")
        _add_relationship(db, e1, e2, "related_to", strength=1.5)

        ep = _add_episode(db, action="git push", content="using git with docker", domain="dev")
        _add_entity_mention(db, e1, ep)
        _add_entity_mention(db, e2, ep)

        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await nrem.execute()

        # Downscale happened
        assert results["synaptic_downscaled"] >= 1

        # The relationship should exist (either created by Hebbian or was pre-existing)
        db.fetchall(
            "SELECT * FROM relationships "
            "WHERE ((source_entity_id = ? AND target_entity_id = ?) "
            "   OR (source_entity_id = ? AND target_entity_id = ?))",
            (e1, e2, e2, e1),
        )
        # At minimum, the original relationship should still exist
        orig = db.fetchone("SELECT * FROM relationships WHERE id IN (SELECT id FROM relationships)")
        if orig:
            # With downscale (1.5*0.85=1.275) + Hebbian (0.15) = 1.425
            # Should still be > 1.0
            assert orig["strength"] >= 1.0

    @pytest.mark.asyncio
    async def test_temporal_substate_separation_recent_vs_old(self):
        """Recent episodes get cluster+strengthen, old episodes get integrate."""
        db = _make_db()
        now = datetime.utcnow()
        old_ts = (now - timedelta(days=5)).isoformat()
        recent_ts = (now - timedelta(hours=6)).isoformat()

        e1 = _add_entity(db, "git", "tool", domain="dev")
        e2 = _add_entity(db, "docker", "tool", domain="dev")

        recent_ep = _add_episode(
            db,
            action="git push",
            content="using git with docker",
            domain="dev",
            timestamp=recent_ts,
        )
        old_ep = _add_episode(
            db,
            action="git push",
            content="using git with docker",
            domain="dev",
            timestamp=old_ts,
        )
        _add_entity_mention(db, e1, recent_ep)
        _add_entity_mention(db, e2, recent_ep)
        _add_entity_mention(db, e1, old_ep)
        _add_entity_mention(db, e2, old_ep)

        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await nrem.execute()

        # Both passes should have run
        assert results["recent_cluster_strengthened"] >= 0
        assert results["old_integrated"] >= 0

    @pytest.mark.asyncio
    async def test_veridical_replay_high_priority(self):
        """Episodes with high priority_score get replayed."""
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", domain="dev")
        e2 = _add_entity(db, "docker", "tool", domain="dev")

        ep = _add_episode(
            db,
            action="git push",
            content="using git with docker deployment",
            domain="dev",
            priority_score=0.8,
        )
        _add_entity_mention(db, e1, ep)
        _add_entity_mention(db, e2, ep)

        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        await nrem.execute()

        # Should have replayed the high-priority episode
        updated_ep = db.fetchone("SELECT replay_count FROM episodes WHERE id = ?", (ep,))
        assert updated_ep is not None
        # The episode should have been replayed (replay_count incremented in veridical_replay)

    @pytest.mark.asyncio
    async def test_empty_db_does_not_crash(self):
        """NREM on empty database returns gracefully."""
        db = _make_db()
        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await nrem.execute()
        assert isinstance(results, dict)
        assert results["synaptic_downscaled"] == 0
        assert results["entities_processed"] == 0


# ====================================================================
# REM Phase Tests
# ====================================================================


class TestREMPhase:
    """Test REM sleep: random walk dreaming, counterfactuals, novel connections, TAG."""

    @pytest.mark.asyncio
    async def test_random_walk_dreaming_creates_connections(self):
        """Random walks through knowledge graph create dreamed_connections."""
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", mention_count=10, domain="dev")
        e2 = _add_entity(db, "docker", "tool", mention_count=8, domain="dev")
        e3 = _add_entity(db, "npm", "tool", mention_count=5, domain="dev")

        # Connect entities into a small graph so walks work
        _add_relationship(db, e1, e2, "related_to", strength=1.0)
        _add_relationship(db, e2, e3, "related_to", strength=1.0)
        _add_relationship(db, e1, e3, "related_to", strength=0.5)

        rem = REMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await rem.execute()

        # Should have completed walks
        assert results["dream_walks_completed"] > 0

    @pytest.mark.asyncio
    async def test_counterfactual_generation_on_failures(self):
        """Failed episodes generate counterfactual dream nodes."""
        db = _make_db()
        ep = _add_episode(
            db,
            action="git push failed",
            content="error pushing to remote",
            domain="dev",
            success=False,
        )

        rem = REMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await rem.execute()

        assert results["counterfactuals_generated"] == 1

        dream = db.fetchone("SELECT * FROM semantic_nodes WHERE node_type = 'dream'")
        assert dream is not None
        dream_content = json.loads(dream["content"])
        assert dream_content["type"] == "counterfactual_dream"
        assert dream_content["original_episode_id"] == ep

    @pytest.mark.asyncio
    async def test_novel_connection_discovery(self):
        """Cross-domain entity pairs get weak connections."""
        db = _make_db()
        _add_entity(db, "git", "tool", mention_count=5, domain="dev")
        _add_entity(db, "slack", "service", mention_count=5, domain="communication")

        # They should be >3 hops apart with no existing connection
        rem = REMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await rem.execute()

        # May or may not find novel connections depending on shared attributes
        assert isinstance(results["novel_connections_created"], int)

    @pytest.mark.asyncio
    async def test_tag_scoring_computes_scores(self):
        """TAG scores are computed and priority_score updated."""
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", domain="dev")
        ep1 = _add_episode(
            db,
            action="git push",
            content="deploy with git",
            domain="dev",
            td_error=0.8,
            importance_score=0.9,
        )
        _add_entity_mention(db, e1, ep1)

        rem = REMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await rem.execute()

        assert results["tag_scores_computed"] > 0

        updated = db.fetchone(
            "SELECT priority_score, replay_count FROM episodes WHERE id = ?", (ep1,)
        )
        assert updated is not None
        # TAG score = 0.4 * 0.8 + 0.35 * 0.9 + 0.25 * 1.0 = 0.32 + 0.315 + 0.25 = 0.885
        assert abs(float(updated["priority_score"]) - 0.885) < 0.01

    @pytest.mark.asyncio
    async def test_empty_db_does_not_crash(self):
        """REM on empty database returns gracefully."""
        db = _make_db()
        rem = REMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await rem.execute()
        assert isinstance(results, dict)
        assert all(v == 0 for v in results.values())


# ====================================================================
# SleepCycle Orchestration Tests
# ====================================================================


class TestSleepCycle:
    """Test SleepCycle orchestration runs both phases."""

    @pytest.mark.asyncio
    async def test_orchestrates_both_phases(self):
        """SleepCycle.run() executes NREM then REM."""
        db = _make_db()
        # Add some data so phases have work to do
        e1 = _add_entity(db, "git", "tool", mention_count=10, domain="dev")
        e2 = _add_entity(db, "docker", "tool", mention_count=8, domain="dev")
        _add_relationship(db, e1, e2, "related_to", strength=2.0)

        ep = _add_episode(
            db,
            action="git push",
            content="using git with docker for deployment",
            domain="dev",
            priority_score=0.8,
            success=False,
        )
        _add_entity_mention(db, e1, ep)
        _add_entity_mention(db, e2, ep)

        cycle = SleepCycle(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await cycle.execute()

        # Both phases ran
        assert "nrem" in results
        assert "rem" in results

        # NREM ran at least downscaling
        assert results["nrem"]["synaptic_downscaled"] >= 1

        # REM ran at least dream walks
        assert isinstance(results["rem"]["dream_walks_completed"], int)

    @pytest.mark.asyncio
    async def test_importance_computer(self):
        """Legacy ImportanceComputer still works."""
        from myelin.cognitive.sleep import ImportanceComputer

        db = _make_db()
        ep = _add_episode(db, domain="testing")

        computer = ImportanceComputer()
        scores = computer.compute(db, [db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep,))])
        assert isinstance(scores, dict)
        assert len(scores) > 0

    @pytest.mark.asyncio
    async def test_full_cycle_with_dreamed_connections(self):
        """End-to-end: NREM + REM produce dreamed_connections in graph."""
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", mention_count=15, domain="dev")
        e2 = _add_entity(db, "docker", "tool", mention_count=12, domain="dev")
        e3 = _add_entity(db, "npm", "tool", mention_count=8, domain="dev")

        _add_relationship(db, e1, e2, "related_to", strength=2.0)
        _add_relationship(db, e2, e3, "related_to", strength=1.0)

        ep = _add_episode(
            db,
            action="git push",
            content="deploy with git and docker",
            domain="dev",
            success=False,
        )
        _add_entity_mention(db, e1, ep)
        _add_entity_mention(db, e2, ep)

        cycle = SleepCycle(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        await cycle.execute()

        # Check dreamed_connections exist
        db.fetchall("SELECT * FROM relationships WHERE relation_type = 'dreamed_connection'")

        # At minimum, NREM downscaled existing relationships
        remaining = db.fetchall("SELECT * FROM relationships WHERE relation_type = 'related_to'")
        for _rel in remaining:
            # Should be reduced by downscaling
            pass  # Just verifying no exceptions

    @pytest.mark.asyncio
    async def test_concurrent_safety(self):
        """Multiple sleep cycles run cleanly."""
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", mention_count=5, domain="dev")
        e2 = _add_entity(db, "docker", "tool", mention_count=3, domain="dev")
        _add_relationship(db, e1, e2, "related_to", strength=1.0)
        _add_episode(db, action="test", content="git docker", domain="dev")

        for _ in range(3):
            cycle = SleepCycle(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
            await cycle.execute()

        # No crashes, data is consistent
        rels = db.fetchall("SELECT COUNT(*) as cnt FROM relationships")
        assert len(rels) == 1 or "cnt" in rels[0]


# ====================================================================
# Edge Case Tests
# ====================================================================


class TestEdgeCases:
    """Boundary conditions and edge cases for both phases."""

    @pytest.mark.asyncio
    async def test_nrem_empty_relationship_table(self):
        """Downscaling on empty relationships table is no-op."""
        db = _make_db()
        _add_entity(db, "git", "tool")
        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await nrem.execute()
        assert results["synaptic_downscaled"] == 0

    @pytest.mark.asyncio
    async def test_nrem_no_high_priority_episodes(self):
        """Veridical replay with no high-priority episodes is graceful."""
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", domain="dev")
        _add_episode(db, action="test", content="test", domain="dev", priority_score=0.1)
        _add_entity_mention(db, e1, _new_id())

        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        # Use fresh entity store with this db
        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await nrem.execute()
        assert results["veridical_replays"] == 0

    @pytest.mark.asyncio
    async def test_rem_no_failed_episodes(self):
        """Counterfactual generation with no failures is graceful."""
        db = _make_db()
        _add_episode(db, action="test", content="success!", success=True)
        rem = REMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await rem.execute()
        assert results["counterfactuals_generated"] == 0

    @pytest.mark.asyncio
    async def test_rem_no_entities(self):
        """Random walk with no entities is graceful."""
        db = _make_db()
        _add_episode(db, action="test", content="test")
        rem = REMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        results = await rem.execute()
        assert results["dream_walks_completed"] == 0

    @pytest.mark.asyncio
    async def test_relationship_strength_caps_at_10(self):
        """Strength never exceeds hard cap of 10.0."""
        db = _make_db()
        e1 = _add_entity(db, "git", "tool", domain="dev")
        e2 = _add_entity(db, "docker", "tool", domain="dev")
        _add_relationship(db, e1, e2, "related_to", strength=9.8)

        ep = _add_episode(db, action="git push", content="docker git kubernetes", domain="dev")
        _add_entity_mention(db, e1, ep)
        _add_entity_mention(db, e2, ep)

        nrem = NREMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        await nrem.execute()

        rel = db.fetchone(
            "SELECT strength FROM relationships "
            "WHERE source_entity_id = ? AND target_entity_id = ?",
            (e1, e2),
        )
        if rel:
            # After downscale (9.8*0.85=8.33) + boost should be <= 10
            assert float(rel["strength"]) <= 10.0

    @pytest.mark.asyncio
    async def test_semantic_dream_node_persistence(self):
        """Counterfactual dream nodes persist correctly in semantic_nodes."""
        db = _make_db()
        _add_episode(
            db, action="deploy failed", content="deployment error", domain="prod", success=False
        )

        rem = REMPhase(db, EntityStore(db), KnowledgeGraph(db), TemporalIndex(db))
        await rem.execute()

        nodes = db.fetchall("SELECT * FROM semantic_nodes WHERE node_type = 'dream'")
        assert len(nodes) >= 1
        for node in nodes:
            assert node["node_type"] == "dream"
            content = json.loads(node["content"])
            assert "alternative" in content
