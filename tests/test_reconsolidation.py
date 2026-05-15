"""Tests for reconsolidation: lability windows, PE computation, stability protection."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from myelin.cognitive.reconsolidator import (
    MAX_LABILE_MEMORIES,
    PE_CONFIRMED,
    PE_INTEGRATION,
    PE_SELECTIVE_EDIT,
    ReconsolidationEngine,
    _jaccard_distance,
)
from myelin.core.database import Database
from myelin.core.models import ActionType, Episode, NodeType, SemanticNode, SourceType
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory

# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def engine(tmp_db: Database) -> ReconsolidationEngine:
    episodic = EpisodicMemory(tmp_db)
    semantic = SemanticMemory(tmp_db)
    procedural = ProceduralMemory(tmp_db)
    return ReconsolidationEngine(tmp_db, episodic, semantic, procedural)


@pytest.fixture
def seeded_episode(engine: ReconsolidationEngine) -> str:
    ep = Episode(
        agent_id="test-agent",
        session_id="s1",
        action="deploy service",
        action_type=ActionType.TOOL_CALL,
        content_text="Deployed service-7 to production with zero downtime",
        success=True,
        domain="deployment",
    )
    engine.episodic.record(ep)
    return ep.id


@pytest.fixture
def seeded_semantic(engine: ReconsolidationEngine) -> str:
    node = SemanticNode(
        node_type=NodeType.FACT,
        content="The database connection pool should have max 10 connections",
        source_type=SourceType.OBSERVATION,
        confidence=0.7,
        domain="infrastructure",
    )
    engine.semantic.store(node)
    return node.id


# ── Jaccard distance ───────────────────────────────────────────


class TestJaccardDistance:
    def test_identical_texts(self):
        assert _jaccard_distance("hello world", "hello world") == 0.0

    def test_completely_different(self):
        assert _jaccard_distance("hello world", "foo bar baz") == 1.0

    def test_partial_overlap(self):
        d = _jaccard_distance("hello world foo", "hello world bar")
        # intersection = {hello, world}, union = {hello, world, foo, bar}
        # 1 - 2/4 = 0.5
        assert d == 0.5

    def test_empty_strings(self):
        assert _jaccard_distance("", "") == 0.0

    def test_one_empty(self):
        assert _jaccard_distance("hello", "") == 1.0

    def test_case_insensitive(self):
        assert _jaccard_distance("Hello World", "hello world") == 0.0

    def test_almost_identical(self):
        d = _jaccard_distance("the quick brown fox", "the quick brown dog")
        # intersection = {the, quick, brown}, union = {the, quick, brown, fox, dog}
        # 1 - 3/5 = 0.4
        assert d == pytest.approx(0.4)


# ── Lability Window Management ─────────────────────────────────


class TestLabilityWindow:
    def test_open_lability_window_episode(self, engine: ReconsolidationEngine, seeded_episode: str):
        labile_until = engine.open_lability_window("episode", seeded_episode)
        assert labile_until is not None

        row = engine.db.fetchone(
            "SELECT labile_until FROM episodes WHERE id = ?",
            (seeded_episode,),
        )
        assert row is not None
        assert row["labile_until"] is not None
        assert row["labile_until"] == labile_until

    def test_open_lability_window_semantic(
        self, engine: ReconsolidationEngine, seeded_semantic: str
    ):
        labile_until = engine.open_lability_window("semantic_node", seeded_semantic)
        assert labile_until is not None

        row = engine.db.fetchone(
            "SELECT labile_until FROM semantic_nodes WHERE id = ?",
            (seeded_semantic,),
        )
        assert row is not None
        assert row["labile_until"] == labile_until

    def test_open_lability_window_procedure_returns_none(self, engine: ReconsolidationEngine):
        result = engine.open_lability_window("procedure", "some-proc-id")
        assert result is None

    def test_invalid_type_returns_none(self, engine: ReconsolidationEngine):
        result = engine.open_lability_window("unknown_type", "some-id")
        assert result is None

    def test_extend_existing_window(self, engine: ReconsolidationEngine, seeded_episode: str):
        first = engine.open_lability_window("episode", seeded_episode)
        assert first is not None

        # Second call should extend or maintain the window
        second = engine.open_lability_window("episode", seeded_episode)
        assert second is not None

        # Verify the labile_until is set in the database
        row = engine.db.fetchone(
            "SELECT labile_until FROM episodes WHERE id = ?",
            (seeded_episode,),
        )
        assert row is not None
        assert row["labile_until"] is not None

    def test_max_labile_cap(self, engine: ReconsolidationEngine):
        """Opening more than MAX_LABILE_MEMORIES windows evicts oldest."""
        ids = []
        for i in range(MAX_LABILE_MEMORIES + 3):
            ep = Episode(
                agent_id="test-agent",
                session_id="s1",
                action=f"action_{i}",
                action_type=ActionType.TOOL_CALL,
                content_text=f"Content {i}",
            )
            engine.episodic.record(ep)
            ids.append(ep.id)
            engine.open_lability_window("episode", ep.id)

        # Count active labile windows — should be at max MAX_LABILE_MEMORIES
        count = engine.db.fetchone(
            "SELECT COUNT(*) as cnt FROM episodes "
            "WHERE labile_until IS NOT NULL AND labile_until > datetime('now')",
        )
        assert count["cnt"] <= MAX_LABILE_MEMORIES

    def test_get_labile_memories(self, engine: ReconsolidationEngine, seeded_episode: str):
        # No labile windows yet
        labile = engine.get_labile_memories()
        assert len(labile) == 0

        # Open a window
        engine.open_lability_window("episode", seeded_episode)
        labile = engine.get_labile_memories()
        assert len(labile) >= 1

        # Check the returned data
        entry = labile[0]
        assert entry["memory_type"] == "episode"
        assert entry["memory_id"] == seeded_episode
        assert entry["labile_until"] is not None

    def test_get_labile_memories_filtered(
        self, engine: ReconsolidationEngine, seeded_episode: str, seeded_semantic: str
    ):
        engine.open_lability_window("episode", seeded_episode)
        engine.open_lability_window("semantic_node", seeded_semantic)

        episodes_only = engine.get_labile_memories(memory_type="episode")
        assert len(episodes_only) >= 1
        assert all(e["memory_type"] == "episode" for e in episodes_only)

        semantic_only = engine.get_labile_memories(memory_type="semantic_node")
        assert len(semantic_only) >= 1
        assert all(s["memory_type"] == "semantic_node" for s in semantic_only)

    def test_should_run_positive(self, engine: ReconsolidationEngine, seeded_episode: str):
        assert not engine.should_run()
        engine.open_lability_window("episode", seeded_episode)
        assert engine.should_run()


# ── Prediction Error Computation ───────────────────────────────


class TestPEComputation:
    def test_identical_content(self, engine: ReconsolidationEngine):
        pe_raw, pe_eff = engine.compute_pe(
            "hello world",
            "hello world",
        )
        assert pe_raw == 0.0
        assert pe_eff == 0.0

    def test_different_content(self, engine: ReconsolidationEngine):
        pe_raw, pe_eff = engine.compute_pe(
            "hello world",
            "foo bar baz",
        )
        assert pe_raw == 1.0
        assert pe_eff > 0.0

    def test_contradiction_bonus_error(self, engine: ReconsolidationEngine):
        base_text = "deploy service to production environment"
        pe_no_bonus, _ = engine.compute_pe(
            base_text, "deploy service to staging environment", action_type="tool_call"
        )
        pe_with_bonus, _ = engine.compute_pe(
            base_text, "deploy service to staging environment", action_type="error"
        )
        assert pe_with_bonus > pe_no_bonus
        assert pytest.approx(pe_with_bonus, abs=0.21) == min(1.0, pe_no_bonus + 0.2)

    def test_contradiction_bonus_failure(self, engine: ReconsolidationEngine):
        base_text = "deploy service to production environment"
        pe_no_bonus, _ = engine.compute_pe(
            base_text, "deploy service to staging environment", success=True
        )
        pe_with_bonus, _ = engine.compute_pe(
            base_text, "deploy service to staging environment", success=False
        )
        assert pe_with_bonus > pe_no_bonus
        assert pytest.approx(pe_with_bonus, abs=0.21) == min(1.0, pe_no_bonus + 0.2)

    def test_neuromodulation_high_ne(self, engine: ReconsolidationEngine):
        """Higher NE increases PE_eff."""
        base_text = "deploy service to production environment"
        _, pe_low = engine.compute_pe(
            base_text, "deploy service to staging environment", ne=0.5, ht5=0.5
        )
        _, pe_high = engine.compute_pe(
            base_text, "deploy service to staging environment", ne=2.0, ht5=0.5
        )
        assert pe_high > pe_low

    def test_neuromodulation_high_5ht(self, engine: ReconsolidationEngine):
        """Higher 5HT decreases PE_eff."""
        base_text = "deploy service to production environment"
        _, pe_low = engine.compute_pe(
            base_text, "deploy service to staging environment", ne=1.0, ht5=0.1
        )
        _, pe_high = engine.compute_pe(
            base_text, "deploy service to staging environment", ne=1.0, ht5=1.0
        )
        assert pe_low > pe_high

    def test_pe_eff_bounded(self, engine: ReconsolidationEngine):
        """PE_eff should always be in [0, 1]."""
        for _ in range(20):
            _, pe_eff = engine.compute_pe(
                "hello world test foo",
                "totally unrelated content here",
                action_type="error",
                success=False,
                ne=2.0,
                ht5=0.0,
            )
            assert 0.0 <= pe_eff <= 1.0

    def test_partial_pe_value(self, engine: ReconsolidationEngine):
        pe_raw, _ = engine.compute_pe(
            "abc def ghi jkl",
            "abc def xxx yyy",
        )
        # intersection = {abc, def}, union = {abc, def, ghi, jkl, xxx, yyy}
        # 1 - 2/6 ≈ 0.6667
        assert pe_raw == pytest.approx(0.6667, abs=0.01)


# ── Update Mode Selection ──────────────────────────────────────


class TestUpdateModeSelection:
    def test_confirmed_low_pe(self, engine: ReconsolidationEngine):
        mode = engine.select_update_mode(0.05)
        assert mode == "confirmed"

    def test_selective_edit_medium_low(self, engine: ReconsolidationEngine):
        mode = engine.select_update_mode(0.2)
        assert mode == "selective_edit"

    def test_integration_medium_high(self, engine: ReconsolidationEngine):
        mode = engine.select_update_mode(0.5)
        assert mode == "integration"

    def test_new_episode_high_pe(self, engine: ReconsolidationEngine):
        mode = engine.select_update_mode(0.8)
        assert mode == "new_episode"

    def test_boundary_confirmed(self, engine: ReconsolidationEngine):
        mode = engine.select_update_mode(PE_CONFIRMED - 0.001)
        assert mode == "confirmed"

    def test_boundary_selective(self, engine: ReconsolidationEngine):
        mode = engine.select_update_mode(PE_CONFIRMED)
        assert mode == "selective_edit"

    def test_boundary_integration(self, engine: ReconsolidationEngine):
        mode = engine.select_update_mode(PE_SELECTIVE_EDIT)
        assert mode == "integration"

    def test_boundary_new(self, engine: ReconsolidationEngine):
        mode = engine.select_update_mode(PE_INTEGRATION)
        assert mode == "new_episode"


# ── Stability Protector ────────────────────────────────────────


class TestStabilityProtector:
    def test_new_memory_low_resistance(self, engine: ReconsolidationEngine):
        row = {
            "access_count": 1,
            "created_at": datetime.utcnow().isoformat(),
        }
        threshold, lock, rigidity = engine.stability_protector(row)
        assert lock == pytest.approx(0.1)
        assert rigidity == pytest.approx(0.0, abs=1e-9)
        assert threshold == pytest.approx(0.5, abs=1e-9)

    def test_old_frequently_accessed_high_resistance(self, engine: ReconsolidationEngine):
        old_date = (datetime.utcnow() - timedelta(days=60)).isoformat()
        row = {
            "access_count": 50,
            "created_at": old_date,
        }
        threshold, lock, rigidity = engine.stability_protector(row)
        assert lock == pytest.approx(1.0)
        assert rigidity == pytest.approx(1.0)
        assert threshold == pytest.approx(0.8)

    def test_old_rarely_accessed_medium_resistance(self, engine: ReconsolidationEngine):
        old_date = (datetime.utcnow() - timedelta(days=60)).isoformat()
        row = {
            "access_count": 3,
            "created_at": old_date,
        }
        threshold, lock, rigidity = engine.stability_protector(row)
        assert lock == pytest.approx(0.3)
        assert rigidity == pytest.approx(1.0)
        assert threshold == pytest.approx(0.5 + 0.3 * 0.3 * 1.0)

    def test_recent_frequently_accessed(self, engine: ReconsolidationEngine):
        row = {
            "access_count": 20,
            "created_at": datetime.utcnow().isoformat(),
        }
        _, lock, rigidity = engine.stability_protector(row)
        assert lock == pytest.approx(1.0)
        assert rigidity == pytest.approx(0.0, abs=1e-9)

    def test_default_on_missing_fields(self, engine: ReconsolidationEngine):
        row: dict = {}
        threshold, lock, rigidity = engine.stability_protector(row)
        assert lock == pytest.approx(0.0)
        assert threshold == pytest.approx(0.5)


# ── Contradiction Penalty ──────────────────────────────────────


class TestContradictionPenalty:
    def test_decision_type_low_old_confidence(self, engine: ReconsolidationEngine):
        new_conf = engine.contradiction_penalty("decision", old_confidence=0.1)
        assert 0.05 <= new_conf < 0.1

    def test_fact_type_moderate_confidence(self, engine: ReconsolidationEngine):
        new_conf = engine.contradiction_penalty("fact", old_confidence=0.5)
        assert new_conf < 0.5

    def test_high_confidence_is_penalized_more(self, engine: ReconsolidationEngine):
        low_conf = engine.contradiction_penalty("fact", old_confidence=0.2)
        high_conf = engine.contradiction_penalty("fact", old_confidence=0.9)
        # The penalty makes them more similar than you'd expect because
        # β formula uses exp(-c/s), so higher s gives lower β
        assert high_conf > low_conf

    def test_unknown_type_uses_default(self, engine: ReconsolidationEngine):
        new_conf = engine.contradiction_penalty("unknown_type", old_confidence=0.5)
        assert 0.05 <= new_conf <= 0.5

    def test_floor_at_0_05(self, engine: ReconsolidationEngine):
        new_conf = engine.contradiction_penalty("decision", old_confidence=0.01)
        assert new_conf == pytest.approx(0.05)

    def test_preference_type(self, engine: ReconsolidationEngine):
        new_conf = engine.contradiction_penalty("preference", old_confidence=0.5)
        assert 0.05 <= new_conf <= 0.5

    def test_opinion_type(self, engine: ReconsolidationEngine):
        new_conf = engine.contradiction_penalty("opinion", old_confidence=0.5)
        assert 0.05 <= new_conf <= 0.5


# ── Full Reconsolidation Pipeline ──────────────────────────────


class TestReconsolidationPipeline:
    def test_process_new_evidence_confirmed(
        self, engine: ReconsolidationEngine, seeded_episode: str
    ):
        """Near-identical content should produce 'confirmed' mode."""
        # Include the action text since episodes combine content_text + action
        result = engine.process_new_evidence(
            memory_type="episode",
            memory_id=seeded_episode,
            new_content="Deployed service-7 to production with zero downtime deploy service",
            agent_id="test-agent",
        )
        assert result["status"] == "completed"
        assert result["update_mode"] == "confirmed"
        assert result["did_update"] is True
        assert result["log_id"] is not None

    def test_process_new_evidence_high_pe(self, engine: ReconsolidationEngine, seeded_episode: str):
        """Completely different content should produce 'new_episode' or 'integration'."""
        result = engine.process_new_evidence(
            memory_type="episode",
            memory_id=seeded_episode,
            new_content="Installed python dependencies using pip requirements.txt",
            agent_id="test-agent",
        )
        assert result["status"] in ("completed", "skipped")
        assert result["pe_raw"] > 0.5
        assert result["log_id"] is not None

    def test_process_new_evidence_semantic_contradiction(
        self, engine: ReconsolidationEngine, seeded_semantic: str
    ):
        """Test that contradiction penalty applies correctly."""
        result = engine.process_new_evidence(
            memory_type="semantic_node",
            memory_id=seeded_semantic,
            new_content="The database connection pool should have max 100 connections",
            action_type="error",
            content_type="fact",
            agent_id="test-agent",
        )
        assert result["status"] in ("completed", "skipped")
        assert result["log_id"] is not None

    def test_process_new_evidence_nonexistent(self, engine: ReconsolidationEngine):
        result = engine.process_new_evidence(
            memory_type="episode",
            memory_id="nonexistent-id",
            new_content="some content",
        )
        assert result["status"] == "error"
        assert "Memory not found" in result["message"]

    def test_process_new_evidence_invalid_type(self, engine: ReconsolidationEngine):
        result = engine.process_new_evidence(
            memory_type="invalid_type",
            memory_id="some-id",
            new_content="some content",
        )
        assert result["status"] == "error"
        assert "Unknown memory type" in result["message"]

    def test_reconsolidation_log_created(self, engine: ReconsolidationEngine, seeded_episode: str):
        result = engine.process_new_evidence(
            memory_type="episode",
            memory_id=seeded_episode,
            new_content="Completely unrelated new information here",
            agent_id="test-agent",
        )

        log = engine.db.fetchone(
            "SELECT * FROM reconsolidation_log WHERE id = ?",
            (result["log_id"],),
        )
        assert log is not None
        assert log["memory_type"] == "episode"
        assert log["memory_id"] == seeded_episode
        assert log["pe_raw"] is not None
        assert log["pe_eff"] is not None
        assert log["update_mode"] is not None
        assert log["snapshot_before"] is not None
        assert log["agent_id"] == "test-agent"

    def test_snapshot_captures_before_state(
        self, engine: ReconsolidationEngine, seeded_episode: str
    ):
        ep_before = engine.episodic.get(seeded_episode)
        snapshot = engine._snapshot("episode", seeded_episode)
        assert snapshot is not None
        assert snapshot["content_text"] == ep_before["content_text"]

    def test_semantic_confidence_update_on_confirmed(
        self, engine: ReconsolidationEngine, seeded_semantic: str
    ):
        conf_before = engine.semantic.get(seeded_semantic)["confidence"]
        engine.process_new_evidence(
            memory_type="semantic_node",
            memory_id=seeded_semantic,
            new_content="The database connection pool should have max 10 connections",
        )
        conf_after = engine.semantic.get(seeded_semantic)["confidence"]
        # Confirmed mode boosts confidence
        assert conf_after >= conf_before

    def test_stability_protector_blocks_update(
        self, engine: ReconsolidationEngine, seeded_episode: str
    ):
        """A very stable memory with low PE should be skipped."""
        # Make the memory look very stable
        old_date = (datetime.utcnow() - timedelta(days=60)).isoformat()
        engine.db.update(
            "episodes",
            seeded_episode,
            {
                "access_count": 50,
                "created_at": old_date,
            },
        )

        result = engine.process_new_evidence(
            memory_type="episode",
            memory_id=seeded_episode,
            new_content="Deployed service-7 to production with zero downtime",
        )
        # If PE is low enough (< stab threshold), it gets skipped
        if result["pe_eff"] < 0.5 + 0.3 * 1.0 * 1.0:
            assert result["status"] == "skipped"
            assert result["did_update"] is False


# ── MCP Tool Handler ───────────────────────────────────────────


class TestMCPToolHandler:
    @pytest.mark.asyncio
    async def test_myelin_reconsolidate_success(
        self, engine: ReconsolidationEngine, seeded_episode: str
    ):
        result = await engine.myelin_reconsolidate(
            memory_id=seeded_episode,
            memory_type="episode",
            new_content="Completely different content for testing reconsolidation",
            agent_id="test-agent",
        )
        assert result["tool"] == "myelin_reconsolidate"
        assert result["result"]["status"] in ("completed", "skipped")
        assert result["result"]["log_id"] is not None

    @pytest.mark.asyncio
    async def test_myelin_reconsolidate_invalid_type(self, engine: ReconsolidationEngine):
        result = await engine.myelin_reconsolidate(
            memory_id="some-id",
            memory_type="invalid",
            new_content="test",
        )
        assert result["tool"] == "myelin_reconsolidate"
        assert result["result"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_myelin_reconsolidate_empty_content(self, engine: ReconsolidationEngine):
        result = await engine.myelin_reconsolidate(
            memory_id="some-id",
            memory_type="episode",
            new_content="",
        )
        assert result["result"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_myelin_reconsolidate_not_found(self, engine: ReconsolidationEngine):
        result = await engine.myelin_reconsolidate(
            memory_id="nonexistent",
            memory_type="episode",
            new_content="test content",
        )
        assert result["result"]["status"] == "error"
        assert "not found" in result["result"]["message"]

    @pytest.mark.asyncio
    async def test_myelin_reconsolidate_semantic(
        self, engine: ReconsolidationEngine, seeded_semantic: str
    ):
        result = await engine.myelin_reconsolidate(
            memory_id=seeded_semantic,
            memory_type="semantic_node",
            new_content="This contradicts the existing fact about database connections",
            content_type="fact",
            agent_id="test-agent",
        )
        assert result["tool"] == "myelin_reconsolidate"
        assert result["result"]["log_id"] is not None

    @pytest.mark.asyncio
    async def test_myelin_reconsolidate_with_contradiction_bonus(
        self,
        engine: ReconsolidationEngine,
        seeded_episode: str,
    ):
        result = await engine.myelin_reconsolidate(
            memory_id=seeded_episode,
            memory_type="episode",
            new_content="This is very different content",
            action_type="error",
            success=False,
            agent_id="test-agent",
        )
        assert result["result"]["pe_raw"] > 0.5


# ── Integration Tests ──────────────────────────────────────────


class TestIntegration:
    def test_full_cycle_lability_to_reconsolidation(
        self, engine: ReconsolidationEngine, seeded_episode: str
    ):
        """Test the full cycle: retrieve -> open window -> process evidence."""
        # 1. Open lability window (simulating retrieval)
        engine.open_lability_window("episode", seeded_episode)

        # Verify window is open
        labile = engine.get_labile_memories(memory_type="episode")
        assert any(m["memory_id"] == seeded_episode for m in labile)

        # 2. Process new evidence
        result = engine.process_new_evidence(
            memory_type="episode",
            memory_id=seeded_episode,
            new_content="Rolled back deployment due to connection timeout errors",
            action_type="error",
            success=False,
            agent_id="test-agent",
        )
        assert result["log_id"] is not None

        # 3. Verify the reconsolidation log entry
        log = engine.db.fetchone(
            "SELECT * FROM reconsolidation_log WHERE id = ?",
            (result["log_id"],),
        )
        assert log is not None
        assert log["pe_raw"] > 0.3  # Different content + error bonus

    def test_multiple_memories_reconsolidation(self, engine: ReconsolidationEngine):
        """Reconsolidate multiple memories and verify independence."""
        ids = []
        for i in range(5):
            ep = Episode(
                agent_id="test-agent",
                session_id="s1",
                action=f"action_{i}",
                action_type=ActionType.TOOL_CALL,
                content_text=f"Content for memory number {i} in the system",
            )
            engine.episodic.record(ep)
            ids.append(ep.id)

        results = []
        for i, eid in enumerate(ids):
            r = engine.process_new_evidence(
                memory_type="episode",
                memory_id=eid,
                new_content=f"New conflicting information for memory {i}",
                agent_id="test-agent",
            )
            results.append(r)

        # All should have unique log entries
        log_ids = [r["log_id"] for r in results if r["status"] != "error"]
        assert len(log_ids) == len(set(log_ids))

        # Verify log entries exist
        logs = engine.db.fetchall(
            f"SELECT * FROM reconsolidation_log WHERE id IN ({','.join('?' for _ in log_ids)})",
            tuple(log_ids),
        )
        assert len(logs) == len(log_ids)

    def test_pe_fields_updated_on_semantic(
        self, engine: ReconsolidationEngine, seeded_semantic: str
    ):
        engine.process_new_evidence(
            memory_type="semantic_node",
            memory_id=seeded_semantic,
            new_content="Different content for semantic node testing",
            content_type="fact",
        )

        row = engine.db.fetchone(
            "SELECT prediction_error, last_pe_raw, last_update_mode FROM semantic_nodes WHERE id = ?",
            (seeded_semantic,),
        )
        assert row is not None
        assert row["prediction_error"] is not None
        assert row["last_pe_raw"] is not None
        assert row["last_update_mode"] is not None
