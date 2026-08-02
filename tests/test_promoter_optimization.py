"""Test promoter optimization: correctness and performance regression tests.

Covers:
1. _has_existing_procedure — single-query vs N+1 correctness
2. Promoter end-to-end with deterministic fixture
3. Performance regression: 1000 episodes under 5s
4. NREM _cluster_episodes batch update correctness
5. ImportanceComputer.persist batch update correctness
"""

import tempfile
import time
from pathlib import Path

import pytest

from myelin.cognitive.nrem_sleep import NREMPhase
from myelin.cognitive.promoter import Promoter
from myelin.cognitive.sleep import ImportanceComputer
from myelin.core.database import Database
from myelin.core.models import ActionType, Episode, Procedure, ProcedureStep, StepType
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory

# ── Helpers ─────────────────────────────────────────────────────


def _make_episodes(db: Database, count: int, sessions: int) -> list[dict]:
    """Create a deterministic set of episodes."""
    episodic = EpisodicMemory(db)
    actions = ["git pull", "npm test", "docker build", "docker push", "kubectl apply"]
    created = []
    for i in range(sessions):
        sid = f"ses_{i}"
        for action in actions:
            ep = Episode(
                agent_id="test-agent",
                session_id=sid,
                action=action,
                action_type=ActionType.TOOL_CALL,
                content_text=f"{action} for workflow-{i % 3}",
                success=True,
                domain="testing",
            )
            episodic.record(ep)
            created.append(ep)
    db.commit()
    return created


# ── Test: _has_existing_procedure single-query optimization ───


class TestHasExistingProcedure:
    def test_no_procedures(self, tmp_db):
        """When no procedures exist, should return False."""
        prom = Promoter(tmp_db, EpisodicMemory(tmp_db), ProceduralMemory(tmp_db))
        assert prom._has_existing_procedure(["ep_1", "ep_2"]) is False

    def test_empty_episode_list(self, tmp_db):
        prom = Promoter(tmp_db, EpisodicMemory(tmp_db), ProceduralMemory(tmp_db))
        assert prom._has_existing_procedure([]) is False

    def test_finds_matching_procedure(self, tmp_db, procedural):
        """Should return True when an episode ID appears in an existing procedure."""
        # Create a procedure with source_episodes containing "ep_match_1"
        proc = Procedure(
            name="test_proc",
            trigger_pattern="test trigger",
            steps=[ProcedureStep(order=0, description="step1", step_type=StepType.CORE)],
            source_agent="test-agent",
            source_episodes=["ep_match_1", "ep_match_2"],
        )
        procedural.store(proc)

        prom = Promoter(tmp_db, EpisodicMemory(tmp_db), procedural)
        assert prom._has_existing_procedure(["ep_match_1"]) is True
        assert prom._has_existing_procedure(["ep_match_1", "ep_other"]) is True

    def test_no_false_positive(self, tmp_db, procedural):
        """Should return False when episode IDs don't match any procedure."""
        proc = Procedure(
            name="test_proc",
            trigger_pattern="test trigger",
            steps=[ProcedureStep(order=0, description="step1", step_type=StepType.CORE)],
            source_agent="test-agent",
            source_episodes=["existing_id"],
        )
        procedural.store(proc)

        prom = Promoter(tmp_db, EpisodicMemory(tmp_db), procedural)
        assert prom._has_existing_procedure(["non_existent_id"]) is False


# ── Test: Promoter end-to-end ─────────────────────────────────


class TestPromoterEndToEnd:
    def test_promoter_creates_procedure(self, tmp_db, episodic):
        """Promoter should create a procedure from repeated action patterns."""
        procedural = ProceduralMemory(tmp_db)
        prom = Promoter(tmp_db, episodic, procedural)

        # Insert 12 episodes across 3 sessions with the same pattern
        for i in range(3):
            for action in ["git pull", "npm test", "docker build", "docker push"]:
                ep = Episode(
                    agent_id="test-agent",
                    session_id=f"end_to_end_ses_{i}",
                    action=action,
                    action_type=ActionType.TOOL_CALL,
                    content_text=f"{action} for deployment",
                    success=True,
                    domain="testing",
                )
                episodic.record(ep)
        tmp_db.commit()

        import asyncio

        result = asyncio.run(prom.execute())
        assert result.get("created", 0) >= 1, f"Expected at least 1 procedure, got {result}"

    def test_promoter_skips_insufficient_data(self, tmp_db, episodic):
        """Promoter should skip when there aren't enough sessions."""
        procedural = ProceduralMemory(tmp_db)
        prom = Promoter(tmp_db, episodic, procedural)

        # Insert just 1 session
        for action in ["git pull", "npm test"]:
            ep = Episode(
                agent_id="test-agent",
                session_id="single_session",
                action=action,
                action_type=ActionType.TOOL_CALL,
                content_text=action,
                success=True,
                domain="testing",
            )
            episodic.record(ep)

        import asyncio

        result = asyncio.run(prom.execute())
        # Should skip — only 1 session (too few to cluster)
        assert result.get("processed", 0) == 0 and result.get("created", 0) == 0


# ── Test: NREM batch cluster update correctness ──────────────


class TestNremBatchUpdate:
    def test_batch_cluster_update(self, tmp_db, episodic):
        """NREM _cluster_episodes should update cluster_id on all episodes."""
        nrem = NREMPhase(tmp_db)

        # Create episodes
        for i in range(3):
            ep = Episode(
                agent_id="test-agent",
                session_id=f"nrem_ses_{i}",
                action="test_action",
                action_type=ActionType.TOOL_CALL,
                content_text=f"testing batch update {i}",
                success=True,
                domain="testing",
            )
            episodic.record(ep)
        tmp_db.commit()

        # Run clustering
        recent_eps = episodic.get_recent(limit=500)
        nrem._cluster_episodes(recent_eps)

        # All episodes should have a cluster_id set
        for ep in recent_eps:
            updated = tmp_db.fetchone("SELECT cluster_id FROM episodes WHERE id = ?", (ep["id"],))
            assert updated and updated.get("cluster_id"), (
                f"Episode {ep['id']} should have cluster_id"
            )


# ── Test: ImportanceComputer.persist batch update correctness ─


class TestImportancePersist:
    def test_batch_importance_update(self, tmp_db, episodic):
        """ImportanceComputer.persist should correctly update all episodes."""

        # Create episodes
        for i in range(10):
            ep = Episode(
                agent_id="test-agent",
                session_id=f"imp_ses_{i}",
                action="action",
                action_type=ActionType.TOOL_CALL,
                content_text=f"test {i}",
                success=True,
                domain="testing",
            )
            episodic.record(ep)
        tmp_db.commit()

        # Get episodes and compute scores
        all_eps = tmp_db.fetchall("SELECT * FROM episodes")

        # Test ImportanceComputer with just domain grouping
        episode_scores = {}
        for ep in all_eps:
            episode_scores[ep["id"]] = 0.75

        computer = ImportanceComputer()
        count = computer.persist(tmp_db, episode_scores)
        assert count == len(all_eps)

        # Verify
        for ep in all_eps:
            row = tmp_db.fetchone("SELECT importance_score FROM episodes WHERE id = ?", (ep["id"],))
            assert row and row["importance_score"] == 0.75

    def test_empty_scores(self, tmp_db):
        computer = ImportanceComputer()
        assert computer.persist(tmp_db, {}) == 0


# ── Performance regression: 1000 episodes under 5s ────────────


@pytest.mark.slow
class TestPromoterPerformanceRegression:
    """Performance regression: full session-end loop on 1000 episodes must complete under 5s.

    Marked @pytest.mark.slow. Run with: pytest tests/ -q -k PerformanceRegression
    """

    def test_full_loop_under_5s(self):
        """Full cognitive loop on 1000 deterministic episodes must complete under 5s."""
        import asyncio

        from myelin.session import Session

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "perf_test.db"
            db = Database(db_path, enable_vec=False)
            _ = db.conn

            # Create 1000 episodes across 200 sessions
            episodic = EpisodicMemory(db)
            actions = ["git pull", "npm test", "docker build", "docker push", "kubectl apply"]
            for session_idx in range(200):
                for action in actions:
                    ep = Episode(
                        agent_id="bench-agent",
                        session_id=f"perf_ses_{session_idx}",
                        action=action,
                        action_type=ActionType.TOOL_CALL,
                        content_text=f"{action} for svc-{session_idx % 50}",
                        success=True,
                        domain="deployment",
                    )
                    episodic.record(ep)
            db.commit()

            # Run full session end
            session = Session(db, agent_id="bench-agent", session_id="perf-final")
            session.orchestrator._write_count = 100

            t0 = time.perf_counter()
            asyncio.run(session.end())
            elapsed = time.perf_counter() - t0

            db.close()

            assert elapsed < 5.0, (
                f"Full cognitive loop took {elapsed:.3f}s, expected < 5.0s for 1000 episodes"
            )
