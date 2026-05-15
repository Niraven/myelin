"""Tests for LLMConsolidator: pattern-aware semantic summarization.

Verifies:
- Episode clustering by domain + content similarity
- Entity extraction from content_text
- Action type frequency counting
- Success/failure pattern analysis
- Informative summary generation (not placeholder)
- Confidence calibration based on cluster quality
"""

from __future__ import annotations

import pytest

from myelin.cognitive.llm_consolidator import (
    JACCARD_SIMILARITY_THRESHOLD,
    LLMConsolidator,
    _action_sequence_summary,
    _compute_cluster_confidence,
    _extract_action_type,
    _extract_entities_from_text,
    _jaccard_similarity,
    _success_analysis,
)
from myelin.core.database import Database
from myelin.core.models import ActionType, Episode, NodeType
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.semantic import SemanticMemory


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def consolidator(tmp_db: Database) -> LLMConsolidator:
    episodic = EpisodicMemory(tmp_db)
    semantic = SemanticMemory(tmp_db)
    return LLMConsolidator(tmp_db, episodic, semantic)


@pytest.fixture
def seeded_deployment_episodes(consolidator: LLMConsolidator) -> list[dict]:
    """Seed 6 episodes in the 'deployment' domain."""
    eps = [
        Episode(
            agent_id="test-agent",
            session_id="s1",
            action="deploy service to production",
            action_type=ActionType.TOOL_CALL,
            content_text="Deployed my-app v2.1 to production using kubectl rolling update",
            success=True,
            domain="deployment",
        ),
        Episode(
            agent_id="test-agent",
            session_id="s1",
            action="verify deployment health",
            action_type=ActionType.TOOL_CALL,
            content_text="Checked deployment status for my-app in production namespace, all pods running",
            success=True,
            domain="deployment",
        ),
        Episode(
            agent_id="test-agent",
            session_id="s1",
            action="deploy canary release",
            action_type=ActionType.TOOL_CALL,
            content_text="Deployed canary v2.2 to staging with 10% traffic using istio",
            success=True,
            domain="deployment",
        ),
        Episode(
            agent_id="test-agent",
            session_id="s2",
            action="roll back failed deployment",
            action_type=ActionType.TOOL_CALL,
            content_text="Rolled back my-app v2.1 to v2.0 after crash loop detected in production",
            success=True,
            domain="deployment",
        ),
        Episode(
            agent_id="test-agent",
            session_id="s2",
            action="run integration tests",
            action_type=ActionType.TOOL_CALL,
            content_text="Ran pytest integration test suite for my-app — 45 passed, 3 failed",
            success=False,
            domain="testing",
        ),
        Episode(
            agent_id="test-agent",
            session_id="s2",
            action="fix failing test",
            action_type=ActionType.TOOL_CALL,
            content_text="Debugged test_auth_flow failure in my-app — fixed async timeout issue",
            success=True,
            domain="testing",
        ),
    ]
    ids = []
    for ep in eps:
        ids.append(consolidator.episodic.record(ep))
    # Return full row data (filter out None entries)
    result = []
    for eid in ids:
        row = consolidator.episodic.get(eid)
        if row:
            result.append(row)
    return result


# ── Unit Tests ─────────────────────────────────────────────────────


class TestJaccardSimilarity:
    def test_identical_texts(self):
        assert _jaccard_similarity("deploy service", "deploy service") == 1.0

    def test_completely_different_texts(self):
        sim = _jaccard_similarity("deploy service", "build docker image")
        assert sim == 0.0  # No overlapping tokens

    def test_partial_overlap(self):
        sim = _jaccard_similarity(
            "deploy my-app to production using kubectl",
            "deploy my-app to staging using helm",
        )
        # Should share: deploy, my-app, to, using
        assert 0.4 <= sim <= 0.8

    def test_empty_strings(self):
        assert _jaccard_similarity("", "") == 0.0
        assert _jaccard_similarity("test", "") == 0.0


class TestExtractActionType:
    def test_running_prefix(self):
        assert _extract_action_type("running npm test") == "run"

    def test_deploying_prefix(self):
        assert _extract_action_type("deploying service to production") == "deploy"

    def test_testing_prefix(self):
        assert _extract_action_type("testing authentication flow") == "test"

    def test_configuring_prefix(self):
        assert _extract_action_type("configuring nginx reverse proxy") == "configure"

    def test_generic_action(self):
        assert _extract_action_type("unknown action") == "unknown"

    def test_past_tense(self):
        assert _extract_action_type("checked deployment status") == "check"


class TestExtractEntitiesFromText:
    def test_extracts_capitalized_words(self):
        entities = _extract_entities_from_text("Deployed MyApp to Production with Kubernetes")
        assert "myapp" in entities
        assert "kubernetes" in entities

    def test_extracts_snake_case(self):
        entities = _extract_entities_from_text("Running npm test for my_package_name")
        assert "my_package_name" in entities

    def test_extracts_known_tools(self):
        entities = _extract_entities_from_text("Deployed with docker and kubernetes")
        assert "docker" in entities
        assert "kubernetes" in entities

    def test_skips_short_words(self):
        entities = _extract_entities_from_text("a an the")
        assert all(len(e) >= 3 for e in entities) or len(entities) == 0


class TestActionSequenceSummary:
    def test_single_action_type(self):
        summary = _action_sequence_summary(["deploy", "deploy", "deploy"])
        assert "deploy" in summary
        assert "3x" in summary

    def test_two_action_types(self):
        summary = _action_sequence_summary(["deploy", "deploy", "test"])
        assert "deploy" in summary
        assert "test" in summary

    def test_many_actions(self):
        actions = ["deploy", "test", "build", "config", "deploy", "fix"]
        summary = _action_sequence_summary(actions)
        assert "varied" in summary or "deploy" in summary

    def test_empty_list(self):
        assert "no specific pattern" in _action_sequence_summary([])


class TestSuccessAnalysis:
    def test_all_success(self):
        analysis = _success_analysis(5, 5, "deploy")
        assert "100%" in analysis or "All 5" in analysis
        assert "reliable" in analysis

    def test_high_success(self):
        analysis = _success_analysis(8, 10, "deploy")
        assert "80%" in analysis or "80" in analysis

    def test_all_failure(self):
        analysis = _success_analysis(0, 5, "deploy")
        assert "unreliable" in analysis

    def test_partial_mixed(self):
        analysis = _success_analysis(3, 8, "deploy")
        assert "37%" in analysis or "37" in analysis or "38%" in analysis


class TestComputeClusterConfidence:
    def test_large_reliable_cluster(self):
        conf = _compute_cluster_confidence(10, 0.9, 5, 3)
        assert 0.4 <= conf <= 0.95

    def test_small_unreliable_cluster(self):
        conf = _compute_cluster_confidence(2, 0.0, 1, 1)
        assert conf >= 0.4  # Min floor

    def test_medium_cluster(self):
        conf = _compute_cluster_confidence(5, 0.6, 3, 2)
        assert 0.4 <= conf <= 0.95


# ── Integration Tests ──────────────────────────────────────────────


class TestClustering:
    def test_cluster_by_domain_separates_correctly(self, consolidator, seeded_deployment_episodes):
        """Episodes from different domains should end up in separate clusters."""
        clusters = consolidator._cluster_by_domain(seeded_deployment_episodes)
        assert "deployment" in clusters
        assert "testing" in clusters
        assert len(clusters["deployment"]) == 4
        assert len(clusters["testing"]) == 2

    def test_cluster_by_content_similarity(self, consolidator, seeded_deployment_episodes):
        """Within a domain, similar content should cluster together."""
        # All deployment episodes share "my-app" and "deploy" keywords
        deployment_eps = [
            ep for ep in seeded_deployment_episodes
            if ep.get("domain") == "deployment"
        ]
        clusters = consolidator._cluster_by_content(deployment_eps)
        # Should produce at least one meaningful cluster
        non_singleton = [c for c in clusters if len(c) >= 2]
        assert len(non_singleton) >= 1


class TestBuildSummary:
    def test_summary_contains_domain_and_count(self, consolidator, seeded_deployment_episodes):
        deployment_eps = [
            ep for ep in seeded_deployment_episodes
            if ep.get("domain") == "deployment"
        ]
        summary, metadata = consolidator._build_summary(deployment_eps, "deployment")
        assert "4 observations" in summary
        assert "deployment" in summary

    def test_summary_includes_entities(self, consolidator, seeded_deployment_episodes):
        deployment_eps = [
            ep for ep in seeded_deployment_episodes
            if ep.get("domain") == "deployment"
        ]
        summary, metadata = consolidator._build_summary(deployment_eps, "deployment")
        # my-app should be extracted as an entity
        entities = metadata.get("entities", [])
        combined = " ".join(entities).lower()
        assert "my-app" in summary.lower() or "myapp" in combined or "my_app" in combined

    def test_summary_includes_success_rate(self, consolidator, seeded_deployment_episodes):
        deployment_eps = [
            ep for ep in seeded_deployment_episodes
            if ep.get("domain") == "deployment"
        ]
        summary, metadata = consolidator._build_summary(deployment_eps, "deployment")
        assert metadata["success_rate"] >= 0.75  # 3/4 successful

    def test_summary_includes_action_pattern(self, consolidator, seeded_deployment_episodes):
        deployment_eps = [
            ep for ep in seeded_deployment_episodes
            if ep.get("domain") == "deployment"
        ]
        summary, metadata = consolidator._build_summary(deployment_eps, "deployment")
        # Should mention deploy
        assert "deploy" in metadata.get("most_common_action", "")
        assert "pattern" in summary.lower() or "followed by" in summary.lower()

    def test_failure_context_appears_in_mixed_cluster(self, consolidator):
        """A cluster with failures should include failure context."""
        test_eps = [
            Episode(
                agent_id="test-agent", session_id="s1",
                action="run tests", action_type=ActionType.TOOL_CALL,
                content_text="Ran test suite for auth module — 5 tests failed with timeout errors",
                success=False, domain="testing",
            ),
            Episode(
                agent_id="test-agent", session_id="s1",
                action="run integration tests", action_type=ActionType.TOOL_CALL,
                content_text="Ran integration tests for payment service — all 12 passed",
                success=True, domain="testing",
            ),
        ]
        rows = [consolidator.episodic.record(ep) and consolidator.episodic.get(ep.id) for ep in test_eps]
        summary, metadata = consolidator._build_summary(rows, "testing")
        # Should include failure context
        assert "Failure observed" in summary or "fail" in summary.lower()


class TestFullExecute:
    async def test_execute_processes_unconsolidated(self, consolidator, seeded_deployment_episodes):
        """Full execute() should mark episodes as consolidated."""
        result = await consolidator.execute()
        assert result["processed"] >= 3  # 3 deployment episodes cluster together
        assert result["created"] >= 1  # At least one semantic node

        # Verify episodes are marked consolidated
        count = consolidator.episodic.db.fetchone(
            "SELECT COUNT(*) as cnt FROM episodes WHERE consolidated = 0"
        )
        assert count["cnt"] == 3  # 3 unclustered episodes remain unconsolidated

    async def test_execute_creates_semantic_nodes(self, consolidator, seeded_deployment_episodes):
        """Full execute() should create semantic nodes."""
        result = await consolidator.execute()
        # Count semantic nodes
        count = consolidator.semantic.count(NodeType.FACT)
        assert count > 0
