"""Tests for LLMReflector: multi-level pattern-aware reflection.

Verifies:
- Level 1: Domain-level observation reflections with trend analysis
- Level 2: Cross-domain synthesis (action chaining across domains)
- Level 3: Meta-reflection on agent behavior patterns over time
- Informative reflection text (not placeholder)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from myelin.cognitive.llm_reflector import (
    LLMReflector,
    _compute_event_density,
    _compute_trend,
    _extract_action_from_content,
    _extract_entities_from_content,
    _insight_for_domain,
)
from myelin.core.database import Database
from myelin.core.models import NodeType, SemanticNode, SourceType
from myelin.memory.semantic import SemanticMemory


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def reflector(tmp_db: Database) -> LLMReflector:
    semantic = SemanticMemory(tmp_db)
    return LLMReflector(tmp_db, semantic)


def _seed_facts(
    semantic: SemanticMemory,
    facts: list[tuple[str, str, float]],
    domain: str,
) -> list[str]:
    """Seed semantic nodes with (content, tag, confidence) tuples."""
    ids = []
    for content, tags_str, confidence in facts:
        node = SemanticNode(
            node_type=NodeType.FACT,
            content=content,
            source_type=SourceType.OBSERVATION,
            source_ids=["mock-src"],
            domain=domain,
            confidence=confidence,
            tags=tags_str.split(",") if tags_str else [],
        )
        ids.append(semantic.store(node))
    return ids


@pytest.fixture
def seeded_deployment_facts(reflector: LLMReflector) -> list[str]:
    """Seed 5 semantic facts in the 'deployment' domain."""
    return _seed_facts(
        reflector.semantic,
        [
            ("Deployed my-app v2.1 to production successfully, all pods running", "deployment,kubectl", 0.8),
            ("Deployed canary v2.2 to staging with 10% traffic using istio", "deployment,istio", 0.75),
            ("Rolled back my-app v2.1 to v2.0 after crash loop detected", "deployment,rollback", 0.7),
            ("Verified deployment health — all 6 replicas are healthy and serving traffic", "deployment,monitoring", 0.85),
            ("Configured kubernetes ingress for my-app with TLS termination", "deployment,networking", 0.65),
        ],
        domain="deployment",
    )


@pytest.fixture
def seeded_testing_facts(reflector: LLMReflector) -> list[str]:
    """Seed 4 semantic facts in the 'testing' domain."""
    return _seed_facts(
        reflector.semantic,
        [
            ("Ran pytest integration test suite — 45 passed, 3 failed due to timeout issues", "testing,pytest", 0.6),
            ("Debugged test_auth_flow failure — fixed async timeout, all tests now passing", "testing,debug", 0.7),
            ("Ran unit tests for payment module — 22/22 passed with 95% coverage", "testing,coverage", 0.8),
            ("Executed load test suite — 1000 req/s sustained, p99 latency 45ms under threshold", "testing,performance", 0.75),
        ],
        domain="testing",
    )


# ── Unit Tests ─────────────────────────────────────────────────────


class TestExtractActionFromContent:
    def test_extracts_running(self):
        assert "run" in _extract_action_from_content("Running npm test for auth module")

    def test_extracts_deploying(self):
        assert "deploy" in _extract_action_from_content("Deploying service to production")

    def test_extracts_debugging(self):
        assert "debug" in _extract_action_from_content("Debugged test_auth_flow failure")

    def test_fallback_first_word(self):
        assert "someth" == _extract_action_from_content("Something happened")


class TestExtractEntitiesFromContent:
    def test_extracts_capitalized(self):
        ents = _extract_entities_from_content("Deployed MyApp to Kubernetes cluster")
        assert "myapp" in ents or "myapp" in str(ents).lower()

    def test_extracts_snake_case(self):
        ents = _extract_entities_from_content("Fixed my_package_name import error")
        assert "my_package_name" in ents

    def test_extracts_known_tools(self):
        ents = _extract_entities_from_content("Running docker and kubernetes deployment")
        assert "docker" in ents
        assert "kubernetes" in ents


class TestComputeTrend:
    def test_improving_trend(self):
        recent = "success success success pass pass"
        older = "fail fail fail error broken"
        assert _compute_trend(recent, older) == "improving"

    def test_declining_trend(self):
        recent = "fail fail fail broken error"
        older = "success success pass complete"
        assert _compute_trend(recent, older) == "declining"

    def test_stable_trend(self):
        recent = "success fail success pass"
        older = "success fail success pass"
        assert _compute_trend(recent, older) == "stable"


class TestComputeEventDensity:
    def test_high_density(self):
        ents = ["a", "b", "c", "d", "e"]
        density = _compute_event_density(ents)
        assert density == 1.0  # All unique

    def test_low_density(self):
        ents = ["a", "a", "a", "a", "a"]
        density = _compute_event_density(ents)
        assert density == 0.2  # 1/5

    def test_empty_list(self):
        assert _compute_event_density([]) == 0.0


class TestInsightForDomain:
    def test_mastery_high_success(self):
        insight = _insight_for_domain("deploy", 5, 0.9, 10, 0.5)
        assert "mastery" in insight or "reliable" in insight

    def test_exploration_limited_data(self):
        insight = _insight_for_domain("test", 2, 1.0, 2, 0.3)
        assert "exploring" in insight or "limited" in insight

    def test_learning_needed(self):
        insight = _insight_for_domain("deploy", 3, 0.6, 6, 0.4)
        assert "Learning needed" in insight or "mixed" in insight

    def test_unreliable(self):
        insight = _insight_for_domain("deploy", 4, 0.0, 5, 0.8)
        assert "unreliable" in insight or "Low success rate" in insight


# ── Level 1 Tests ──────────────────────────────────────────────────


class TestLevel1Reflection:
    async def test_builds_reflection_from_deployment_facts(
        self, reflector: LLMReflector, seeded_deployment_facts,
    ):
        """Should generate Level 1 reflection for deployment domain."""
        facts = reflector.semantic.get_facts(domain="deployment")
        assert len(facts) == 5

        reflection = reflector._build_level1_reflection("deployment", facts)
        assert reflection is not None
        assert "deployment" in reflection
        assert "observations" in reflection or "deploy" in reflection.lower()

    async def test_reflection_contains_trend(
        self, reflector: LLMReflector, seeded_deployment_facts,
    ):
        facts = reflector.semantic.get_facts(domain="deployment")
        reflection = reflector._build_level1_reflection("deployment", facts)
        assert reflection is not None
        assert "trend" in reflection.lower()

    async def test_reflection_includes_entity_info(
        self, reflector: LLMReflector, seeded_deployment_facts,
    ):
        facts = reflector.semantic.get_facts(domain="deployment")
        reflection = reflector._build_level1_reflection("deployment", facts)
        assert reflection is not None
        # Should mention my-app or kubernetes
        assert "entity" in reflection.lower() or "kub" in reflection.lower() or "my-app" in reflection.lower()


# ── Level 2 Tests ──────────────────────────────────────────────────


class TestLevel2Reflection:
    async def test_cross_domain_linking(
        self, reflector: LLMReflector, seeded_deployment_facts, seeded_testing_facts,
    ):
        """Should detect temporal precedence between domain activities."""
        domain_groups = {
            "deployment": reflector.semantic.get_facts(domain="deployment"),
            "testing": reflector.semantic.get_facts(domain="testing"),
        }
        reflections = reflector._build_level2_reflections(domain_groups)
        # Should find at least one cross-domain reflection
        assert len(reflections) >= 1

    async def test_cross_domain_contains_both_domains(
        self, reflector: LLMReflector, seeded_deployment_facts, seeded_testing_facts,
    ):
        domain_groups = {
            "deployment": reflector.semantic.get_facts(domain="deployment"),
            "testing": reflector.semantic.get_facts(domain="testing"),
        }
        reflections = reflector._build_level2_reflections(domain_groups)
        for _, text in reflections:
            assert "deployment" in text or "testing" in text

    async def test_single_domain_returns_empty(
        self, reflector: LLMReflector, seeded_deployment_facts,
    ):
        domain_groups = {
            "deployment": reflector.semantic.get_facts(domain="deployment"),
        }
        reflections = reflector._build_level2_reflections(domain_groups)
        assert len(reflections) == 0


# ── Level 3 Tests ──────────────────────────────────────────────────


class TestLevel3Reflection:
    async def test_meta_reflection_generated(
        self, reflector: LLMReflector, seeded_deployment_facts, seeded_testing_facts,
    ):
        """Should generate meta-reflection when multiple domains exist."""
        domain_groups = {
            "deployment": reflector.semantic.get_facts(domain="deployment"),
            "testing": reflector.semantic.get_facts(domain="testing"),
        }
        meta = reflector._build_level3_reflection(domain_groups)
        assert meta is not None
        assert "Meta-reflection" in meta
        assert "2" in meta  # Two domains

    async def test_meta_reflection_contains_success_rate(
        self, reflector: LLMReflector, seeded_deployment_facts, seeded_testing_facts,
    ):
        domain_groups = {
            "deployment": reflector.semantic.get_facts(domain="deployment"),
            "testing": reflector.semantic.get_facts(domain="testing"),
        }
        meta = reflector._build_level3_reflection(domain_groups)
        assert meta is not None
        assert "%" in meta or "success rate" in meta.lower()


# ── Full Execute Tests ─────────────────────────────────────────────


class TestFullExecute:
    async def test_execute_should_run_with_recent_facts(
        self, reflector: LLMReflector, seeded_deployment_facts, seeded_testing_facts,
    ):
        """Full execute should process recent facts and create reflections."""
        result = await reflector.execute()
        # The domain_groups has 2 domains (deployment, testing)
        # Level1: one per domain if >= 2 facts
        # Level2: cross-domain linking
        # Level3: meta-reflection
        assert result["created"] >= 3  # L1 + L2 + L3
        assert result["processed"] >= 9  # 5 deployment + 4 testing

    async def test_execute_without_enough_facts(
        self, reflector: LLMReflector, tmp_db: Database,
    ):
        """Should return early when there aren't enough recent facts."""
        # Seed just 1 fact
        node = SemanticNode(
            node_type=NodeType.FACT,
            content="Just one observation",
            source_type=SourceType.OBSERVATION,
            source_ids=["mock"],
            domain="lonely",
            confidence=0.5,
        )
        reflector.semantic.store(node)

        result = await reflector.execute()
        assert result["processed"] == 1
        assert result["created"] == 0

    async def test_execute_creates_reflection_nodes(
        self, reflector: LLMReflector, seeded_deployment_facts, seeded_testing_facts,
    ):
        """Reflections should be stored as semantic nodes."""
        await reflector.execute()

        reflections = reflector.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE node_type IN (?, ?)",
            (NodeType.REFLECTION.value, NodeType.META_REFLECTION.value),
        )
        assert len(reflections) >= 3

    async def test_reflection_content_is_informative(
        self, reflector: LLMReflector, seeded_deployment_facts, seeded_testing_facts,
    ):
        """Reflections should contain meaningful content, not placeholders."""
        await reflector.execute()

        reflections = reflector.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE node_type IN (?, ?)",
            (NodeType.REFLECTION.value, NodeType.META_REFLECTION.value),
        )
        for ref in reflections:
            content = ref["content"]
            assert len(content) > 50  # Not empty/trivial
            # Should not contain placeholder text
            assert "Pattern observed across" not in content
            assert "placeholder" not in content.lower()
            assert "TODO" not in content
