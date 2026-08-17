"""Regression tests for the retrieval-surface fixes.

Covers:
- facts() merging distilled semantic_nodes with legacy semantic_facts
- recall() synthesize flag and content truncation
- synthesizer SearchResult.from_mapping accepting Myelin result shapes
"""

import pytest

from myelin.core.database import Database
from myelin.core.models import NodeType
from myelin.intelligence.synthesizer import SearchResult, Synthesizer
from myelin.memory.embedding import NoOpEmbedding
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory
from myelin.tools.handlers import ToolHandlers


@pytest.fixture
def db(tmp_path):
    d = Database(path=tmp_path / "retrieval.db", enable_vec=False)
    _ = d.conn
    yield d
    d.close()


@pytest.fixture
def handlers(db):
    return ToolHandlers(
        EpisodicMemory(db),
        SemanticMemory(db),
        ProceduralMemory(db),
        NoOpEmbedding(),
        synthesizer=Synthesizer(),
    )


@pytest.fixture
def synthesizer():
    return Synthesizer()


class TestFactsMergesSemanticNodes:
    """myelin_facts must surface distilled knowledge from semantic_nodes."""

    async def test_facts_returns_distilled_nodes(self, db, handlers):
        # Write a distilled fact via the same store consolidation uses.
        from myelin.core.models import SemanticNode

        handlers.semantic.store(
            SemanticNode(
                node_type=NodeType.FACT.value,
                content="Hermes is an autonomous agent framework by Nous Research.",
                source_type="observation",
                confidence=0.9,
                domain="tools",
                source_ids=["ep-1"],
            )
        )
        result = await handlers.facts(agent_id="hermes")
        assert result["distilled_count"] >= 1
        values = [f["value"] for f in result["facts"]]
        assert any("autonomous agent framework" in v for v in values)
        assert all(f["origin"] == "semantic_nodes" for f in result["facts"])

    async def test_facts_merges_legacy_and_distilled(self, db, handlers):
        from myelin.core.models import SemanticNode

        handlers.semantic.store(
            SemanticNode(
                node_type=NodeType.FACT.value,
                content="Obsidian is the canonical durable knowledge layer.",
                source_type="observation",
                confidence=0.8,
                source_ids=["ep-2"],
            )
        )
        await handlers.memorize(
            agent_id="hermes",
            key="model_routing_preference",
            value="prefer local ollama for private extraction",
            domain="knowledge-management",
        )
        result = await handlers.facts(agent_id="hermes")
        origins = {f["origin"] for f in result["facts"]}
        assert origins == {"semantic_nodes", "semantic_facts"}
        assert result["legacy_count"] >= 1
        legacy = [f for f in result["facts"] if f["origin"] == "semantic_facts"]
        assert legacy[0]["key"] == "model_routing_preference"


class TestRecallSynthesisAndTruncation:
    """myelin_recall must distill answers and never return raw dumps."""

    async def test_recall_truncates_long_content(self, db, handlers):
        long = "x" * 5000
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="test action",
            action_type="tool_call",
            content_text=long,
            domain="test",
        )
        result = await handlers.recall(query="test action", limit=5)
        episodes = result["results"].get("episodes", [])
        assert episodes, "expected at least one episode back"
        assert all(len(e.get("content_text", "")) <= 800 + 20 for e in episodes)
        assert all(e.get("_full_content_length", 0) == 5000 for e in episodes)

    async def test_recall_synthesize_runs_rule_based(self, db, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="git commit",
            action_type="tool_call",
            content_text="committed the retrieval fix and pushed to origin",
            domain="dev",
        )
        result = await handlers.recall(query="git commit", limit=5, synthesize=True)
        assert result["synthesis_mode"] == "synthesized"
        assert result["synthesis"]
        assert result["message"] == "Answer synthesized from retrieved memories."
        assert result["sources"]

    async def test_recall_synthesize_empty(self, db, handlers):
        result = await handlers.recall(query="nonexistent topic zzz", limit=5, synthesize=True)
        assert result["synthesis_mode"] == "empty"
        assert result["synthesis"] is None

    async def test_recall_no_synthesis_by_default(self, db, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="git commit",
            action_type="tool_call",
            content_text="committed the fix",
            domain="dev",
        )
        result = await handlers.recall(query="git commit", limit=5)
        assert "synthesis" not in result
        assert "results" in result


class TestSynthesizerShapeContract:
    """SearchResult.from_mapping must accept Myelin's result fields."""

    def test_from_mapping_reads_composite_score(self):
        r = SearchResult.from_mapping(
            {
                "id": "ep-1",
                "content_text": "ran git commit and pushed",
                "_composite_score": 0.83,
                "source_type": "episode",
            }
        )
        assert r.snippet == "ran git commit and pushed"
        assert r.score == pytest.approx(0.83)
        assert r.title == "Untitled result"

    def test_from_mapping_reads_composite_score_plain(self):
        r = SearchResult.from_mapping(
            {
                "content_text": "installed browserbase cli",
                "composite_score": 0.7,
                "source_type": "observation",
            }
        )
        assert r.snippet == "installed browserbase cli"
        assert r.score == pytest.approx(0.7)

    def test_synthesize_rule_based_with_myelin_shape(self, synthesizer):
        results = [
            {
                "id": "ep-1",
                "content_text": "rebuilt myelin from scratch with 261 procedures",
                "_composite_score": 0.9,
            },
            {
                "id": "ep-2",
                "content_text": "fixed qdrant by restarting docker container",
                "_composite_score": 0.8,
            },
        ]
        out = synthesizer.synthesize(query="what happened", results=results)
        assert out["mode"] == "synthesized"
        assert out["synthesis"]
        assert out["source_count"] == 2
        # Snippets must not be empty "Untitled result" garbage.
        assert all(s["content"] for s in out["sources"])
