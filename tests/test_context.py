"""Test context assembly engine."""

import pytest

from myelin.core.database import Database
from myelin.intelligence.context import ContextAssembler
from myelin.knowledge.entities import EntityStore
from myelin.knowledge.graph import KnowledgeGraph
from myelin.knowledge.temporal import TemporalIndex
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.retriever import MultiSignalRetriever
from myelin.memory.semantic import SemanticMemory
from myelin.metacognition.confidence import ConfidenceMap
from myelin.core.models import (
    ActionType, Episode, Procedure, ProcedureStatus,
    ProcedureStep, StepType,
)


@pytest.fixture
def db(tmp_path):
    d = Database(path=tmp_path / "test.db", enable_vec=False)
    _ = d.conn
    yield d
    d.close()


@pytest.fixture
def assembler(db):
    entities = EntityStore(db)
    graph = KnowledgeGraph(db)
    temporal = TemporalIndex(db)
    procedural = ProceduralMemory(db)
    confidence = ConfidenceMap(db)
    retriever = MultiSignalRetriever(db, entities, graph, temporal)
    return ContextAssembler(
        db, retriever, entities, graph, temporal, procedural, confidence
    )


@pytest.fixture
def populated(db, assembler):
    """Populate with some data for richer tests."""
    episodic = EpisodicMemory(db)
    procedural = ProceduralMemory(db)
    entities = EntityStore(db)
    graph = KnowledgeGraph(db)
    temporal = TemporalIndex(db)
    confidence = ConfidenceMap(db)

    ep = Episode(
        agent_id="agent1", session_id="s1", action="git pull",
        action_type=ActionType.TOOL_CALL, content_text="Running git pull origin main",
        success=True, domain="deployment",
    )
    episodic.record(ep)

    proc = Procedure(
        name="Deploy to prod",
        trigger_pattern="deploy to production",
        steps=[
            ProcedureStep(order=0, description="Run git pull", step_type=StepType.CORE),
            ProcedureStep(order=1, description="Run npm test", step_type=StepType.CORE),
            ProcedureStep(order=2, description="Run npm build", step_type=StepType.CORE),
        ],
        preconditions=["tests pass"],
        postconditions=["app deployed"],
        confidence=0.85,
        source_agent="agent1",
        status=ProcedureStatus.ACTIVE,
        domain="deployment",
    )
    procedural.store(proc)

    e1 = entities.upsert_entity("git pull", "tool", "git pull")
    e2 = entities.upsert_entity("npm test", "tool", "npm test")
    graph.add_relationship(e1, e2, "triggers")

    temporal.record_state("Service healthy", entity_id=e1, domain="deployment")

    confidence.update_domain("deployment", episode_delta=10, procedure_delta=2)

    return assembler


class TestContextAssembler:
    def test_assemble_empty_query(self, assembler):
        result = assembler.assemble("nonexistent query xyz abc 123")
        assert "query" in result
        assert "relevant_memories" in result
        assert "matching_procedures" in result
        assert "entity_context" in result
        assert "assembled_text" in result
        assert "stats" in result

    def test_assemble_returns_all_sections(self, populated):
        result = populated.assemble("deploy to production", domain="deployment")
        assert result["query"] == "deploy to production"
        assert result["domain"] == "deployment"
        assert isinstance(result["relevant_memories"], list)
        assert isinstance(result["matching_procedures"], list)
        assert isinstance(result["entity_context"], list)
        assert isinstance(result["suggested_actions"], list)
        assert isinstance(result["assembled_text"], str)

    def test_assemble_finds_procedures(self, populated):
        result = populated.assemble("deploy to production")
        procs = result["matching_procedures"]
        assert len(procs) >= 1
        assert procs[0]["name"] == "Deploy to prod"
        assert procs[0]["confidence"] == 0.85
        assert len(procs[0]["steps"]) == 3

    def test_assemble_finds_memories(self, populated):
        result = populated.assemble("git pull")
        assert result["stats"]["memories_retrieved"] >= 1

    def test_assemble_includes_entity_context(self, populated):
        result = populated.assemble("git pull origin main")
        entities = result["entity_context"]
        if entities:
            assert "name" in entities[0]
            assert "type" in entities[0]

    def test_assemble_includes_domain_confidence(self, populated):
        result = populated.assemble("deploy", domain="deployment")
        conf = result["domain_confidence"]
        assert conf is not None
        assert conf["domain"] == "deployment"
        assert conf["confidence"] > 0

    def test_assemble_generates_suggestions(self, populated):
        result = populated.assemble("deploy to production")
        assert len(result["suggested_actions"]) >= 1
        assert "Deploy to prod" in result["suggested_actions"][0]

    def test_assemble_text_rendering(self, populated):
        result = populated.assemble("deploy to production", domain="deployment")
        text = result["assembled_text"]
        assert len(text) > 0
        assert "No relevant context found" not in text

    def test_assemble_respects_limits(self, populated):
        result = populated.assemble("deploy", max_memories=1, max_procedures=1)
        assert result["stats"]["memories_retrieved"] <= 1
        assert result["stats"]["procedures_matched"] <= 1

    def test_assemble_without_optional_sections(self, assembler):
        result = assembler.assemble(
            "test query",
            include_graph=False,
            include_temporal=False,
            include_confidence=False,
        )
        assert result["domain_confidence"] is None
