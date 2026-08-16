"""Test context assembly engine."""

import pytest

from myelin.core.database import Database
from myelin.core.models import (
    ActionType,
    Episode,
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    StepType,
    TrustState,
)
from myelin.intelligence.context import ContextAssembler
from myelin.knowledge.entities import EntityStore
from myelin.knowledge.graph import KnowledgeGraph
from myelin.knowledge.temporal import TemporalIndex
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.retriever import MultiSignalRetriever
from myelin.metacognition.confidence import ConfidenceMap


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
    return ContextAssembler(db, retriever, entities, graph, temporal, procedural, confidence)


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
        agent_id="agent1",
        session_id="s1",
        action="git pull",
        action_type=ActionType.TOOL_CALL,
        content_text="Running git pull origin main",
        success=True,
        domain="deployment",
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
    db.update("procedures", proc.id, {"trust_state": TrustState.TRUSTED.value})

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


def _store_proc(
    procedural, db, name, domain="deployment", trust_state=TrustState.TRUSTED.value, **kwargs
):
    """Store an ACTIVE procedure and set its trust_state explicitly."""
    proc = Procedure(
        name=name,
        trigger_pattern=f"deploy {name}",
        steps=[ProcedureStep(order=0, description=f"{name} step", step_type=StepType.CORE)],
        preconditions=["pre"],
        postconditions=["post"],
        confidence=kwargs.pop("confidence", 0.8),
        source_agent=kwargs.pop("source_agent", "agent1"),
        status=kwargs.pop("status", ProcedureStatus.ACTIVE),
        domain=domain,
    )
    procedural.store(proc)
    db.update("procedures", proc.id, {"trust_state": trust_state})
    return proc


class TestContextTrustShield:
    def test_context_excludes_stale_but_surfaces_trust_band(self, db, assembler):
        # Contract (post 510b93c): the read path must never be empty, so
        # seed/candidate procedures ARE admitted — with their trust_state
        # surfaced so consumers can treat unvalidated ones as review-before-use.
        # Stale/archived procedures remain shielded at the SQL layer.
        names = {
            TrustState.TRUSTED.value: "trusted_deploy",
            TrustState.VALIDATED.value: "validated_deploy",
            TrustState.CANDIDATE.value: "candidate_deploy",
            TrustState.SEED.value: "seed_deploy",
            TrustState.STALE.value: "stale_deploy",
        }
        for state, name in names.items():
            _store_proc(assembler.procedural, db, name, trust_state=state)

        result = assembler.assemble("deploy the application", max_procedures=10)
        proc_names = [p["name"] for p in result["matching_procedures"]]

        assert set(proc_names) == {
            "trusted_deploy",
            "validated_deploy",
            "candidate_deploy",
            "seed_deploy",
        }
        assert "stale_deploy" not in proc_names
        assert "stale_deploy" not in result["assembled_text"]
        assert all("stale_deploy" not in s for s in result["suggested_actions"])

        # Safety property: every admitted procedure carries its trust_state so
        # the agent sees candidate/seed as unvalidated (review-before-use).
        trust_by_name = {p["name"]: p["trust_state"] for p in result["matching_procedures"]}
        assert trust_by_name["trusted_deploy"] == TrustState.TRUSTED.value
        assert trust_by_name["validated_deploy"] == TrustState.VALIDATED.value
        assert trust_by_name["candidate_deploy"] == TrustState.CANDIDATE.value
        assert trust_by_name["seed_deploy"] == TrustState.SEED.value

    def test_context_filters_procedures_by_domain(self, db, assembler):
        _store_proc(assembler.procedural, db, "deploy_prod", domain="deployment")
        _store_proc(assembler.procedural, db, "deploy_staging", domain="staging")

        result = assembler.assemble("deploy the application", domain="deployment")
        proc_names = [p["name"] for p in result["matching_procedures"]]
        assert "deploy_prod" in proc_names
        assert "deploy_staging" not in proc_names
        assert "deploy_staging" not in result["assembled_text"]

    def test_context_domain_none_does_not_filter(self, db, assembler):
        _store_proc(assembler.procedural, db, "deploy_prod", domain="deployment")
        _store_proc(assembler.procedural, db, "deploy_staging", domain="staging")

        result = assembler.assemble("deploy the application")
        proc_names = [p["name"] for p in result["matching_procedures"]]
        assert {"deploy_prod", "deploy_staging"} <= set(proc_names)

    def test_context_includes_trust_state_per_procedure(self, db, assembler):
        _store_proc(
            assembler.procedural, db, "trusted_deploy", trust_state=TrustState.TRUSTED.value
        )
        _store_proc(
            assembler.procedural, db, "validated_deploy", trust_state=TrustState.VALIDATED.value
        )

        result = assembler.assemble("deploy the application")
        states = {p["name"]: p["trust_state"] for p in result["matching_procedures"]}
        assert states["trusted_deploy"] == TrustState.TRUSTED.value
        assert states["validated_deploy"] == TrustState.VALIDATED.value

    def test_context_composes_agent_and_trust_filters(self, db, assembler):
        _store_proc(assembler.procedural, db, "own_trusted", source_agent="agent1")
        _store_proc(assembler.procedural, db, "other_trusted", source_agent="agent2")
        _store_proc(
            assembler.procedural,
            db,
            "own_candidate",
            source_agent="agent1",
            trust_state=TrustState.CANDIDATE.value,
        )

        result = assembler.assemble("deploy the application", agent_ids=["agent1"])
        proc_names = [p["name"] for p in result["matching_procedures"]]
        # Same-agent candidate is admitted (unvalidated, review-before-use);
        # other agents' procedures stay shielded; trust_state is always surfaced.
        assert "own_trusted" in proc_names
        assert "own_candidate" in proc_names
        assert "other_trusted" not in proc_names
        trust_by_name = {p["name"]: p["trust_state"] for p in result["matching_procedures"]}
        assert trust_by_name["own_trusted"] == TrustState.TRUSTED.value
        assert trust_by_name["own_candidate"] == TrustState.CANDIDATE.value
