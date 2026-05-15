"""Test cross-agent transfer protocol."""

import pytest

from myelin.core.database import Database
from myelin.core.models import (
    AgentProfile,
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    StepType,
)
from myelin.memory.procedural import ProceduralMemory
from myelin.transfer.profiling import AgentProfiler
from myelin.transfer.protocol import TransferProtocol


@pytest.fixture
def db(tmp_path):
    d = Database(path=tmp_path / "test.db", enable_vec=False)
    _ = d.conn
    yield d
    d.close()


@pytest.fixture
def procedural(db):
    return ProceduralMemory(db)


@pytest.fixture
def profiler(db):
    return AgentProfiler(db)


@pytest.fixture
def protocol(db, procedural):
    return TransferProtocol(db, procedural)


def _register_agents(profiler):
    profiler.register(
        AgentProfile(
            agent_id="agent_a",
            agent_name="Claude Agent",
            tools=["git pull", "npm test", "npm build", "docker build"],
            context_format="mcp_stdio",
            model_family="claude",
        )
    )
    profiler.register(
        AgentProfile(
            agent_id="agent_b",
            agent_name="GPT Agent",
            tools=["git pull", "npm test", "curl"],
            context_format="mcp_stdio",
            model_family="gpt",
        )
    )


def _store_procedure(procedural):
    proc = Procedure(
        name="Deploy pipeline",
        trigger_pattern="deploy to production",
        steps=[
            ProcedureStep(order=0, description="Run git pull", step_type=StepType.CORE),
            ProcedureStep(order=1, description="Run npm test", step_type=StepType.CORE),
            ProcedureStep(order=2, description="Run docker build", step_type=StepType.CORE),
        ],
        preconditions=["tests pass"],
        postconditions=["image built"],
        confidence=0.85,
        source_agent="agent_a",
        status=ProcedureStatus.ACTIVE,
        domain="deployment",
    )
    return procedural.store(proc)


class TestTransferProtocol:
    def test_export_procedure(self, protocol, procedural, profiler):
        _register_agents(profiler)
        proc_id = _store_procedure(procedural)
        package = protocol.export_procedure(proc_id, "agent_a", "agent_b")
        assert package["success"] is True
        assert package["procedure_name"] == "Deploy pipeline"
        assert package["source_agent"] == "agent_a"
        assert package["target_agent"] == "agent_b"
        assert 0 < package["transfer_confidence"] <= package["source_confidence"]
        assert len(package["original_steps"]) == 3
        assert len(package["adapted_steps"]) == 3

    def test_export_missing_procedure(self, protocol):
        package = protocol.export_procedure("nonexistent", "a", "b")
        assert package["success"] is False

    def test_import_procedure(self, protocol, procedural, profiler):
        _register_agents(profiler)
        proc_id = _store_procedure(procedural)
        package = protocol.export_procedure(proc_id, "agent_a", "agent_b")
        result = protocol.import_procedure(package, "agent_b")
        assert result["success"] is True
        assert result["name"] == "Deploy pipeline"
        assert result["status"] == "draft"
        assert result["transfer_confidence"] < 0.85

    def test_imported_procedure_stored(self, protocol, procedural, profiler):
        _register_agents(profiler)
        proc_id = _store_procedure(procedural)
        package = protocol.export_procedure(proc_id, "agent_a", "agent_b")
        result = protocol.import_procedure(package, "agent_b")
        stored = procedural.get(result["new_procedure_id"])
        assert stored is not None
        assert stored["promotion_method"] == "transferred"

    def test_adaptation_flags_missing_tools(self, protocol, procedural, profiler):
        _register_agents(profiler)
        proc_id = _store_procedure(procedural)
        package = protocol.export_procedure(proc_id, "agent_a", "agent_b")
        notes = package["adaptation_notes"]
        has_docker_note = any("docker" in n.lower() for n in notes)
        assert has_docker_note

    def test_get_transferable_procedures(self, protocol, procedural, profiler):
        _register_agents(profiler)
        _store_procedure(procedural)
        available = protocol.get_transferable_procedures("agent_a", "agent_b")
        assert len(available) >= 1
        assert available[0]["procedure_id"] is not None
        assert available[0]["transfer_confidence"] > 0

    def test_already_transferred_excluded(self, protocol, procedural, profiler):
        _register_agents(profiler)
        proc_id = _store_procedure(procedural)
        package = protocol.export_procedure(proc_id, "agent_a", "agent_b")
        protocol.import_procedure(package, "agent_b")
        available = protocol.get_transferable_procedures("agent_a", "agent_b")
        ids = [a["procedure_id"] for a in available]
        assert proc_id not in ids

    def test_transfer_history(self, protocol, procedural, profiler):
        _register_agents(profiler)
        proc_id = _store_procedure(procedural)
        package = protocol.export_procedure(proc_id, "agent_a", "agent_b")
        protocol.import_procedure(package, "agent_b")
        history = protocol.get_transfer_history("agent_a", direction="sent")
        assert len(history) >= 1
        assert history[0]["direction"] == "sent"

    def test_transfer_history_received(self, protocol, procedural, profiler):
        _register_agents(profiler)
        proc_id = _store_procedure(procedural)
        package = protocol.export_procedure(proc_id, "agent_a", "agent_b")
        protocol.import_procedure(package, "agent_b")
        history = protocol.get_transfer_history("agent_b", direction="received")
        assert len(history) >= 1
        assert history[0]["direction"] == "received"

    def test_no_profile_uses_original_steps(self, protocol, procedural):
        proc_id = _store_procedure(procedural)
        package = protocol.export_procedure(proc_id, "unknown_a", "unknown_b")
        assert package["success"] is True
        assert "No target profile" in package["adaptation_notes"][0]


class TestAgentProfiler:
    def test_register_and_get(self, profiler):
        profiler.register(
            AgentProfile(
                agent_id="test_agent",
                tools=["git pull"],
                model_family="claude",
            )
        )
        profile = profiler.get("test_agent")
        assert profile is not None
        assert profile["agent_id"] == "test_agent"

    def test_similarity_identical(self, profiler):
        _register_agents(profiler)
        profiler.register(
            AgentProfile(
                agent_id="agent_a_clone",
                tools=["git pull", "npm test", "npm build", "docker build"],
                context_format="mcp_stdio",
                model_family="claude",
            )
        )
        sim = profiler.compute_similarity("agent_a", "agent_a_clone")
        assert sim > 0.8

    def test_similarity_different(self, profiler):
        _register_agents(profiler)
        sim = profiler.compute_similarity("agent_a", "agent_b")
        assert 0 < sim < 1.0

    def test_transfer_confidence_discounted(self, profiler):
        _register_agents(profiler)
        conf = profiler.compute_transfer_confidence(0.9, "agent_a", "agent_b")
        assert conf < 0.9
        assert conf > 0
