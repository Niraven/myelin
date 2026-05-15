"""Test Transfer Protocol v2: capability-aware export/adapt/import."""

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
from myelin.transfer.adaptation import AdaptationResult, StepAdaptationEngine
from myelin.transfer.profiling import AgentCapability, AgentProfiler
from myelin.transfer.protocol import TransferProtocol
from myelin.transfer.tool_map import (
    TOOL_ALIASES,
    find_alternative_tool,
    get_aliases,
    get_canonical_tool,
)


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


@pytest.fixture
def engine():
    return StepAdaptationEngine()


def _register_hermes(profiler):
    profiler.register(
        AgentProfile(
            agent_id="hermes",
            agent_name="Hermes Agent",
            tools=["git pull", "npm test", "npm build", "docker build", "pytest"],
            context_format="mcp_stdio",
            model_family="claude",
        )
    )


def _register_zo(profiler):
    profiler.register(
        AgentProfile(
            agent_id="zo",
            agent_name="Zo Agent",
            tools=["git pull", "npm test", "yarn", "podman"],
            context_format="mcp_stdio",
            model_family="gpt",
        )
    )


def _store_docker_procedure(procedural):
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
        source_agent="hermes",
        status=ProcedureStatus.ACTIVE,
        domain="deployment",
    )
    return procedural.store(proc)


class TestToolMap:
    def test_tool_aliases_present(self):
        assert "git" in TOOL_ALIASES
        assert "docker" in TOOL_ALIASES
        assert "npm" in TOOL_ALIASES

    def test_get_canonical_tool(self):
        assert get_canonical_tool("gh") == "git"
        assert get_canonical_tool("podman") == "docker"
        assert get_canonical_tool("git") == "git"

    def test_get_aliases(self):
        assert "gh" in get_aliases("git")
        assert "podman" in get_aliases("docker")

    def test_find_alternative_tool_exact(self):
        assert find_alternative_tool("git", {"git", "npm"}) == "git"

    def test_find_alternative_tool_alias(self):
        assert find_alternative_tool("docker", {"podman", "npm"}) == "podman"

    def test_find_alternative_tool_none(self):
        assert find_alternative_tool("aws", {"git", "npm"}) is None


class TestAgentCapability:
    def test_get_toolset(self, profiler):
        _register_hermes(profiler)
        toolset = profiler.get_toolset("hermes")
        names = [c.tool_name for c in toolset]
        assert "git pull" in names
        assert "docker build" in names

    def test_has_tool(self, profiler):
        _register_hermes(profiler)
        assert profiler.has_tool("hermes", "git pull") is True
        assert profiler.has_tool("hermes", "yarn") is False

    def test_find_alternative(self, profiler):
        _register_zo(profiler)
        assert profiler.find_alternative("docker", "zo") == "podman"
        assert profiler.find_alternative("npm", "zo") == "yarn"

    def test_agent_capability_dataclass(self):
        cap = AgentCapability(tool_name="git", tool_type="vcs", usage_count=5)
        assert cap.tool_name == "git"
        assert cap.tool_type == "vcs"


class TestStepAdaptationEngine:
    def test_adapt_compatible_step(self, engine):
        step = {"description": "Run git pull", "type": "core"}
        result = engine.adapt(step, ["git pull", "npm test"])
        assert result.quality == 1.0
        assert result.changed is False
        assert result.flag is False

    def test_adapt_rewrites_step(self, engine):
        step = {"description": "Run docker build", "type": "core"}
        result = engine.adapt(step, ["podman", "npm test"])
        assert result.changed is True
        assert result.quality == 0.8
        assert "podman" in result.step["description"]

    def test_adapt_flags_missing_tool(self, engine):
        step = {"description": "Run kubectl apply", "type": "core"}
        result = engine.adapt(step, ["git", "npm"])
        assert result.flag is True
        assert result.quality == 0.4
        assert result.step.get("_flagged") is True

    def test_analyze_requirements(self, engine):
        proc = {
            "steps": [
                {"description": "Run git pull"},
                {"description": "Run docker build"},
                {"description": "Run npm test"},
            ]
        }
        reqs = engine.analyze_requirements(proc)
        assert "git" in reqs
        assert "docker" in reqs
        assert "npm" in reqs

    def test_calculate_confidence_no_changes(self, engine):
        results = [
            AdaptationResult({}, quality=1.0, changed=False),
            AdaptationResult({}, quality=1.0, changed=False),
        ]
        conf, status = engine.calculate_confidence_discount(0.8, results)
        assert conf == 0.8
        assert status == "active"

    def test_calculate_confidence_minor(self, engine):
        results = [
            AdaptationResult({}, quality=0.8, changed=True),
            AdaptationResult({}, quality=1.0, changed=False),
        ]
        conf, status = engine.calculate_confidence_discount(0.8, results)
        assert conf == pytest.approx(0.64)  # 0.8 * 0.8
        assert status == "draft"

    def test_calculate_confidence_multiple(self, engine):
        results = [
            AdaptationResult({}, quality=0.8, changed=True),
            AdaptationResult({}, quality=0.8, changed=True),
            AdaptationResult({}, quality=0.8, changed=True),
        ]
        conf, status = engine.calculate_confidence_discount(0.8, results)
        assert conf == pytest.approx(0.48)  # 0.8 * 0.6
        assert status == "draft"

    def test_calculate_confidence_flagged(self, engine):
        results = [
            AdaptationResult({}, quality=0.4, flag=True),
            AdaptationResult({}, quality=1.0, changed=False),
        ]
        conf, status = engine.calculate_confidence_discount(0.8, results)
        assert conf == pytest.approx(0.32)  # 0.8 * 0.4
        assert status == "draft"

    def test_adapt_procedure(self, engine):
        steps = [
            {"description": "Run git pull"},
            {"description": "Run docker build"},
        ]
        adapted, results, notes = engine.adapt_procedure(steps, ["git", "podman"])
        assert len(adapted) == 2
        assert any(r.changed for r in results)
        assert any(n for n in notes)


class TestTransferProtocolV2:
    def test_export_includes_capability_analysis(self, protocol, procedural, profiler):
        _register_hermes(profiler)
        _register_zo(profiler)
        proc_id = _store_docker_procedure(procedural)
        package = protocol.export_procedure(proc_id, "hermes", "zo")
        assert package["success"] is True
        assert "capability_analysis" in package
        analysis = package["capability_analysis"]
        assert "docker" in analysis["required_tools"]
        assert "podman" in analysis["target_tools_available"]

    def test_export_adapts_steps(self, protocol, procedural, profiler):
        _register_hermes(profiler)
        _register_zo(profiler)
        proc_id = _store_docker_procedure(procedural)
        package = protocol.export_procedure(proc_id, "hermes", "zo")
        adapted = package["adapted_steps"]
        # docker build should be rewritten to podman
        docker_step = adapted[2]
        assert docker_step.get("_adapted_tool") == "podman"
        assert docker_step["type"] == "variant"

    def test_import_applies_confidence_discount(self, protocol, procedural, profiler):
        _register_hermes(profiler)
        _register_zo(profiler)
        proc_id = _store_docker_procedure(procedural)
        package = protocol.export_procedure(proc_id, "hermes", "zo")
        result = protocol.import_procedure(package, "zo")
        assert result["success"] is True
        assert result["discount"] == 0.8
        assert result["transfer_confidence"] < package["transfer_confidence"]
        assert result["status"] == "draft"
        assert result["review_needed"] is False

    def test_import_flagged_review_needed(self, protocol, procedural, profiler):
        _register_hermes(profiler)
        # Register an agent with almost no tools
        profiler.register(
            AgentProfile(
                agent_id="minimal",
                tools=["git"],
                context_format="mcp_stdio",
                model_family="unknown",
            )
        )
        proc_id = _store_docker_procedure(procedural)
        package = protocol.export_procedure(proc_id, "hermes", "minimal")
        result = protocol.import_procedure(package, "minimal")
        assert result["review_needed"] is True
        assert result["discount"] == 0.4
        assert result["status"] == "draft"

    def test_import_no_adaptation_preserved_confidence(self, protocol, procedural, profiler):
        _register_hermes(profiler)
        profiler.register(
            AgentProfile(
                agent_id="hermes_clone",
                tools=["git pull", "npm test", "npm build", "docker build", "pytest"],
                context_format="mcp_stdio",
                model_family="claude",
            )
        )
        proc_id = _store_docker_procedure(procedural)
        package = protocol.export_procedure(proc_id, "hermes", "hermes_clone")
        result = protocol.import_procedure(package, "hermes_clone")
        assert result["discount"] == 1.0
        assert result["status"] == "active"
        assert result["transfer_confidence"] == package["transfer_confidence"]

    def test_transfer_history_records_adaptation(self, protocol, procedural, profiler):
        _register_hermes(profiler)
        _register_zo(profiler)
        proc_id = _store_docker_procedure(procedural)
        package = protocol.export_procedure(proc_id, "hermes", "zo")
        protocol.import_procedure(package, "zo")
        history = protocol.get_transfer_history("zo", direction="received")
        assert len(history) >= 1
        # SQLite stores booleans as integers
        assert bool(history[0].get("adapted")) is True

    def test_adaptation_notes_populated(self, protocol, procedural, profiler):
        _register_hermes(profiler)
        _register_zo(profiler)
        proc_id = _store_docker_procedure(procedural)
        package = protocol.export_procedure(proc_id, "hermes", "zo")
        notes = package["adaptation_notes"]
        assert any("docker" in n.lower() or "podman" in n.lower() for n in notes)
