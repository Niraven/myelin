"""Integration tests exercising full MCP tool flows end-to-end.

These tests simulate real agent usage patterns: observe actions, build up
memory, query context, learn and transfer procedures, run cognitive processes.
"""

import pytest

from myelin.core.database import Database
from myelin.core.models import AgentProfile, TrustState
from myelin.memory.embedding import NoOpEmbedding
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory
from myelin.session import Session
from myelin.tools.handlers import ToolHandlers
from myelin.transfer.profiling import AgentProfiler


@pytest.fixture
def db(tmp_path):
    d = Database(path=tmp_path / "integration.db", enable_vec=False)
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
    )


class TestObserveAndRecall:
    """Test the observe -> recall flow."""

    async def test_observe_then_recall(self, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="git pull origin main",
            action_type="tool_call",
            content_text="Pulled latest changes from main branch",
            domain="deployment",
        )
        result = await handlers.recall("pulled latest changes", limit=5)
        assert result["total_results"] >= 1

    async def test_observe_extracts_entities(self, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="npm test",
            action_type="tool_call",
            content_text="Running npm test in the project",
            domain="testing",
        )
        result = await handlers.entities_query(search="npm")
        assert len(result["entities"]) >= 1

    async def test_observe_updates_confidence(self, handlers):
        for i in range(5):
            await handlers.observe(
                agent_id="agent1",
                session_id="s1",
                action=f"deploy v{i}",
                action_type="tool_call",
                content_text=f"Deploying version {i} to production",
                domain="deployment",
            )
        result = await handlers.confidence(domain="deployment")
        assert result["domain"]["confidence"] > 0

    async def test_observe_batch_records_valid_events_and_reports_invalid(self, handlers):
        result = await handlers.observe_batch(
            [
                {
                    "agent_id": "agent1",
                    "session_id": "batch-1",
                    "action": "deploy service-7",
                    "action_type": "tool_call",
                    "content_text": "Deploy service-7 with kubectl rollout",
                    "domain": "deployment",
                },
                {
                    "agent_id": "agent1",
                    "session_id": "batch-1",
                    "action": "missing content",
                    "action_type": "tool_call",
                },
            ]
        )

        assert result["recorded"] == 1
        assert len(result["failed"]) == 1
        recall = await handlers.recall("service-7", limit=5)
        assert recall["total_results"] >= 1


class TestContextAssembly:
    """Test the context assembly flow."""

    async def test_context_with_observations(self, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="docker build",
            action_type="tool_call",
            content_text="Building docker image for deployment",
            domain="deployment",
        )
        result = await handlers.context(query="docker deployment", domain="deployment")
        assert "relevant_memories" in result
        assert "matching_procedures" in result
        assert "assembled_text" in result

    async def test_context_with_taught_procedure(self, handlers, db):
        teach_result = await handlers.teach(
            name="Build and Deploy",
            trigger_pattern="build and deploy docker image",
            steps=[
                {"description": "Run docker build", "type": "core"},
                {"description": "Run docker push", "type": "core"},
                {"description": "Update k8s deployment", "type": "core"},
            ],
            agent_id="agent1",
            domain="deployment",
        )
        proc_id = teach_result["procedure_id"]

        # Shield: a freshly taught procedure is only CANDIDATE, so it must not
        # enter assembled context until its trust_state is promoted.
        candidate_result = await handlers.context(
            query="build and deploy docker", domain="deployment"
        )
        assert candidate_result["matching_procedures"] == []

        # Promote to TRUSTED, then the exact-domain context must surface it.
        db.update("procedures", proc_id, {"trust_state": TrustState.TRUSTED.value})
        result = await handlers.context(query="build and deploy docker", domain="deployment")
        assert len(result["matching_procedures"]) >= 1
        assert result["matching_procedures"][0]["name"] == "Build and Deploy"


class TestProcedureLifecycle:
    """Test teach -> execute -> feedback flow."""

    async def test_full_procedure_lifecycle(self, handlers):
        teach_result = await handlers.teach(
            name="Run Tests",
            trigger_pattern="run the test suite",
            steps=[
                {"description": "Install dependencies", "type": "core"},
                {"description": "Run pytest", "type": "core"},
                {"description": "Check coverage", "type": "optional"},
            ],
            agent_id="agent1",
            domain="testing",
        )
        proc_id = teach_result["procedure_id"]
        assert teach_result["confidence"] == 0.7

        exec_result = await handlers.execute_procedure(
            query="run the test suite", agent_id="agent1"
        )
        assert exec_result["found"] is True
        assert exec_result["procedure_id"] == proc_id

        feedback = await handlers.procedure_feedback(procedure_id=proc_id, success=True)
        assert feedback["new_confidence"] > 0.7
        assert feedback["success_count"] == 1

        feedback2 = await handlers.procedure_feedback(procedure_id=proc_id, success=True)
        assert feedback2["new_confidence"] > feedback["new_confidence"]

    async def test_failure_decreases_confidence(self, handlers):
        teach_result = await handlers.teach(
            name="Fragile Deploy",
            trigger_pattern="fragile deploy",
            steps=[{"description": "Deploy", "type": "core"}],
            agent_id="agent1",
        )
        proc_id = teach_result["procedure_id"]
        feedback = await handlers.procedure_feedback(procedure_id=proc_id, success=False)
        assert feedback["new_confidence"] < 0.7

    async def test_repeated_workflows_promote_executable_procedure(self, db, handlers):
        workflow = [
            "git pull origin main",
            "npm test",
            "docker build myelin:latest",
            "docker push registry/myelin:latest",
            "kubectl rollout restart deployment/myelin",
        ]

        for run in range(5):
            session = Session(db, agent_id="demo-agent", session_id=f"deploy-{run}")
            for action in workflow:
                await session.observe(
                    action=action,
                    action_type="tool_call",
                    content_text=f"{action} during production deployment",
                    domain="deployment",
                )

        final_session = Session(db, agent_id="demo-agent", session_id="launch-proof")
        result = await final_session.end()

        promoter_result = next(
            item for item in result["cognitive_results"] if item["process"] == "promoter"
        )
        assert promoter_result["created"] >= 1

        execution = await handlers.execute_procedure(
            query="deployment workflow",
            agent_id="demo-agent",
        )
        assert execution["found"] is True
        assert execution["confidence"] >= 0.65
        assert execution["trust_level"] in {"candidate", "validated", "trusted"}
        assert execution["recommendation"] == "suggest_only_review_before_execution"
        assert len(execution["steps"]) >= 3

    async def test_observe_batch_and_sleep_promote_procedure(self, handlers):
        workflow = [
            "git pull origin main",
            "npm test",
            "docker build myelin:latest",
            "docker push registry/myelin:latest",
            "kubectl rollout restart deployment/myelin",
        ]
        events = []
        for run in range(5):
            for action in workflow:
                events.append(
                    {
                        "agent_id": "demo-agent",
                        "session_id": f"batch-deploy-{run}",
                        "action": action,
                        "action_type": "tool_call",
                        "content_text": f"{action} during batched production deployment",
                        "domain": "deployment",
                    }
                )

        batch = await handlers.observe_batch(events)
        assert batch["recorded"] == len(events)
        assert batch["failed"] == []

        sleep = await handlers.trigger_sleep(agent_id="demo-agent")
        assert sleep["promoter"]["created"] >= 1

        execution = await handlers.execute_procedure(
            query="deployment workflow",
            agent_id="demo-agent",
        )
        assert execution["found"] is True
        assert len(execution["steps"]) >= 3


class TestMultiSignalQuery:
    """Test the multi-signal retriever via MCP tools."""

    async def test_query_with_entity_boosting(self, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="git pull",
            action_type="tool_call",
            content_text="Pulled latest with git pull origin main",
            domain="dev",
        )
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="npm test",
            action_type="tool_call",
            content_text="Running npm test after git pull",
            domain="dev",
        )
        result = await handlers.query(query="git pull", limit=5)
        assert result["total"] >= 1

    async def test_query_with_custom_weights(self, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="deploy",
            action_type="tool_call",
            content_text="Deploying the application",
            domain="ops",
        )
        result = await handlers.query(
            query="deploy",
            weights={"text": 0.5, "vector": 0.0, "entity": 0.2, "temporal": 0.2, "activation": 0.1},
        )
        assert result["total"] >= 1


class TestKnowledgeGraph:
    """Test graph query tools."""

    async def test_graph_query_after_observations(self, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="git pull",
            action_type="tool_call",
            content_text="Running git pull origin main",
        )
        result = await handlers.graph_query(entity_name="git pull")
        if result.get("found"):
            assert "entity" in result
            assert "neighbors" in result


class TestTemporalQuery:
    """Test temporal state queries."""

    async def test_temporal_domain_query(self, handlers):
        result = await handlers.temporal_query(domain="deployment")
        assert "domain" in result
        assert "current_states" in result

    async def test_what_changed(self, handlers):
        e1 = handlers.entities.upsert_entity("cloudflared", "service", "cloudflared")
        handlers.temporal.record_state(
            "booting", entity_id=e1, domain="infrastructure", confidence=0.55
        )
        handlers.temporal.record_state(
            "running", entity_id=e1, domain="infrastructure", confidence=0.95
        )

        result = await handlers.what_changed(
            domain="infrastructure",
            since="1990-01-01T00:00:00",
        )

        assert result["change_count"] == 1
        assert result["changes"][0]["entity"] == "cloudflared"
        assert result["changes"][0]["from"] == "booting"
        assert result["changes"][0]["to"] == "running"
        assert "markdown" in result

    async def test_entity_status(self, handlers):
        e1 = handlers.entities.upsert_entity("redis", "service", "redis")
        handlers.temporal.record_state("starting", entity_id=e1, domain="deployment")
        handlers.temporal.record_state("degraded", entity_id=e1, domain="deployment")

        result = await handlers.entity_status(entity_name="redis")

        assert result["found"] is True
        assert result["entity"]["name"] == "redis"
        assert result["current_state"]["description"] == "degraded"
        assert len(result["recent_history"]) == 2
        assert len(result["transitions"]) == 1
        assert "markdown" in result

    async def test_entity_status_not_found(self, handlers):
        result = await handlers.entity_status(entity_name="does-not-exist")
        assert result["found"] is False
        assert "error" in result
        assert "markdown" in result


class TestTransferFlow:
    """Test the full transfer lifecycle."""

    async def test_discover_export_import(self, handlers, db):
        profiler = AgentProfiler(db)
        profiler.register(
            AgentProfile(
                agent_id="claude_agent",
                tools=["git pull", "npm test", "docker build"],
                model_family="claude",
            )
        )
        profiler.register(
            AgentProfile(
                agent_id="gpt_agent",
                tools=["git pull", "npm test"],
                model_family="gpt",
            )
        )

        teach_result = await handlers.teach(
            name="CI Pipeline",
            trigger_pattern="run CI pipeline",
            steps=[
                {"description": "Run git pull", "type": "core"},
                {"description": "Run npm test", "type": "core"},
                {"description": "Run docker build", "type": "core"},
            ],
            agent_id="claude_agent",
            domain="ci",
        )
        proc_id = teach_result["procedure_id"]

        for _ in range(3):
            await handlers.procedure_feedback(procedure_id=proc_id, success=True)

        discover = await handlers.transfer_discover(
            source_agent="claude_agent",
            target_agent="gpt_agent",
            min_confidence=0.5,
        )
        assert discover["count"] >= 1

        package = await handlers.transfer_export(
            procedure_id=proc_id,
            source_agent="claude_agent",
            target_agent="gpt_agent",
        )
        assert package["success"] is True
        assert package["transfer_confidence"] > 0

        imported = await handlers.transfer_import(
            package=package,
            agent_id="gpt_agent",
        )
        assert imported["success"] is True
        assert imported["status"] == "draft"


class TestSleepCycle:
    """Test manual sleep trigger."""

    async def test_trigger_sleep(self, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="deploy v2",
            action_type="tool_call",
            content_text="Deployed version 2 to production",
            domain="deployment",
        )
        result = await handlers.trigger_sleep()
        assert result["status"] == "completed"
        assert "entities_extracted" in result


class TestStatus:
    """Test status tool."""

    async def test_status_returns_all_metrics(self, handlers):
        await handlers.observe(
            agent_id="agent1",
            session_id="s1",
            action="test",
            action_type="tool_call",
            content_text="Running a test",
        )
        status = await handlers.status(agent_id="agent1")
        assert status["episodes"] >= 1
        assert "procedures" in status
        assert "entities" in status
        assert "relationships" in status
        assert "temporal_states" in status
