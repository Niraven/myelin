"""Test session lifecycle and cognitive loop."""

import time

import pytest

from myelin.cognitive.orchestrator import CognitiveOrchestrator
from myelin.core.database import Database
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory
from myelin.session import Session


class DummyProcess:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def should_run(self) -> bool:
        return True

    async def run(self) -> dict:
        self.calls += 1
        return {"items_processed": self.calls}


@pytest.fixture
def session(tmp_path):
    db = Database(path=tmp_path / "test.db", enable_vec=False)
    _ = db.conn
    sess = Session(db=db, agent_id="test-agent", session_id="test-session")
    yield sess
    db.close()


@pytest.mark.asyncio
async def test_observe_records_episode(session):
    ep_id = await session.observe(
        action="git pull origin main",
        action_type="tool_call",
        content_text="Pulling latest changes",
        domain="deployment",
    )
    assert ep_id is not None
    assert session._episode_count == 1


@pytest.mark.asyncio
async def test_multiple_observations(session):
    for i in range(5):
        await session.observe(
            action=f"action_{i}",
            action_type="tool_call",
            content_text=f"Did action {i}",
        )
    assert session._episode_count == 5


@pytest.mark.asyncio
async def test_session_end_runs_processes(session):
    for i in range(5):
        await session.observe(
            action=f"action_{i}",
            action_type="tool_call",
            content_text=f"Content {i}",
            domain="testing",
        )

    result = await session.end()
    assert result["session_id"] == "test-session"
    assert result["episodes_recorded"] == 5
    assert "cognitive_results" in result
    assert len(result["cognitive_results"]) > 0


@pytest.mark.asyncio
async def test_session_stats(session):
    await session.observe(
        action="test",
        action_type="tool_call",
        content_text="test content",
    )
    stats = session.get_stats()
    assert stats["session_id"] == "test-session"
    assert stats["episodes"] == 1
    assert stats["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_domain_confidence_updates(session):
    for i in range(5):
        await session.observe(
            action=f"npm test {i}",
            action_type="tool_call",
            content_text=f"Running test suite {i}",
            domain="testing",
        )

    domain = session.confidence_map.get_domain("testing")
    assert domain is not None
    assert domain["episode_count"] == 5
    assert domain["confidence"] > 0


@pytest.mark.asyncio
async def test_write_threshold_runs_consolidator_and_nrem(tmp_path):
    db = Database(path=tmp_path / "trigger.db", enable_vec=False)
    _ = db.conn
    orchestrator = CognitiveOrchestrator(
        db,
        EpisodicMemory(db),
        SemanticMemory(db),
        ProceduralMemory(db),
    )
    orchestrator.reconsolidation = DummyProcess("reconsolidation")
    orchestrator.llm_consolidator = DummyProcess("consolidator")
    orchestrator.prioritized_replay = DummyProcess("prioritized_replay")
    orchestrator.nrem_sleep = DummyProcess("nrem_sleep")
    orchestrator.rem_sleep = DummyProcess("rem_sleep")
    orchestrator.decayer = DummyProcess("decayer")
    orchestrator._last_decay = time.time()

    for _ in range(50):
        orchestrator.on_write()

    results = await orchestrator.check_triggers()
    names = [r["process"] for r in results]

    assert "consolidator" in names
    assert "prioritized_replay" in names
    assert "nrem_sleep" in names
    assert orchestrator._write_count == 0
    assert orchestrator.llm_consolidator.calls == 1
    assert orchestrator.prioritized_replay.calls == 1
    assert orchestrator.nrem_sleep.calls == 1

    db.close()


@pytest.mark.asyncio
async def test_session_end_runs_full_cognitive_sequence(tmp_path):
    db = Database(path=tmp_path / "session-end.db", enable_vec=False)
    _ = db.conn
    orchestrator = CognitiveOrchestrator(
        db,
        EpisodicMemory(db),
        SemanticMemory(db),
        ProceduralMemory(db),
    )
    expected = [
        "reconsolidation",
        "consolidator",
        "schema_learner",
        "reflector",
        "nrem_sleep",
        "promoter",
        "composer",
        "decayer",
        "challenger",
    ]
    attrs = {
        "reconsolidation": "reconsolidation",
        "consolidator": "llm_consolidator",
        "schema_learner": "schema_learner",
        "reflector": "llm_reflector",
        "nrem_sleep": "nrem_sleep",
        "promoter": "promoter",
        "composer": "composer",
        "decayer": "decayer",
        "challenger": "challenger",
    }
    for name, attr in attrs.items():
        setattr(orchestrator, attr, DummyProcess(name))

    results = await orchestrator.on_session_end()
    assert [r["process"] for r in results] == expected

    db.close()
