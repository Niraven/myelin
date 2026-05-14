"""Test session lifecycle and cognitive loop."""

import pytest

from myelin.core.database import Database
from myelin.session import Session


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
