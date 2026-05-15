"""Test episodic memory operations."""

from myelin.core.models import ActionType, Episode


def test_record_and_retrieve(episodic):
    ep = Episode(
        agent_id="test-agent",
        session_id="session-1",
        action="git pull origin main",
        action_type=ActionType.TOOL_CALL,
        content_text="Pulled latest changes from main branch",
        success=True,
        domain="deployment",
    )
    episode_id = episodic.record(ep)
    assert episode_id == ep.id

    retrieved = episodic.get(episode_id)
    assert retrieved is not None
    assert retrieved["agent_id"] == "test-agent"
    assert retrieved["action"] == "git pull origin main"
    assert retrieved["domain"] == "deployment"


def test_count(episodic):
    for i in range(5):
        ep = Episode(
            agent_id="agent-a",
            session_id="s1",
            action=f"action_{i}",
            action_type=ActionType.TOOL_CALL,
            content_text=f"Did action {i}",
        )
        episodic.record(ep)

    assert episodic.count("agent-a") == 5
    assert episodic.count("agent-b") == 0
    assert episodic.count() == 5


def test_access_updates_activation(episodic):
    ep = Episode(
        agent_id="test-agent",
        session_id="s1",
        action="test action",
        action_type=ActionType.TOOL_CALL,
        content_text="testing access tracking",
    )
    episodic.record(ep)

    episodic.access(ep.id)
    episodic.access(ep.id)

    retrieved = episodic.get(ep.id)
    assert retrieved["access_count"] == 3  # 1 initial + 2 accesses
    assert len(retrieved["access_times"]) == 3


def test_fts_search(episodic):
    ep1 = Episode(
        agent_id="a",
        session_id="s1",
        action="npm test",
        action_type=ActionType.TOOL_CALL,
        content_text="Running npm test suite for the project",
        domain="testing",
    )
    ep2 = Episode(
        agent_id="a",
        session_id="s1",
        action="git push",
        action_type=ActionType.TOOL_CALL,
        content_text="Pushing code to remote repository",
        domain="deployment",
    )
    episodic.record(ep1)
    episodic.record(ep2)

    results = episodic.search_text("npm test")
    assert len(results) >= 1
    assert any("npm" in r.get("action", "") for r in results)


def test_get_by_session(episodic):
    for i in range(3):
        ep = Episode(
            agent_id="a",
            session_id="target-session",
            action=f"action_{i}",
            action_type=ActionType.TOOL_CALL,
            content_text=f"Content {i}",
        )
        episodic.record(ep)

    ep_other = Episode(
        agent_id="a",
        session_id="other-session",
        action="other",
        action_type=ActionType.TOOL_CALL,
        content_text="Other session",
    )
    episodic.record(ep_other)

    results = episodic.get_by_session("target-session")
    assert len(results) == 3


def test_mark_consolidated(episodic):
    ids = []
    for i in range(3):
        ep = Episode(
            agent_id="a",
            session_id="s1",
            action=f"action_{i}",
            action_type=ActionType.TOOL_CALL,
            content_text=f"Content {i}",
        )
        episodic.record(ep)
        ids.append(ep.id)

    episodic.mark_consolidated(ids, "cluster-001")

    for eid in ids:
        row = episodic.get(eid)
        assert row["consolidated"] == 1
        assert row["cluster_id"] == "cluster-001"


def test_get_unconsolidated(episodic):
    for i in range(3):
        ep = Episode(
            agent_id="a",
            session_id="s1",
            action=f"action_{i}",
            action_type=ActionType.TOOL_CALL,
            content_text=f"Content {i}",
        )
        episodic.record(ep)

    assert len(episodic.get_unconsolidated()) == 3
