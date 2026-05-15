"""Test agent profile learning and tool extraction."""

import pytest

from myelin.core.database import Database
from myelin.core.models import AgentProfile
from myelin.transfer.profiling import AgentProfiler, extract_tools_from_text


@pytest.fixture
def db(tmp_path):
    d = Database(path=tmp_path / "test.db", enable_vec=False)
    _ = d.conn
    yield d
    d.close()


@pytest.fixture
def profiler(db):
    return AgentProfiler(db)


class TestExtractToolsFromText:
    def test_extracts_git(self):
        text = "Ran git pull and then git status"
        tools = extract_tools_from_text(text)
        assert "git" in tools

    def test_extracts_docker(self):
        text = "Built docker image and ran docker compose up"
        tools = extract_tools_from_text(text)
        assert "docker" in tools

    def test_extracts_npm(self):
        text = "npm install && npm test"
        tools = extract_tools_from_text(text)
        assert "npm" in tools

    def test_extracts_kubectl(self):
        text = "kubectl get pods"
        tools = extract_tools_from_text(text)
        assert "kubectl" in tools

    def test_extracts_pytest(self):
        text = "Running pytest on the test suite"
        tools = extract_tools_from_text(text)
        assert "pytest" in tools

    def test_deduplicates_tools(self):
        text = "git pull && git push && git status"
        tools = extract_tools_from_text(text)
        assert tools == ["git"]

    def test_returns_sorted(self):
        text = "docker build && git push && npm test"
        tools = extract_tools_from_text(text)
        assert tools == ["docker", "git", "npm"]

    def test_ignores_case(self):
        text = "GIT PULL && Docker BUILD"
        tools = extract_tools_from_text(text)
        assert "git" in tools
        assert "docker" in tools


class TestAgentProfilerLearning:
    def test_learn_from_episode_creates_profile(self, profiler):
        profiler.learn_from_episode(
            {
                "agent_id": "test_agent",
                "action": "ran tests",
                "content_text": "pytest passed all tests",
            }
        )
        profile = profiler.get("test_agent")
        assert profile is not None
        assert "pytest" in profile["tools"]

    def test_learn_from_episode_increments_usage(self, profiler):
        for _ in range(5):
            profiler.learn_from_episode(
                {
                    "agent_id": "test_agent",
                    "action": "deploy",
                    "content_text": "docker build and docker push",
                }
            )
        toolset = profiler.get_toolset("test_agent", min_usage=3)
        names = [c.tool_name for c in toolset]
        assert "docker" in names

    def test_get_toolset_respects_min_usage(self, profiler):
        profiler.learn_from_episode(
            {
                "agent_id": "test_agent",
                "action": "once",
                "content_text": "git init",
            }
        )
        # git only seen once, so min_usage=3 should exclude it
        toolset = profiler.get_toolset("test_agent", min_usage=3)
        names = [c.tool_name for c in toolset]
        assert "git" not in names

    def test_get_toolset_includes_after_threshold(self, profiler):
        for _ in range(3):
            profiler.learn_from_episode(
                {
                    "agent_id": "test_agent",
                    "action": "build",
                    "content_text": "npm run build",
                }
            )
        toolset = profiler.get_toolset("test_agent", min_usage=3)
        names = [c.tool_name for c in toolset]
        assert "npm" in names

    def test_multiple_tools_per_episode(self, profiler):
        profiler.learn_from_episode(
            {
                "agent_id": "test_agent",
                "action": "full deploy",
                "content_text": "git pull, npm test, docker build",
            }
        )
        profile = profiler.get("test_agent")
        tools = profile["tools"]
        assert "git" in tools
        assert "npm" in tools
        assert "docker" in tools

    def test_record_tool_usage_updates_profile(self, profiler):
        profiler.record_tool_usage("agent_a", "python")
        profile = profiler.get("agent_a")
        assert "python" in profile["tools"]

    def test_register_and_update(self, profiler):
        profiler.register(AgentProfile(agent_id="agent_a", tools=["git"], model_family="claude"))
        profiler.register(
            AgentProfile(agent_id="agent_a", tools=["git", "docker"], model_family="claude")
        )
        profile = profiler.get("agent_a")
        assert "docker" in profile["tools"]
        assert "git" in profile["tools"]

    def test_similarity_with_learned_tools(self, profiler):
        for _ in range(5):
            profiler.learn_from_episode(
                {
                    "agent_id": "agent_a",
                    "action": "build",
                    "content_text": "git pull && npm test",
                }
            )
            profiler.learn_from_episode(
                {
                    "agent_id": "agent_b",
                    "action": "build",
                    "content_text": "git pull && npm test",
                }
            )
        sim = profiler.compute_similarity("agent_a", "agent_b")
        assert sim > 0.8
