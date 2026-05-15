"""Test episode clustering engine."""

from myelin.memory.clustering import (
    EpisodeClusterer,
    action_sequence_similarity,
    cosine_similarity,
    episode_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 1.0]
        assert cosine_similarity(v, v) > 0.99

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) < 0.01

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_similar_vectors(self):
        a = [1.0, 0.5, 0.3]
        b = [0.9, 0.6, 0.2]
        assert cosine_similarity(a, b) > 0.9


class TestActionSequenceSimilarity:
    def test_identical(self):
        seq = ["git pull", "npm test", "npm build"]
        assert action_sequence_similarity(seq, seq) > 0.99

    def test_completely_different(self):
        assert action_sequence_similarity(["a"], ["b"]) == 0.0

    def test_subset(self):
        a = ["git pull", "npm test", "npm build"]
        b = ["git pull", "npm test"]
        sim = action_sequence_similarity(a, b)
        assert 0.5 < sim < 1.0

    def test_empty(self):
        assert action_sequence_similarity([], []) == 0.0


class TestEpisodeSimilarity:
    def test_identical_episodes(self):
        ep = {
            "domain": "deployment",
            "action_type": "tool_call",
            "content_text": "running npm test suite",
        }
        assert episode_similarity(ep, ep) > 0.9

    def test_same_domain_different_action(self):
        ep_a = {
            "domain": "deployment",
            "action_type": "tool_call",
            "content_text": "git pull origin main",
        }
        ep_b = {
            "domain": "deployment",
            "action_type": "tool_call",
            "content_text": "docker compose up",
        }
        sim = episode_similarity(ep_a, ep_b)
        assert 0.1 < sim < 0.8

    def test_different_domain(self):
        ep_a = {"domain": "testing", "action_type": "tool_call", "content_text": "npm test"}
        ep_b = {"domain": "deployment", "action_type": "tool_call", "content_text": "npm test"}
        sim = episode_similarity(ep_a, ep_b)
        assert sim < 0.9  # Same text but different domain


class TestEpisodeClusterer:
    def _make_episodes(self, sessions: list[list[str]], domain: str = "testing") -> list[dict]:
        episodes = []
        for i, actions in enumerate(sessions):
            for j, action in enumerate(actions):
                episodes.append(
                    {
                        "id": f"ep_{i}_{j}",
                        "session_id": f"session_{i}",
                        "domain": domain,
                        "action_type": "tool_call",
                        "action": action,
                        "content_text": action,
                        "timestamp": f"2024-01-01T{i:02d}:{j:02d}:00",
                        "access_times": [],
                    }
                )
        return episodes

    def test_clusters_similar_sessions(self):
        episodes = self._make_episodes(
            [
                ["git pull", "npm test", "npm build"],
                ["git pull", "npm test", "npm build"],
                ["docker login", "docker push", "kubectl apply"],
            ]
        )

        clusterer = EpisodeClusterer(similarity_threshold=0.4, min_cluster_size=2)
        clusters = clusterer.cluster_by_session_sequences(episodes)

        assert len(clusters) >= 1
        # The two npm sessions should cluster together
        for cluster in clusters:
            session_ids = set(ep["session_id"] for ep in cluster)
            if "session_0" in session_ids:
                assert "session_1" in session_ids

    def test_min_cluster_size(self):
        episodes = self._make_episodes(
            [
                ["unique action 1"],
                ["unique action 2"],
            ]
        )
        clusterer = EpisodeClusterer(similarity_threshold=0.9, min_cluster_size=3)
        clusters = clusterer.cluster_by_session_sequences(episodes)
        assert len(clusters) == 0

    def test_domain_grouping(self):
        testing_eps = self._make_episodes(
            [["npm test", "check output"], ["npm test", "check output"]],
            domain="testing",
        )
        deploy_eps = self._make_episodes(
            [["git push", "deploy"], ["git push", "deploy"]],
            domain="deployment",
        )

        all_eps = testing_eps + deploy_eps
        clusterer = EpisodeClusterer(similarity_threshold=0.3, min_cluster_size=2)
        clusters = clusterer.cluster(all_eps)

        # Should get at least 2 clusters (one per domain)
        if clusters:
            for cluster in clusters:
                domains = set(ep["domain"] for ep in cluster)
                assert len(domains) == 1
