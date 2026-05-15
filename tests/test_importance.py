import os
import unittest
from datetime import datetime, timezone

from myelin.cognitive.importance import score_clusters, score_episodes, temporal_decay


class ImportanceScoringTests(unittest.TestCase):
    def test_default_scores_are_frequency_only(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        episodes = [
            {"cluster_id": "alpha", "success": True, "timestamp": now - 10},
            {"cluster_id": "alpha", "success": True, "timestamp": now - 20},
            {"cluster_id": "beta", "success": False, "timestamp": now - 5},
        ]

        scores = score_clusters(episodes)
        self.assertEqual(scores, {"alpha": 2.0, "beta": 1.0})

    def test_configurable_weights_include_consequence(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        episodes = [
            {"cluster_id": "low", "success": False, "timestamp": now - 60},
            {"cluster_id": "low", "success": False, "timestamp": now - 120},
            {"cluster_id": "high", "success": True, "timestamp": now - 60},
        ]

        scores = score_clusters(
            episodes,
            weights={"frequency": 1, "consequence": 1, "recency": 0},
            now=now,
            use_normalized_default=True,
        )
        # low: frequency dominates at 1.0, consequence 0.0 -> 0.5
        # high: freq 0.5, consequence 1.0 -> 0.75
        self.assertGreater(scores["high"], scores["low"])

    def test_recency_controls_but_frequency_only_is_unchanged(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        episodes = [
            {"cluster_id": "fresh", "success": True, "timestamp": now - 100},
            {"cluster_id": "stale", "success": True, "timestamp": now - 7 * 24 * 3600},
        ]

        scores = score_clusters(
            episodes,
            weights={"frequency": 0, "consequence": 0, "recency": 1},
            now=now,
            use_normalized_default=True,
        )
        self.assertGreater(scores["fresh"], scores["stale"])

    def test_temporal_decay_half_life(self):
        now = 11.0
        last_seen = 1.0
        # Half-life of 10 seconds => 10s old should decay to 0.5 exactly.
        self.assertAlmostEqual(temporal_decay(last_seen_timestamp=last_seen, now=now, half_life_seconds=10), 2 ** (-(10 / 10)))

    def test_env_weight_override(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        episodes = [
            {"cluster_id": "high", "success": True, "timestamp": now - 100},
            {"cluster_id": "low", "success": False, "timestamp": now - 100},
        ]

        os.environ["IMPORTANCE_WEIGHT_FREQUENCY"] = "0"
        os.environ["IMPORTANCE_WEIGHT_CONSEQUENCE"] = "1"
        os.environ["IMPORTANCE_WEIGHT_RECENCY"] = "0"

        scores = score_clusters(episodes, now=now, use_normalized_default=True)
        self.assertGreater(scores["high"], scores["low"])

        del os.environ["IMPORTANCE_WEIGHT_FREQUENCY"]
        del os.environ["IMPORTANCE_WEIGHT_CONSEQUENCE"]
        del os.environ["IMPORTANCE_WEIGHT_RECENCY"]


if __name__ == "__main__":
    unittest.main()
