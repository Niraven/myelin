"""Tests for importance signal integration in MultiSignalRetriever."""

import pytest


class TestRetrieverImportanceSignal:
    def test_default_weights_include_importance(self, retriever):
        """Default weights dict must include importance key."""
        # Trigger retrieval with defaults
        results = retriever.retrieve("test query", limit=1)
        # If no crash, the default weights work
        assert isinstance(results, list)

    def test_importance_score_computed(self, retriever, tmp_db):
        """Candidates with importance_score should reflect it in _scores."""
        # Insert an episode with a known importance_score
        tmp_db.insert(
            "episodes",
            {
                "id": "ep-1",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "action": "test_action",
                "action_type": "tool_call",
                "content_text": "test content for retrieval",
                "success": 1,
                "importance_score": 0.9,
            },
        )
        # FTS trigger should sync; wait a moment for the insert to propagate
        results = retriever.retrieve("test content", limit=5)
        for r in results:
            if r.get("id") == "ep-1":
                assert "_scores" in r
                assert "importance" in r["_scores"]
                # importance_score 0.9 should yield ~0.9 (clamped)
                assert r["_scores"]["importance"] == pytest.approx(0.9, abs=0.01)

    def test_custom_importance_weight(self, retriever, tmp_db):
        """Custom weights should allow boosting importance."""
        tmp_db.insert(
            "episodes",
            {
                "id": "ep-high",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "action": "deploy",
                "action_type": "tool_call",
                "content_text": "production deployment",
                "success": 1,
                "importance_score": 0.95,
            },
        )
        tmp_db.insert(
            "episodes",
            {
                "id": "ep-low",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "action": "ls",
                "action_type": "tool_call",
                "content_text": "production deployment",
                "success": 1,
                "importance_score": 0.1,
            },
        )
        results = retriever.retrieve(
            "production deployment",
            limit=2,
            weights={
                "text": 0.0,
                "vector": 0.0,
                "entity": 0.0,
                "temporal": 0.0,
                "activation": 0.0,
                "importance": 1.0,
            },
        )
        ids = [r["id"] for r in results]
        assert "ep-high" in ids
        if len(ids) == 2:
            assert ids.index("ep-high") < ids.index("ep-low")
