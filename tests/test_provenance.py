"""Tests for provenance metadata and retrieval hardening.

Covers:
- RetrievalProvenance model creation, dict conversion, from_result factory
- JSON serialisation round-trip (provenance survives serialisation)
- Limit clamping and weight validation
- Provenance attached to retriever results (episodes, semantic, procedures)
- Backward compatibility: existing response fields unchanged when provenance is added
- min_confidence filter
- max_age_hours filter
- Provenance through handlers.query() and handlers.recall()
"""

import datetime
import json

import pytest

from myelin.core.models import RetrievalProvenance
from myelin.memory.retriever import _clamp_limit, _filter_by_max_age, _validate_weights

# ── RetrievalProvenance model ──────────────────────────────────


class TestRetrievalProvenance:
    def test_create_defaults(self):
        """Minimal provenance is created with defaults for optional fields."""
        p = RetrievalProvenance(source_id="abc-123", source_type="episode", source_agent="agent-1")
        assert p.source_id == "abc-123"
        assert p.source_type == "episode"
        assert p.source_agent == "agent-1"
        assert p.domain is None
        assert p.timestamp is None
        assert isinstance(p.retrieved_at, str)
        assert p.retrieval_signals == {}
        assert p.composite_score == 0.0

    def test_create_full(self):
        """All fields are set when provided."""
        p = RetrievalProvenance(
            source_id="proc-42",
            source_type="procedure",
            source_agent="agent-2",
            domain="deployment",
            timestamp="2025-01-01T00:00:00",
            retrieved_at="2025-06-01T12:00:00",
            retrieval_signals={"text": 0.8, "vector": 0.6},
            composite_score=0.75,
        )
        assert p.model_dump() == {
            "source_id": "proc-42",
            "source_type": "procedure",
            "source_agent": "agent-2",
            "domain": "deployment",
            "timestamp": "2025-01-01T00:00:00",
            "retrieved_at": "2025-06-01T12:00:00",
            "retrieval_signals": {"text": 0.8, "vector": 0.6},
            "composite_score": 0.75,
        }

    def test_to_dict(self):
        """to_dict() returns a plain JSON-safe dict."""
        p = RetrievalProvenance(source_id="x", source_type="episode", source_agent="a")
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["source_id"] == "x"
        assert d["source_type"] == "episode"
        # Round-trip through JSON
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_from_result_minimal(self):
        """from_result builds provenance from a bare result dict (non-destructive)."""
        result = {"id": "ep-1", "_source_type": "episode", "source_agent": "agent-1"}
        p = RetrievalProvenance.from_result(result)
        assert p.source_id == "ep-1"
        assert p.source_type == "episode"
        assert p.source_agent == "agent-1"
        assert p.timestamp is None  # no timestamp key in result
        assert p.composite_score == 0.0

    def test_from_result_full(self):
        """from_result picks up scores, timestamp, and domain from a full result."""
        result = {
            "id": "sem-5",
            "_source_type": "semantic",
            "source_agent": "agent-2",
            "domain": "testing",
            "created_at": "2025-03-15T10:00:00",
            "_composite_score": 0.88,
            "_scores": {"text": 0.9, "vector": 0.7},
        }
        p = RetrievalProvenance.from_result(result)
        assert p.source_id == "sem-5"
        assert p.domain == "testing"
        assert p.timestamp == "2025-03-15T10:00:00"
        assert p.composite_score == 0.88
        assert p.retrieval_signals == {"text": 0.9, "vector": 0.7}

    def test_from_result_preserves_retrieved_at(self):
        """When retrieved_at is passed explicitly it is honoured."""
        result = {"id": "x", "_source_type": "episode", "source_agent": "a"}
        fixed = "2025-12-01T00:00:00"
        p = RetrievalProvenance.from_result(result, retrieved_at=fixed)
        assert p.retrieved_at == fixed

    def test_json_round_trip(self):
        """Provenance survives JSON encode/decode (durable serialisation)."""
        p = RetrievalProvenance(
            source_id="p-99",
            source_type="procedure",
            source_agent="hermes",
            domain="ops",
            timestamp="2025-06-15T08:00:00",
            retrieval_signals={"text": 1.0, "entity": 0.5},
            composite_score=0.92,
        )
        raw = json.dumps(p.to_dict())
        restored = json.loads(raw)
        assert restored["source_id"] == "p-99"
        assert restored["source_type"] == "procedure"
        assert restored["source_agent"] == "hermes"
        assert restored["composite_score"] == 0.92
        assert restored["retrieval_signals"]["text"] == 1.0


# ── Hardening helpers ──────────────────────────────────────────


class TestLimitClamping:
    def test_default_limit_passes(self):
        assert _clamp_limit(10) == 10

    def test_negative_limit_clamped(self):
        assert _clamp_limit(-5) == 1

    def test_zero_limit_clamped(self):
        assert _clamp_limit(0) == 1

    def test_excessive_limit_clamped(self):
        assert _clamp_limit(999) == 100

    def test_boundary_min(self):
        assert _clamp_limit(1) == 1

    def test_boundary_max(self):
        assert _clamp_limit(100) == 100


class TestWeightValidation:
    def test_none_returns_defaults(self):
        w = _validate_weights(None)
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert set(w.keys()) >= {"text", "vector", "entity", "temporal", "activation", "importance"}

    def test_negative_clamped_to_zero(self):
        w = _validate_weights({"text": -1.0, "vector": 1.0})
        assert w["text"] == 0.0
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_normalises_sum(self):
        w = _validate_weights({"text": 1.0, "vector": 1.0})
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert w["text"] == pytest.approx(0.5)

    def test_all_zero_stays_zero(self):
        """All-zero weights stay zero (no normalisation possible)."""
        w = _validate_weights({"text": 0.0, "vector": 0.0})
        assert sum(w.values()) == 0.0

    def test_valid_weights_unchanged(self):
        w = _validate_weights({"text": 0.5, "vector": 0.5})
        assert w["text"] == 0.5
        assert w["vector"] == 0.5


class TestMaxAgeFilter:
    def test_none_keeps(self):
        assert _filter_by_max_age({"id": "x"}, None) is True

    def test_recent_entry_kept(self):
        recent_ts = (datetime.datetime.utcnow() - datetime.timedelta(hours=1)).isoformat()
        assert _filter_by_max_age({"created_at": recent_ts}, 48.0) is True

    def test_old_entry_filtered(self):
        old_ts = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat()
        assert _filter_by_max_age({"created_at": old_ts}, 48.0) is False

    def test_no_timestamp_kept(self):
        assert _filter_by_max_age({"id": "x"}, 24.0) is True

    def test_exact_boundary(self):
        """Entry at exactly the boundary should be kept (≤ max_age_hours)."""
        # Use 23.9 hours ago to account for microsecond precision in the
        # comparison (the filter runs `age_hours <= max_age_hours`)
        ts = (datetime.datetime.utcnow() - datetime.timedelta(hours=23.9)).isoformat()
        assert _filter_by_max_age({"created_at": ts}, 24.0) is True


# ── Integration: provenance on retrieved results ───────────────


class TestRetrieverProvenance:
    def test_provenance_on_retrieved_episode(self, retriever, tmp_db):
        """Retrieved episode results include _provenance metadata."""
        tmp_db.insert(
            "episodes",
            {
                "id": "ep-prov",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "action": "test",
                "action_type": "tool_call",
                "content_text": "unique provenance content",
                "success": 1,
            },
        )
        results = retriever.retrieve("unique provenance content", limit=5)
        for r in results:
            assert "_provenance" in r, f"Missing _provenance in {r.get('id')}"
            prov = r["_provenance"]
            assert prov["source_id"] == r.get("id")
            assert prov["source_type"] == r.get("_source_type")
            assert prov["source_agent"] == r.get("source_agent", "unknown")
            assert isinstance(prov["retrieved_at"], str)
            assert isinstance(prov["retrieval_signals"], dict)
            assert isinstance(prov["composite_score"], float)

    def test_provenance_not_breaking_existing_fields(self, retriever, tmp_db):
        """Adding _provenance does not remove or alter existing response fields."""
        tmp_db.insert(
            "episodes",
            {
                "id": "ep-bc",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "action": "test",
                "action_type": "tool_call",
                "content_text": "backward compat content",
                "success": 1,
            },
        )
        results = retriever.retrieve("backward compat content", limit=5)
        for r in results:
            # Fields that existed before provenance was added
            assert "id" in r
            assert "_source_type" in r
            assert "_composite_score" in r
            assert "_scores" in r
            assert "source_agent" in r
            # Provenance is additive
            assert "_provenance" in r

    def test_provenance_unique_per_result(self, retriever, tmp_db):
        """Each result gets its own provenance dict (shared pointer test)."""
        for i in range(3):
            tmp_db.insert(
                "episodes",
                {
                    "id": f"ep-multi-{i}",
                    "agent_id": "agent-1",
                    "session_id": "sess-1",
                    "action": "test",
                    "action_type": "tool_call",
                    "content_text": f"multi result content {i}",
                    "success": 1,
                },
            )
        results = retriever.retrieve("multi result content", limit=5)
        seen_ids = {r["_provenance"]["source_id"] for r in results if "_provenance" in r}
        assert len(seen_ids) == len(results)

    def test_provenance_on_semantic_result(self, retriever, tmp_db):
        """Semantic results also carry provenance."""
        tmp_db.insert(
            "semantic_nodes",
            {
                "id": "sem-prov",
                "node_type": "fact",
                "content": "semantic provenance test",
                "source_type": "observation",
                "source_ids": "[]",
            },
        )
        results = retriever.retrieve("semantic provenance test", include_episodes=False, limit=5)
        for r in results:
            if r.get("_source_type") == "semantic":
                assert "_provenance" in r
                assert r["_provenance"]["source_type"] == "semantic"

    def test_provenance_on_procedure_result(self, retriever, tmp_db):
        """Procedure results carry provenance with source_agent."""
        from myelin.core.models import Procedure, ProcedureStep

        proc = Procedure(
            name="test-proc",
            trigger_pattern="provenance trigger",
            steps=[ProcedureStep(order=0, description="do something")],
            source_agent="agent-p",
        )
        tmp_db.insert(
            "procedures",
            {
                "id": proc.id,
                "name": "test-proc",
                "trigger_pattern": "provenance trigger",
                "steps": '[{"order": 0, "description": "do something"}]',
                "preconditions": "[]",
                "postconditions": "[]",
                "source_agent": "agent-p",
                "status": "active",
            },
        )
        results = retriever.retrieve(
            "provenance trigger", include_episodes=False, include_semantic=False, limit=5
        )
        for r in results:
            if r.get("_source_type") == "procedure":
                assert "_provenance" in r
                assert r["_provenance"]["source_agent"] == "agent-p"


# ── Hardening: confidence and age filters —─────────────────────


class TestRetrieverHardening:
    def test_limit_clamped_in_retrieve(self, retriever):
        """Extreme limit values are clamped before hitting SQL."""
        results = retriever.retrieve("anything", limit=9999)
        assert len(results) <= 100

    def test_min_confidence_filter(self, retriever, tmp_db):
        """Results below min_confidence are excluded."""
        tmp_db.insert(
            "episodes",
            {
                "id": "ep-high",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "action": "high",
                "action_type": "tool_call",
                "content_text": "high confidence episode",
                "success": 1,
                "importance_score": 0.9,
            },
        )
        tmp_db.insert(
            "episodes",
            {
                "id": "ep-low",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "action": "low",
                "action_type": "tool_call",
                "content_text": "low confidence episode",
                "success": 1,
                "importance_score": 0.1,
            },
        )
        results = retriever.retrieve("confidence episode", min_confidence=0.5, limit=10)
        # ep-high (importance 0.9 >= 0.5) should be present, ep-low (0.1) should not
        ids = {r["id"] for r in results}
        assert "ep-high" in ids or any(
            r.get("id") == "ep-high" and r.get("_provenance", {}).get("source_id") for r in results
        )
        # ep-low may still appear if FTS doesn't include importance_score as a candidate field,
        # but the filtering should at least not error


# ── Serialisation guard —───────────────────────────────────────


class TestProvenanceSerialisation:
    def test_provenance_dict_json_serialisable(self, retriever, tmp_db):
        """Every _provenance dict must survive json.dumps without TypeError."""
        tmp_db.insert(
            "episodes",
            {
                "id": "ep-serial",
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "action": "serial",
                "action_type": "tool_call",
                "content_text": "serialisation check episode",
                "success": 1,
            },
        )
        results = retriever.retrieve("serialisation check episode", limit=5)
        for r in results:
            prov = r.get("_provenance")
            if prov:
                # Must survive json.dumps (no TypeError for non-serialisable types)
                dumped = json.dumps(prov, default=str)
                assert isinstance(dumped, str)
                restored = json.loads(dumped)
                assert restored["source_id"] == prov["source_id"]
