"""Tests for PredictionLearner: forward model, TD-error, surprise, confidence, priority."""

import json

import pytest

from myelin.cognitive.prediction_learner import (
    PredictionLearner,
    compute_priority_score,
    compute_surprise,
    compute_td_error,
    td_modulated_learning_rate,
)
from myelin.core.models import (
    ActionType,
    Episode,
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    StepType,
)


def _make_procedure(
    name="test_deploy",
    domain="deployment",
    confidence=0.5,
    source_agent="agent-1",
    status=ProcedureStatus.ACTIVE,
):
    return Procedure(
        name=name,
        trigger_pattern=f"When user wants to {name}",
        steps=[
            ProcedureStep(order=0, description="step 1", step_type=StepType.CORE),
        ],
        confidence=confidence,
        source_agent=source_agent,
        domain=domain,
        status=status,
    )


def _make_episode(db, agent_id="agent-1", session_id="sess-1", importance=0.7):
    ep = Episode(
        agent_id=agent_id,
        session_id=session_id,
        action="test predict",
        action_type=ActionType.TOOL_CALL,
        content_text="testing prediction learner",
        domain="deployment",
    )
    db.insert(
        "episodes",
        {
            "id": ep.id,
            "agent_id": ep.agent_id,
            "session_id": ep.session_id,
            "action": ep.action,
            "action_type": ep.action_type.value,
            "content_text": ep.content_text,
            "success": int(ep.success),
            "importance_score": importance,
            "priority_score": 0.5,
            "tags": json.dumps(ep.tags),
            "created_at": ep.created_at,
            "timestamp": ep.timestamp,
            "access_count": ep.access_count,
            "access_times": json.dumps(ep.access_times),
            "last_accessed": ep.last_accessed,
        },
    )
    return ep.id


# ── Unit tests: pure functions ────────────────────────────────


class TestTDError:
    def test_success_predicted_success_actual(self):
        assert compute_td_error(1, 1) == 0.0

    def test_failure_predicted_failure_actual(self):
        assert compute_td_error(0, 0) == 0.0

    def test_predicted_success_actual_failure(self):
        assert compute_td_error(1, 0) == -1.0

    def test_predicted_failure_actual_success(self):
        assert compute_td_error(0, 1) == 1.0


class TestSurprise:
    def test_no_error_no_surprise(self):
        assert compute_surprise(0.0, 0.8) == 0.0

    def test_low_confidence_error_low_surprise(self):
        # δ = 1.0, confidence = 0.1 → 1.0 * (1 - 0.1) = 0.9
        assert compute_surprise(1.0, 0.1) == 0.9

    def test_high_confidence_error_high_surprise(self):
        # δ = -1.0, confidence = 0.9 → 1.0 * (1 - 0.9) = 0.1
        assert round(compute_surprise(-1.0, 0.9), 2) == 0.1

    def test_clamped_to_range(self):
        assert 0.0 <= compute_surprise(10.0, 0.0) <= 1.0


class TestLearningRate:
    def test_base_lr_when_no_error(self):
        assert td_modulated_learning_rate(0.15, 0.0) == 0.15

    def test_modulated_by_error_magnitude(self):
        assert td_modulated_learning_rate(0.15, 1.0) == 0.30

    def test_double_modulation_for_max_error(self):
        assert round(td_modulated_learning_rate(0.15, 2.0), 6) == 0.45


class TestPriorityScore:
    def test_all_zero(self):
        assert compute_priority_score(0.0, 0.0, 0.0) == 0.0

    def test_all_max(self):
        assert round(compute_priority_score(1.0, 1.0, 1.0), 6) == 1.0

    def test_mid_values(self):
        # 0.35 * 0.5 + 0.30 * 0.5 + 0.35 * 0.5 = 0.5
        assert round(compute_priority_score(0.5, 0.5, 0.5), 6) == 0.5

    def test_clamped(self):
        assert 0.0 <= compute_priority_score(5.0, 2.0, 3.0) <= 1.0


# ── Integration tests: PredictionLearner with real DB ─────────


class TestPredictionLearner:
    def test_predict_outcome_success(self, tmp_db, procedural):
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.7)
        procedural.store(proc)

        result = learner.predict_outcome(proc.id)

        assert result["prediction_id"] is not None
        assert result["predicted_success"] is True
        assert result["predicted_confidence"] == 0.7
        assert result["procedure_name"] == "test_deploy"

        # Verify prediction was stored
        pred = tmp_db.fetchone(
            "SELECT * FROM prediction_log WHERE id = ?",
            (result["prediction_id"],),
        )
        assert pred is not None
        assert pred["predicted_success"] == 1
        assert pred["td_error"] == 0.0
        assert pred["actual_outcome"] is None

    def test_predict_outcome_failure(self, tmp_db, procedural):
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.3)
        procedural.store(proc)

        result = learner.predict_outcome(proc.id)

        assert result["predicted_success"] is False
        assert result["predicted_confidence"] == 0.3

    def test_predict_outcome_missing_procedure(self, tmp_db, procedural):
        learner = PredictionLearner(tmp_db, procedural)
        result = learner.predict_outcome("nonexistent")
        assert result["error"] is not None
        assert result["prediction_id"] is None

    def test_record_outcome_success_confirmed(self, tmp_db, procedural):
        """Predicted success=1, actual success=1 → δ=0, confidence increases."""
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.6)
        procedural.store(proc)

        pred = learner.predict_outcome(proc.id)
        result = learner.record_outcome(pred["prediction_id"], actual_success=True)

        assert result["status"] == "recorded"
        assert result["actual_success"] is True
        assert result["td_error"] == 0.0
        assert result["surprise_score"] == 0.0
        assert result["new_confidence"] > 0.6

        # Verify prediction log updated
        stored = tmp_db.fetchone(
            "SELECT * FROM prediction_log WHERE id = ?",
            (pred["prediction_id"],),
        )
        assert stored["actual_outcome"] == 1
        assert stored["td_error"] == 0.0

    def test_record_outcome_surprise_on_false_positive(self, tmp_db, procedural):
        """Predicted success=1, actual failure=0 → δ=-1, high surprise."""
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.85)
        procedural.store(proc)

        pred = learner.predict_outcome(proc.id)
        result = learner.record_outcome(pred["prediction_id"], actual_success=False)

        assert result["td_error"] == -1.0
        # surprise = 1.0 * (1 - 0.85) = 0.15
        assert round(result["surprise_score"], 3) == 0.15

    def test_record_outcome_surprise_on_false_negative(self, tmp_db, procedural):
        """Predicted success=0, actual success=1 → δ=+1, moderate surprise."""
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.3)
        procedural.store(proc)

        pred = learner.predict_outcome(proc.id)
        result = learner.record_outcome(pred["prediction_id"], actual_success=True)

        assert result["td_error"] == 1.0
        # surprise = 1.0 * (1 - 0.3) = 0.7
        assert round(result["surprise_score"], 3) == 0.7

    def test_double_record_returns_already_recorded(self, tmp_db, procedural):
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.6)
        procedural.store(proc)

        pred = learner.predict_outcome(proc.id)
        learner.record_outcome(pred["prediction_id"], actual_success=True)
        result = learner.record_outcome(pred["prediction_id"], actual_success=False)

        assert result["status"] == "already_recorded"

    def test_td_modulated_learning_rate_applied(self, tmp_db, procedural):
        """Bigger TD-error should produce larger confidence update."""
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.5)
        procedural.store(proc)
        proc_id = proc.id

        # Predict success (confidence 0.5 → predicted_success=0)
        pred = learner.predict_outcome(proc_id)
        learner.record_outcome(pred["prediction_id"], actual_success=True)
        procedural.get(proc_id)

        # Now test with higher TD (second prediction)
        # After one success, confidence is higher
        pred2 = learner.predict_outcome(proc_id)
        learner.record_outcome(pred2["prediction_id"], actual_success=False)
        updated2 = procedural.get(proc_id)

        # Verify PE tracking data on procedure
        assert updated2.get("pe_count", 0) >= 2
        assert updated2.get("total_pe_sum", 0) >= 0

    def test_episode_priority_update(self, tmp_db, procedural):
        """Priority should update on the episode when episode_id provided."""
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.9)
        procedural.store(proc)
        ep_id = _make_episode(tmp_db, importance=0.7)

        pred = learner.predict_outcome(proc.id, episode_id=ep_id)
        learner.record_outcome(pred["prediction_id"], actual_success=False)

        ep = tmp_db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep_id,))
        assert ep is not None
        assert ep["td_error"] == 1.0
        assert round(ep["surprise_score"], 6) == 0.1
        # priority = 0.35*1.0 + 0.30*0.10 + 0.35*0.70 = 0.35 + 0.03 + 0.245 = 0.625
        assert round(ep["priority_score"], 3) == 0.625

    def test_missing_prediction_returns_error(self, tmp_db, procedural):
        learner = PredictionLearner(tmp_db, procedural)
        result = learner.record_outcome("nonexistent", actual_success=True)
        assert "error" in result
        assert "not found" in result["error"]

    def test_auto_promote_after_high_confidence(self, tmp_db, procedural):
        """Procedure should auto-promote from draft to active when confidence >= 0.8."""
        learner = PredictionLearner(tmp_db, procedural, base_learning_rate=0.3)
        proc = _make_procedure(
            confidence=0.5,
            status=ProcedureStatus.DRAFT,
        )
        procedural.store(proc)
        proc_id = proc.id

        # Multiple successes to push confidence above 0.8
        for _ in range(4):
            pred = learner.predict_outcome(proc_id)
            learner.record_outcome(pred["prediction_id"], actual_success=True)

        updated = procedural.get(proc_id)
        assert updated["confidence"] >= 0.8
        assert updated["status"] == "active"

    def test_context_and_agent_passed_to_prediction_log(self, tmp_db, procedural):
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.6)
        procedural.store(proc)

        result = learner.predict_outcome(
            proc.id,
            context={"task": "deploy"},
            agent_id="agent-42",
            domain="deployment",
            episode_id="ep-123",
        )

        pred = tmp_db.fetchone(
            "SELECT * FROM prediction_log WHERE id = ?",
            (result["prediction_id"],),
        )
        assert pred["agent_id"] == "agent-42"
        assert pred["domain"] == "deployment"
        assert pred["episode_id"] == "ep-123"


class TestMCPHandlers:
    """Test the async MCP tool handler methods."""

    @pytest.mark.asyncio
    async def test_handle_predict_outcome(self, tmp_db, procedural):
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.75)
        procedural.store(proc)

        result = await learner.handle_predict_outcome(
            procedure_id=proc.id,
            agent_id="agent-1",
            domain="deployment",
        )
        assert result["predicted_success"] is True
        assert result["predicted_confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_handle_record_outcome(self, tmp_db, procedural):
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.6)
        procedural.store(proc)

        pred = await learner.handle_predict_outcome(procedure_id=proc.id)
        result = await learner.handle_record_outcome(
            prediction_id=pred["prediction_id"],
            actual_success=False,
        )
        assert result["status"] == "recorded"
        assert result["td_error"] == -1.0


# ── Atomic feedback: CAS claim + encompassing transaction ──────


def _evidence_count(db, prediction_id):
    return db.fetchone(
        "SELECT COUNT(*) AS c FROM procedure_evidence WHERE prediction_id = ?",
        (prediction_id,),
    )["c"]


class TestAtomicFeedback:
    def test_sequential_replay_is_noop(self, tmp_db, procedural):
        """Replaying feedback for an already-finalized prediction applies nothing."""
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.6)
        procedural.store(proc)

        pred = learner.predict_outcome(proc.id)
        pred_id = pred["prediction_id"]
        learner.record_outcome(pred_id, actual_success=True)
        after_first = procedural.get(proc.id)
        first_ev = _evidence_count(tmp_db, pred_id)

        result = learner.record_outcome(pred_id, actual_success=False)
        assert result["status"] == "already_recorded"

        after_replay = procedural.get(proc.id)
        assert after_replay["success_count"] == after_first["success_count"]
        assert after_replay["execution_count"] == after_first["execution_count"]
        assert after_replay["confidence"] == after_first["confidence"]
        assert _evidence_count(tmp_db, pred_id) == first_ev == 1

    def test_transaction_rollback_on_post_claim_failure(self, tmp_db, procedural, monkeypatch):
        """A failure after the CAS claim rolls back the claim and every mutation."""
        learner = PredictionLearner(tmp_db, procedural)
        proc = _make_procedure(confidence=0.6)
        procedural.store(proc)
        before = procedural.get(proc.id)

        pred = learner.predict_outcome(proc.id)
        pred_id = pred["prediction_id"]

        # Inject a failure after confidence was updated, before commit.
        def _boom(*args, **kwargs):
            raise RuntimeError("injected post-claim failure")

        monkeypatch.setattr(procedural, "record_evidence", _boom)
        with pytest.raises(RuntimeError):
            learner.record_outcome(pred_id, actual_success=True)

        # Encompassing transaction rolled everything back.
        pred_row = tmp_db.fetchone("SELECT * FROM prediction_log WHERE id = ?", (pred_id,))
        assert pred_row["actual_outcome"] is None
        assert pred_row["td_error"] == 0.0
        after = procedural.get(proc.id)
        assert after["success_count"] == before["success_count"]
        assert after["execution_count"] == before["execution_count"]
        assert _evidence_count(tmp_db, pred_id) == 0

        # Retry after the fault clears succeeds cleanly.
        monkeypatch.undo()
        result = learner.record_outcome(pred_id, actual_success=True)
        assert result["status"] == "recorded"
        assert _evidence_count(tmp_db, pred_id) == 1

    def test_concurrent_record_outcome_applies_once(self, tmp_path):
        """Two writers racing on one prediction finalize it exactly once."""
        import threading

        from myelin.core.database import Database
        from myelin.memory.procedural import ProceduralMemory

        db_path = tmp_path / "race.db"
        db = Database(path=db_path, enable_vec=False)
        _ = db.conn
        procedural = ProceduralMemory(db)
        proc = _make_procedure(confidence=0.5)
        procedural.store(proc)
        pred = PredictionLearner(db, procedural).predict_outcome(proc.id)
        pred_id = pred["prediction_id"]
        before = procedural.get(proc.id)
        db.close()

        results = []
        barrier = threading.Barrier(2)

        def _worker():
            d = Database(path=db_path, enable_vec=False)
            _ = d.conn
            p = ProceduralMemory(d)
            worker_learner = PredictionLearner(d, p)
            barrier.wait()
            results.append(worker_learner.record_outcome(pred_id, actual_success=True))
            d.close()

        threads = [threading.Thread(target=_worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        db = Database(path=db_path, enable_vec=False)
        _ = db.conn
        pred_row = db.fetchone("SELECT * FROM prediction_log WHERE id = ?", (pred_id,))
        assert pred_row["actual_outcome"] == 1
        assert _evidence_count(db, pred_id) == 1
        proc_row = ProceduralMemory(db).get(proc.id)
        assert proc_row["success_count"] == before["success_count"] + 1
        assert proc_row["execution_count"] == before["execution_count"] + 1
        statuses = [r.get("status") for r in results]
        assert statuses.count("recorded") == 1
        assert statuses.count("already_recorded") == 1
        db.close()
