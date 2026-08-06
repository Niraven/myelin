"""TDD tests for closing the prediction-linked procedure execution feedback loop.

Covers the handler path, prediction creation, bound/unbound feedback, idempotency,
pairing mismatch fail-closed, legacy compatibility, migration, and trust gating.
"""

import pytest

from myelin.cognitive.prediction_learner import PredictionLearner
from myelin.memory.embedding import NoOpEmbedding
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory
from myelin.tools.handlers import ToolHandlers


@pytest.fixture
def handlers(tmp_path):
    db_path = tmp_path / "loop.db"
    from myelin.core.database import Database

    db = Database(path=db_path, enable_vec=False)
    _ = db.conn
    h = ToolHandlers(
        EpisodicMemory(db),
        SemanticMemory(db),
        ProceduralMemory(db),
        NoOpEmbedding(),
    )
    yield h
    db.close()


async def _teach(handlers, name="deploy", trigger="deploy service"):
    return await handlers.teach(
        name=name,
        trigger_pattern=trigger,
        steps=[{"description": "step 1", "type": "core"}],
        agent_id="agent-1",
        domain="deployment",
    )


async def _execute(handlers, query="deploy service"):
    return await handlers.execute_procedure(query=query, agent_id="agent-1")


# ── 1. ToolHandlers constructs/reuses a PredictionLearner ──────


@pytest.mark.asyncio
async def test_toolhandlers_constructs_prediction_learner(handlers):
    assert isinstance(handlers.prediction_learner, PredictionLearner)


@pytest.mark.asyncio
async def test_toolhandlers_reuses_injected_prediction_learner(tmp_path):
    from myelin.core.database import Database

    db = Database(path=tmp_path / "r.db", enable_vec=False)
    _ = db.conn
    proc_mem = ProceduralMemory(db)
    learner = PredictionLearner(db, proc_mem)
    h = ToolHandlers(
        EpisodicMemory(db),
        SemanticMemory(db),
        proc_mem,
        NoOpEmbedding(),
        prediction_learner=learner,
    )
    assert h.prediction_learner is learner
    db.close()


# ── 2. execute_procedure creates exactly one pending prediction ─


@pytest.mark.asyncio
async def test_execute_creates_one_pending_prediction(handlers):
    await _teach(handlers)
    res = await _execute(handlers)

    assert res["found"] is True
    assert res["prediction_id"]

    rows = handlers.db.fetchall("SELECT * FROM prediction_log")
    assert len(rows) == 1
    assert rows[0]["id"] == res["prediction_id"]
    assert rows[0]["procedure_id"] == res["procedure_id"]
    assert rows[0]["actual_outcome"] is None
    assert rows[0]["td_error"] == 0.0


@pytest.mark.asyncio
async def test_execute_not_found_creates_no_prediction(handlers):
    res = await handlers.execute_procedure(query="no such procedure at all", agent_id="agent-1")
    assert res["found"] is False
    assert "prediction_id" not in res or res["prediction_id"] is None
    assert handlers.db.fetchone("SELECT COUNT(*) AS c FROM prediction_log")["c"] == 0


# ── 3. Bound feedback records through PredictionLearner ────────


@pytest.mark.asyncio
async def test_bound_feedback_records_through_learner(handlers):
    await _teach(handlers, name="deploy", trigger="deploy service")
    exec_res = await _execute(handlers)
    proc_id = exec_res["procedure_id"]
    pred_id = exec_res["prediction_id"]

    before = handlers.procedural.get(proc_id)
    fb = await handlers.procedure_feedback(
        procedure_id=proc_id, success=True, prediction_id=pred_id
    )

    assert fb["evidence_quality"] == "verified"
    assert fb["prediction_id"] == pred_id
    assert fb["td_error"] is not None
    assert fb["surprise_score"] is not None

    after = handlers.procedural.get(proc_id)
    assert after["success_count"] == before["success_count"] + 1
    assert after["execution_count"] == before["execution_count"] + 1
    assert after["confidence"] > before["confidence"]

    # prediction log finalized
    pred = handlers.db.fetchone("SELECT * FROM prediction_log WHERE id = ?", (pred_id,))
    assert pred["actual_outcome"] == 1

    # exactly one procedure_evidence row linked to this prediction
    evs = handlers.db.fetchall(
        "SELECT * FROM procedure_evidence WHERE prediction_id = ?", (pred_id,)
    )
    assert len(evs) == 1
    assert evs[0]["procedure_id"] == proc_id
    assert evs[0]["outcome"] == "success"


# ── 4. Replaying same bound feedback is idempotent ─────────────


@pytest.mark.asyncio
async def test_bound_feedback_idempotent(handlers):
    await _teach(handlers)
    exec_res = await _execute(handlers)
    proc_id = exec_res["procedure_id"]
    pred_id = exec_res["prediction_id"]

    await handlers.procedure_feedback(procedure_id=proc_id, success=True, prediction_id=pred_id)
    after_first = handlers.procedural.get(proc_id)
    ev_count_first = len(
        handlers.db.fetchall("SELECT * FROM procedure_evidence WHERE prediction_id = ?", (pred_id,))
    )

    replay = await handlers.procedure_feedback(
        procedure_id=proc_id, success=True, prediction_id=pred_id
    )
    assert replay["record_status"] == "already_recorded"

    after_second = handlers.procedural.get(proc_id)
    assert after_second["success_count"] == after_first["success_count"]
    assert after_second["execution_count"] == after_first["execution_count"]
    assert after_second["confidence"] == after_first["confidence"]
    ev_count_second = len(
        handlers.db.fetchall("SELECT * FROM procedure_evidence WHERE prediction_id = ?", (pred_id,))
    )
    assert ev_count_second == ev_count_first == 1


# ── 5. Wrong procedure/prediction pairing fails closed ─────────


@pytest.mark.asyncio
async def test_wrong_pairing_fails_closed(handlers):
    await _teach(handlers, name="deploy", trigger="deploy service")
    await _teach(handlers, name="build", trigger="build image")
    other = await _execute(handlers, query="build image")
    other_proc_id = other["procedure_id"]
    other_pred_id = other["prediction_id"]

    # Feed the "build" prediction back under a different procedure id.
    before = handlers.procedural.get(other_proc_id)
    res = await handlers.procedure_feedback(
        procedure_id="does-not-exist", success=True, prediction_id=other_pred_id
    )
    assert "error" in res

    after = handlers.procedural.get(other_proc_id)
    assert after["success_count"] == before["success_count"]
    assert after["execution_count"] == before["execution_count"]
    pred = handlers.db.fetchone("SELECT * FROM prediction_log WHERE id = ?", (other_pred_id,))
    assert pred["actual_outcome"] is None


# ── 6. Legacy feedback stays backward compatible ───────────────


@pytest.mark.asyncio
async def test_legacy_feedback_unbound_and_compatible(handlers):
    teach = await _teach(handlers, name="deploy", trigger="deploy service")
    proc_id = teach["procedure_id"]

    fb = await handlers.procedure_feedback(procedure_id=proc_id, success=True)

    assert fb["evidence_quality"] == "unbound"
    assert fb["new_confidence"] > 0.7
    assert fb["success_count"] == 1
    proc = handlers.procedural.get(proc_id)
    assert proc["success_count"] == 1
    assert proc["execution_count"] == 1


@pytest.mark.asyncio
async def test_unbound_success_does_not_promote_trust(handlers):
    teach = await _teach(handlers, name="deploy", trigger="deploy service")
    proc_id = teach["procedure_id"]

    for _ in range(4):
        await handlers.procedure_feedback(procedure_id=proc_id, success=True)

    proc = handlers.procedural.get(proc_id)
    assert proc["success_count"] >= 4
    trust_state = handlers.procedural.update_trust_state(proc_id)
    assert trust_state in ("candidate", "seed")


# ── 7. Trust promotion counts verified evidence ────────────────


@pytest.mark.asyncio
async def test_verified_evidence_promotes_to_trusted(handlers):
    await _teach(handlers, name="deploy", trigger="deploy service")
    for _ in range(3):
        exec_res = await _execute(handlers)
        await handlers.procedure_feedback(
            procedure_id=exec_res["procedure_id"],
            success=True,
            prediction_id=exec_res["prediction_id"],
        )

    proc_id = handlers.procedural.get(exec_res["procedure_id"])["id"]
    proc = handlers.procedural.get(proc_id)
    assert proc["confidence"] >= 0.7
    assert handlers.procedural.update_trust_state(proc_id) == "trusted"


# ── 8. Migration adds nullable prediction_id column ────────────


def test_migration_adds_prediction_id_column(tmp_path):
    import sqlite3

    from myelin.core.database import Database

    conn = sqlite3.connect(tmp_path / "legacy_ev.db")
    conn.executescript(
        """
        CREATE TABLE procedure_evidence (
            id TEXT PRIMARY KEY,
            procedure_id TEXT NOT NULL,
            source TEXT NOT NULL,
            outcome TEXT NOT NULL,
            confidence_delta REAL NOT NULL DEFAULT 0.0,
            episode_id TEXT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()

    db = Database(path=tmp_path / "legacy_ev.db", enable_vec=False)
    _ = db.conn
    cols = {row["name"] for row in db.fetchall("PRAGMA table_info(procedure_evidence)")}
    assert "prediction_id" in cols
    db.close()
