"""Regression tests for the Decayer — the confidence-crater fix.

Covers the read-path deadlock protection: unvalidated candidates
(success_count < 3) must never be archived, even after long decay.
Only validated/trusted procedures (>= 3 successes) may decay to archive.
"""

import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from myelin.core.database import Database
from myelin.core.models import Procedure, ProcedureStatus, ProcedureStep, StepType
from myelin.memory.procedural import ProceduralMemory
from myelin.cognitive.decayer import (
    ARCHIVE_THRESHOLD,
    GRACE_HOURS,
    UNVALIDATED_FLOOR,
    Decayer,
)


def _iso_days_ago(days: float) -> str:
    # Naive UTC ISO, matching the runtime's datetime.utcnow().isoformat()
    return (datetime.utcnow() - timedelta(days=days)).isoformat()


def _store_proc(db: Database, procedural: ProceduralMemory, name: str, **kwargs) -> str:
    proc = Procedure(
        name=name,
        trigger_pattern=f"trigger {name}",
        steps=[ProcedureStep(order=0, description=f"{name} step", step_type=StepType.CORE)],
        confidence=kwargs.pop("confidence", 0.8),
        status=kwargs.pop("status", ProcedureStatus.ACTIVE),
        domain=kwargs.pop("domain", "test"),
        source_agent=kwargs.pop("source_agent", "test-agent"),
        **kwargs,
    )
    return procedural.store(proc)


@pytest.mark.asyncio
async def test_single_success_candidate_never_archives(tmp_db):
    """A procedure with 1 success is a candidate awaiting validation.

    After the grace period expires it may decay in salience but must stay
    ACTIVE — archiving it would remove it from the read path (the
    empty-context deadlock).
    """
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db, procedural, "single_success",
        last_executed=_iso_days_ago(30),
        success_count=1,
        failure_count=0,
        confidence=0.6,
    )

    decayer = Decayer(db)
    # Fast-forward: pretend the decayer last ran 25h ago so should_run() is True.
    db.execute(
        "INSERT INTO process_runs (process_name, status, completed_at) VALUES (?, ?, ?)",
        (decayer.name.value, "completed", _iso_days_ago(2)),
    )
    db.commit()

    result = await decayer.run()
    assert result["archived"] == 0

    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?",
        ("single_success",),
    )
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] >= UNVALIDATED_FLOOR


@pytest.mark.asyncio
async def test_validated_procedure_can_archive_after_long_idle(tmp_db):
    """A procedure with >= 3 successes is trusted; after long disuse and
    confidence below threshold it may be archived — that is the point of decay.
    """
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db, procedural, "trusted_idle",
        last_executed=_iso_days_ago(90),
        success_count=5,
        failure_count=0,
        confidence=0.9,
    )

    decayer = Decayer(db)
    db.execute(
        "INSERT INTO process_runs (process_name, status, completed_at) VALUES (?, ?, ?)",
        (decayer.name.value, "completed", _iso_days_ago(2)),
    )
    db.commit()

    result = await decayer.run()
    assert result["archived"] == 1

    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?",
        ("trusted_idle",),
    )
    assert row["status"] == ProcedureStatus.ARCHIVED.value
    assert row["confidence"] < ARCHIVE_THRESHOLD


@pytest.mark.asyncio
async def test_grace_period_skips_recently_executed(tmp_db):
    """Within the grace window a procedure is awaiting feedback — no decay."""
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db, procedural, "fresh_exec",
        last_executed=_iso_days_ago(0.1),
        success_count=1,
        failure_count=0,
        confidence=0.8,
    )

    decayer = Decayer(db)
    db.execute(
        "INSERT INTO process_runs (process_name, status, completed_at) VALUES (?, ?, ?)",
        (decayer.name.value, "completed", _iso_days_ago(2)),
    )
    db.commit()

    result = await decayer.run()
    assert result["modified"] == 0

    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?",
        ("fresh_exec",),
    )
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_candidate_decays_but_floors(tmp_db):
    """A long-idle single-success candidate decays in confidence but is
    clamped at the unvalidated floor — visible, not cratered to zero."""
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db, procedural, "crater_risk",
        last_executed=_iso_days_ago(60),
        success_count=1,
        failure_count=0,
        confidence=0.8,
    )

    decayer = Decayer(db)
    db.execute(
        "INSERT INTO process_runs (process_name, status, completed_at) VALUES (?, ?, ?)",
        (decayer.name.value, "completed", _iso_days_ago(2)),
    )
    db.commit()

    result = await decayer.run()
    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?",
        ("crater_risk",),
    )
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] == pytest.approx(UNVALIDATED_FLOOR)
    assert result["archived"] == 0
