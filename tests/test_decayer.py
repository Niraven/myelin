"""Regression tests for the Decayer — the confidence-crater fix.

Covers the read-path deadlock protection: unvalidated candidates
(success_count < 3) must never be archived, even after long decay.
Only validated/trusted procedures (>= 3 successes) may decay to archive.
"""

from datetime import datetime, timedelta

import pytest

from myelin.cognitive.decayer import (
    ARCHIVE_THRESHOLD,
    GRACE_HOURS,
    UNVALIDATED_FLOOR,
    Decayer,
)
from myelin.core.database import Database
from myelin.core.models import Procedure, ProcedureStatus, ProcedureStep, StepType
from myelin.memory.procedural import ProceduralMemory


def _iso_days_ago(days: float) -> str:
    # Naive UTC ISO, matching the runtime's datetime.utcnow().isoformat()
    return (datetime.utcnow() - timedelta(days=days)).isoformat()


def _iso_hours_ago(hours: float) -> str:
    return (datetime.utcnow() - timedelta(hours=hours)).isoformat()


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


def _mark_decayer_due(db: Database, decayer: Decayer) -> None:
    db.execute(
        "INSERT INTO process_runs (process_name, status, completed_at) VALUES (?, ?, ?)",
        (decayer.name.value, "completed", _iso_days_ago(2)),
    )
    db.commit()


@pytest.mark.asyncio
async def test_single_success_candidate_never_archives(tmp_db):
    """A procedure with 1 success decays to the floor but stays ACTIVE."""
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "single_success",
        last_executed=_iso_days_ago(30),
        success_count=1,
        failure_count=0,
        confidence=0.6,
    )

    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    assert result["archived"] == 0

    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?", ("single_success",)
    )
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] == pytest.approx(UNVALIDATED_FLOOR)


@pytest.mark.asyncio
async def test_validated_procedure_can_archive_after_long_idle(tmp_db):
    """A trusted procedure may archive after long disuse."""
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "trusted_idle",
        last_executed=_iso_days_ago(90),
        success_count=5,
        failure_count=0,
        confidence=0.9,
    )

    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    assert result["archived"] == 1

    row = db.fetchone("SELECT status, confidence FROM procedures WHERE name = ?", ("trusted_idle",))
    assert row["status"] == ProcedureStatus.ARCHIVED.value
    assert row["confidence"] < ARCHIVE_THRESHOLD


@pytest.mark.asyncio
async def test_grace_period_skips_recently_executed(tmp_db):
    """Within the grace window a procedure is selected but deliberately skipped."""
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "fresh_exec",
        last_executed=_iso_days_ago(0.1),
        success_count=1,
        failure_count=0,
        confidence=0.8,
    )

    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    assert result["processed"] == 1
    assert result["modified"] == 0

    row = db.fetchone("SELECT status, confidence FROM procedures WHERE name = ?", ("fresh_exec",))
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_candidate_decays_but_floors(tmp_db):
    """A long-idle candidate decays but is clamped at the unvalidated floor."""
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "crater_risk",
        last_executed=_iso_days_ago(60),
        success_count=1,
        failure_count=0,
        confidence=0.8,
    )

    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    row = db.fetchone("SELECT status, confidence FROM procedures WHERE name = ?", ("crater_risk",))
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] == pytest.approx(UNVALIDATED_FLOOR)
    assert result["archived"] == 0


@pytest.mark.asyncio
async def test_two_successes_never_archive_even_without_failures(tmp_db):
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "two_successes",
        last_executed=_iso_days_ago(90),
        success_count=2,
        failure_count=0,
        confidence=0.9,
    )
    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?", ("two_successes",)
    )
    assert result["processed"] == 1
    assert result["archived"] == 0
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] == pytest.approx(UNVALIDATED_FLOOR)


@pytest.mark.asyncio
async def test_two_successes_and_one_failure_are_still_unvalidated(tmp_db):
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "mixed_outcomes",
        last_executed=_iso_days_ago(90),
        success_count=2,
        failure_count=1,
        confidence=0.9,
    )
    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?", ("mixed_outcomes",)
    )
    assert result["archived"] == 0
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] == pytest.approx(UNVALIDATED_FLOOR)


@pytest.mark.asyncio
async def test_three_successes_are_exactly_the_validation_boundary(tmp_db):
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "boundary_trusted",
        last_executed=_iso_days_ago(90),
        success_count=3,
        failure_count=0,
        confidence=0.9,
    )
    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?", ("boundary_trusted",)
    )
    assert result["archived"] == 1
    assert row["status"] == ProcedureStatus.ARCHIVED.value
    assert row["confidence"] < ARCHIVE_THRESHOLD


@pytest.mark.asyncio
async def test_just_inside_grace_period_is_not_modified(tmp_db):
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "inside_grace",
        last_executed=_iso_hours_ago(GRACE_HOURS - 1),
        success_count=5,
        failure_count=0,
        confidence=0.8,
    )
    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    row = db.fetchone("SELECT confidence FROM procedures WHERE name = ?", ("inside_grace",))
    assert result["processed"] == 1
    assert result["modified"] == 0
    assert row["confidence"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_just_outside_grace_period_decays(tmp_db):
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "outside_grace",
        last_executed=_iso_hours_ago(GRACE_HOURS + 1),
        success_count=5,
        failure_count=0,
        confidence=0.8,
    )
    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    row = db.fetchone("SELECT confidence FROM procedures WHERE name = ?", ("outside_grace",))
    assert result["processed"] == 1
    assert result["modified"] == 1
    assert row["confidence"] < 0.8


@pytest.mark.asyncio
async def test_validated_procedure_within_grace_is_untouched(tmp_db):
    db, procedural = tmp_db, ProceduralMemory(tmp_db)
    _store_proc(
        db,
        procedural,
        "trusted_fresh",
        last_executed=_iso_hours_ago(GRACE_HOURS - 1),
        success_count=5,
        failure_count=0,
        confidence=0.8,
    )
    decayer = Decayer(db)
    _mark_decayer_due(db, decayer)
    result = await decayer.run()
    row = db.fetchone(
        "SELECT status, confidence FROM procedures WHERE name = ?", ("trusted_fresh",)
    )
    assert result["modified"] == 0
    assert row["status"] == ProcedureStatus.ACTIVE.value
    assert row["confidence"] == pytest.approx(0.8)
