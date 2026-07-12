"""Test procedural memory operations."""

from myelin.core.models import (
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    PromotionMethod,
    StepType,
)


def _make_procedure(name="test_deploy", domain="deployment", source_agent="agent-1", **kwargs):
    defaults = dict(
        preconditions=["git repo exists", "node installed"],
        postconditions=["build artifacts available", "tests passed"],
    )
    defaults.update(kwargs)
    return Procedure(
        name=name,
        trigger_pattern=f"When user wants to {name}",
        steps=[
            ProcedureStep(order=0, description="git pull", step_type=StepType.CORE),
            ProcedureStep(order=1, description="npm test", step_type=StepType.CORE),
            ProcedureStep(order=2, description="npm run build", step_type=StepType.CORE),
            ProcedureStep(order=3, description="notify slack", step_type=StepType.OPTIONAL),
        ],
        source_agent=source_agent,
        domain=domain,
        **defaults,
    )


def test_store_and_retrieve(procedural):
    proc = _make_procedure()
    proc_id = procedural.store(proc)

    retrieved = procedural.get(proc_id)
    assert retrieved is not None
    assert retrieved["name"] == "test_deploy"
    assert len(retrieved["steps"]) == 4
    assert retrieved["confidence"] == 0.5


def test_bayesian_confidence_on_success(procedural):
    proc = _make_procedure(status=ProcedureStatus.ACTIVE)
    procedural.store(proc)

    new_conf = procedural.record_execution(proc.id, success=True)
    assert new_conf > 0.5

    retrieved = procedural.get(proc.id)
    assert retrieved["success_count"] == 1
    assert retrieved["failure_count"] == 0
    assert retrieved["execution_count"] == 1


def test_bayesian_confidence_on_failure(procedural):
    proc = _make_procedure(status=ProcedureStatus.ACTIVE)
    procedural.store(proc)

    new_conf = procedural.record_execution(proc.id, success=False)
    assert new_conf < 0.5

    retrieved = procedural.get(proc.id)
    assert retrieved["failure_count"] == 1


def test_confidence_approaches_one_asymptotically(procedural):
    proc = _make_procedure(status=ProcedureStatus.ACTIVE)
    procedural.store(proc)

    conf = 0.5
    for _ in range(50):
        conf = procedural.record_execution(proc.id, success=True)

    assert conf > 0.95
    assert conf <= 1.0


def test_auto_promote_to_active(procedural):
    proc = _make_procedure(status=ProcedureStatus.DRAFT)
    procedural.store(proc)

    for _ in range(20):
        procedural.record_execution(proc.id, success=True)

    retrieved = procedural.get(proc.id)
    assert retrieved["status"] == ProcedureStatus.ACTIVE.value


def test_find_matching_by_text(procedural):
    proc = _make_procedure(name="deploy_app", status=ProcedureStatus.ACTIVE, confidence=0.8)
    procedural.store(proc)

    matches = procedural.find_matching("deploy app")
    assert len(matches) >= 1
    assert matches[0]["name"] == "deploy_app"


def test_count_by_status(procedural):
    for i, status in enumerate(
        [ProcedureStatus.DRAFT, ProcedureStatus.ACTIVE, ProcedureStatus.ACTIVE]
    ):
        proc = _make_procedure(name=f"proc_{i}", status=status)
        procedural.store(proc)

    assert procedural.count() == 3
    assert procedural.count(ProcedureStatus.ACTIVE) == 2
    assert procedural.count(ProcedureStatus.DRAFT) == 1


def test_composite_procedure(procedural):
    proc_a = _make_procedure(
        name="run_tests",
        postconditions=["test_results_available"],
        status=ProcedureStatus.ACTIVE,
    )
    proc_b = _make_procedure(
        name="deploy_if_green",
        preconditions=["test_results_available"],
        status=ProcedureStatus.ACTIVE,
    )
    procedural.store(proc_a)
    procedural.store(proc_b)

    composite_id = procedural.create_composite(
        name="test_and_deploy",
        components=[proc_a.id, proc_b.id],
        trigger_pattern="Run tests then deploy if green",
        source_agent="agent-1",
    )

    composite = procedural.get(composite_id)
    assert composite is not None
    assert composite["is_composite"] == 1
    assert len(composite["component_procedures"]) == 2
    assert len(composite["steps"]) == 8  # 4 from each


def test_get_composable_pairs(procedural):
    proc_a = _make_procedure(
        name="run_tests",
        postconditions=["test_results_available"],
        status=ProcedureStatus.ACTIVE,
    )
    proc_b = _make_procedure(
        name="deploy_if_green",
        preconditions=["test_results_available"],
        status=ProcedureStatus.ACTIVE,
    )
    proc_c = _make_procedure(
        name="unrelated",
        preconditions=["something_else"],
        postconditions=["another_thing"],
        status=ProcedureStatus.ACTIVE,
    )
    procedural.store(proc_a)
    procedural.store(proc_b)
    procedural.store(proc_c)

    pairs = procedural.get_composable_pairs()
    assert len(pairs) >= 1
    pair_names = [(p[0]["name"], p[1]["name"]) for p in pairs]
    assert ("run_tests", "deploy_if_green") in pair_names


def test_archive_procedure(procedural):
    proc = _make_procedure(status=ProcedureStatus.ACTIVE)
    procedural.store(proc)

    procedural.archive(proc.id)
    retrieved = procedural.get(proc.id)
    assert retrieved["status"] == ProcedureStatus.ARCHIVED.value


# ── Trust Lifecycle Tests ──────────────────────────────────────


def test_initial_trust_state_is_seed(procedural):
    """A newly created auto-generated procedure starts at seed trust state."""
    proc = _make_procedure(confidence=0.2, status=ProcedureStatus.DRAFT)
    procedural.store(proc)
    state = procedural.update_trust_state(proc.id)
    assert state == "seed"


def test_candidate_trust_state_from_confidence(procedural):
    """A procedure with confidence >= 0.3 becomes a candidate."""
    proc = _make_procedure(confidence=0.5, status=ProcedureStatus.DRAFT)
    procedural.store(proc)
    state = procedural.update_trust_state(proc.id)
    assert state == "candidate"


def test_candidate_trust_state_from_taught(procedural):
    """A manually taught procedure starts as candidate even at low confidence."""
    proc = _make_procedure(
        confidence=0.2,
        status=ProcedureStatus.ACTIVE,
        promotion_method=PromotionMethod.TAUGHT,
    )
    procedural.store(proc)
    state = procedural.update_trust_state(proc.id)
    assert state == "candidate"


def test_trusted_requires_confidence_and_executions(procedural):
    """Trusted requires confidence >= 0.7 and 3+ successful executions."""
    proc = _make_procedure(
        confidence=0.72,
        status=ProcedureStatus.ACTIVE,
        success_count=3,
        failure_count=0,
    )
    procedural.store(proc)
    state = procedural.update_trust_state(proc.id)
    assert state == "trusted"


def test_trusted_not_without_enough_successes(procedural):
    """High confidence alone without 3+ successes should not be trusted."""
    proc = _make_procedure(
        confidence=0.72,
        status=ProcedureStatus.ACTIVE,
        success_count=1,
        failure_count=0,
    )
    procedural.store(proc)
    state = procedural.update_trust_state(proc.id)
    assert state == "candidate"  # high conf but < 3 successes


def test_validated_requires_transfer(procedural):
    """Validated requires confidence >= 0.85 and cross-agent transfer."""
    proc = _make_procedure(
        confidence=0.88,
        status=ProcedureStatus.ACTIVE,
        success_count=3,
        transferred_to=["agent-2"],
    )
    procedural.store(proc)
    state = procedural.update_trust_state(proc.id)
    assert state == "validated"


def test_record_evidence_stores_procedure_evidence(procedural):
    """record_evidence stores a row and returns an id."""
    proc = _make_procedure(confidence=0.5)
    procedural.store(proc)

    ev_id = procedural.record_evidence(
        procedure_id=proc.id,
        source="feedback",
        outcome="success",
        confidence_delta=0.08,
    )
    assert ev_id is not None
    rows = procedural.get_evidence(proc.id)
    assert len(rows) == 1
    assert rows[0]["source"] == "feedback"
    assert rows[0]["outcome"] == "success"

    # last_evidence_timestamp should have been set
    retrieved = procedural.get(proc.id)
    assert retrieved.get("last_evidence_timestamp") is not None


def test_execution_automatically_records_evidence(procedural):
    """record_execution should automatically create an evidence row."""
    proc = _make_procedure(confidence=0.5, status=ProcedureStatus.ACTIVE)
    procedural.store(proc)

    procedural.record_execution(proc.id, success=True)

    evidence = procedural.get_evidence(proc.id)
    assert len(evidence) >= 1
    assert evidence[0]["source"] == "execution"
    assert evidence[0]["outcome"] == "success"


def test_trust_summary_contains_all_fields(procedural):
    """get_trust_summary returns a complete trust snapshot."""
    proc = _make_procedure(confidence=0.5)
    procedural.store(proc)
    procedural.record_execution(proc.id, success=True)

    summary = procedural.get_trust_summary(proc.id)
    assert summary["trust_state"] in ("seed", "candidate")
    assert "confidence" in summary
    assert "evidence_count" in summary
    assert summary["evidence_count"] >= 1
