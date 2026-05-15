"""Tests for FSRS-5 spaced repetition scheduler."""

import math
from myelin.cognitive.fsrs_scheduler import (
    DEFAULT_W,
    FSRSScheduler,
    forgetting_curve,
    initial_stability,
    initial_difficulty,
    stability_after_review,
    stability_after_fail,
    difficulty_after_review,
    myelin_signals_to_grade,
    optimal_review_interval,
    review_priority,
    hybrid_activation,
)


# ── Forgetting Curve ────────────────────────────────────────────────


def test_forgetting_curve_zero_elapsed():
    """At t=0, retrievability should be 1.0."""
    assert forgetting_curve(1.0, 0.0) == 1.0


def test_forgetting_curve_at_stability():
    """At t=stability, retrievability should be exactly 0.9."""
    r = forgetting_curve(10.0, 10.0)
    assert abs(r - 0.9) < 0.001


def test_forgetting_curve_after_stability():
    """Beyond stability, retrievability drops below 0.9."""
    r = forgetting_curve(10.0, 20.0)
    assert r < 0.9


def test_forgetting_curve_decays():
    """Retrievability should decrease monotonically."""
    r1 = forgetting_curve(30.0, 1.0)
    r2 = forgetting_curve(30.0, 10.0)
    r3 = forgetting_curve(30.0, 30.0)
    assert r1 > r2 > r3


def test_forgetting_curve_negative_stability():
    """Negative or zero stability returns 0."""
    assert forgetting_curve(0, 1.0) == 0.0
    assert forgetting_curve(-1, 1.0) == 0.0


# ── Initial Values ──────────────────────────────────────────────────


def test_initial_stability_higher_for_better_grade():
    """Higher grade -> higher initial stability."""
    s4 = initial_stability(4)
    s2 = initial_stability(2)
    s1 = initial_stability(1)
    assert s4 > s2 > s1


def test_initial_stability_positive():
    """Initial stability should be positive."""
    for g in range(1, 5):
        assert initial_stability(g) > 0


def test_initial_difficulty_clamped():
    """Initial difficulty should be in [1, 10]."""
    for g in range(1, 5):
        d = initial_difficulty(g)
        assert 1 <= d <= 10


def test_initial_difficulty_grade_4():
    """Grade 4 (complete success) should give lower difficulty."""
    d4 = initial_difficulty(4)
    d1 = initial_difficulty(1)
    assert d4 < d1


# ── Stability Updates ───────────────────────────────────────────────


def test_stability_after_review_increases_with_good_grade():
    """Grade 4 should increase stability."""
    s = stability_after_review(1.0, 5.0, 0.9, 4)
    assert s > 1.0


def test_stability_after_review_grade_2_is_lower_than_grade_4():
    """Grade 2 should give lower stability boost than grade 4."""
    s4 = stability_after_review(10.0, 5.0, 0.7, 4)
    s2 = stability_after_review(10.0, 5.0, 0.7, 2)
    assert s4 > s2


def test_stability_after_fail_decreases():
    """Failed review lowers stability via fail formula."""
    s = stability_after_fail(10.0, 5.0, 0.8, 2)
    assert s < 10.0


# ── Difficulty Updates ──────────────────────────────────────────────


def test_difficulty_decreases_with_good_grade():
    """Grade 4 should decrease difficulty (memory becomes easier)."""
    d4 = difficulty_after_review(7.0, 4)
    assert d4 < 7.0

    # Grade 3 should not change difficulty
    d3 = difficulty_after_review(7.0, 3)
    assert d3 == 7.0


def test_difficulty_increases_with_bad_grade():
    """Grade 1 should increase difficulty."""
    d = difficulty_after_review(3.0, 1)
    assert d > 3.0


def test_difficulty_stays_clamped():
    """Difficulty should stay in [1, 10]."""
    for g in range(1, 5):
        d = difficulty_after_review(7.5, g)
        assert 1 <= d <= 10
    d_low = difficulty_after_review(0.5, 4)
    assert d_low >= 1.0
    d_high = difficulty_after_review(12.0, 1)
    assert d_high <= 10.0


# ── Grade Mapping ───────────────────────────────────────────────────


def test_grade_complete_failure():
    """Complete failure with high surprise -> grade 1."""
    assert myelin_signals_to_grade(False, 0.8, 0.9, 0.3) == 1


def test_grade_partial_failure():
    """Failure with low surprise -> grade 2."""
    assert myelin_signals_to_grade(False, 0.2, 0.1, 0.3) == 2


def test_grade_partial_success():
    """Success with high prediction error -> grade 3."""
    assert myelin_signals_to_grade(True, 0.8, 0.5, 0.5) == 3


def test_grade_complete_success():
    """Success with high confidence -> grade 4."""
    assert myelin_signals_to_grade(True, 0.1, 0.1, 0.9) == 4


def test_grade_default_without_signals():
    """No signals -> default to success."""
    grade = myelin_signals_to_grade(None, None, None, None)
    assert grade in (3, 4)


# ── Optimal Interval ────────────────────────────────────────────────


def test_optimal_interval_increases_with_stability():
    """Higher stability -> longer optimal interval."""
    i1 = optimal_review_interval(1.0)
    i10 = optimal_review_interval(10.0)
    assert i10 > i1


def test_optimal_interval_zero_stability():
    """Zero stability returns 0."""
    assert optimal_review_interval(0.0) == 0.0


def test_optimal_interval_positive():
    """Positive stability should yield positive interval."""
    assert optimal_review_interval(5.0) > 0


# ── Review Priority ─────────────────────────────────────────────────


def test_review_priority_lower_retrievability_higher():
    """Lower retrievability = higher priority."""
    p1 = review_priority(0.3, 5.0, 0.5, 10, 7)
    p2 = review_priority(0.9, 5.0, 0.5, 10, 7)
    assert p1 > p2


def test_review_priority_overdue_higher():
    """More overdue = higher priority."""
    p1 = review_priority(0.5, 5.0, 0.5, 100, 7)
    p2 = review_priority(0.5, 5.0, 0.5, 1, 7)
    assert p1 > p2


def test_review_priority_important_higher():
    """Higher importance = higher priority."""
    p1 = review_priority(0.5, 5.0, 0.9, 5, 7)
    p2 = review_priority(0.5, 5.0, 0.2, 5, 7)
    assert p1 > p2


# ── Hybrid Activation ───────────────────────────────────────────────


def test_hybrid_activation_default_weights():
    """Default weights blend ACT-R and FSRS."""
    h = hybrid_activation(-1.0, 0.8, 10.0)
    # ACT-R: (-1 + 3) / 6 = 0.333
    # FSRS: 0.8
    # Blend: 0.3 * 0.333 + 0.7 * 0.8 = 0.66
    assert abs(h - 0.66) < 0.05


def test_hybrid_activation_full_actr():
    """All weight on ACT-R."""
    h = hybrid_activation(0.0, 0.5, 10.0, actr_weight=1.0, fsrs_weight=0.0)
    assert abs(h - 0.5) < 0.05


def test_hybrid_activation_clamped():
    """ACT-R activation clamps to [0,1]."""
    h = hybrid_activation(100.0, 0.5, 10.0)
    assert h <= 1.0


# ── FSRSScheduler Integration ───────────────────────────────────────


def test_scheduler_init_memory():
    """init_memory returns valid FSRS state."""
    sched = FSRSScheduler()
    state = sched.init_memory(grade=4)
    assert state["stability"] > 0
    assert 1 <= state["difficulty"] <= 10
    assert state["optimal_interval_days"] > 0
    assert state["retrievability"] == 1.0


def test_scheduler_init_memory_low_grade():
    """Low grade = lower stability, higher difficulty."""
    sched = FSRSScheduler()
    g4 = sched.init_memory(grade=4)
    g1 = sched.init_memory(grade=1)
    assert g4["stability"] > g1["stability"]
    assert g4["difficulty"] < g1["difficulty"]


def test_scheduler_record_review_success():
    """Record a successful review (grade 4 decreases difficulty)."""
    sched = FSRSScheduler()
    result = sched.record_review(stability=5.0, difficulty=5.0, retrievability=0.8, grade=4)
    assert result["new_stability"] > 5.0
    assert result["optimal_interval_days"] > 0
    # Grade 4 should decrease difficulty
    assert result["new_difficulty"] < 5.0


def test_scheduler_record_review_failure():
    """Record a failed review (grade 1-2) uses fail formula."""
    sched = FSRSScheduler()
    result = sched.record_review(stability=10.0, difficulty=5.0, retrievability=0.7, grade=1)
    # Fail formula reduces stability
    assert result["new_stability"] < 10.0
    assert result["new_difficulty"] > 5.0


def test_scheduler_schedule_next_review():
    """schedule_next_review returns expected keys."""
    sched = FSRSScheduler()
    result = sched.schedule_next_review(
        current_stability=5.0,
        current_difficulty=5.0,
        current_retrievability=0.85,
        grade=3,
    )
    assert "new_stability" in result
    assert "new_difficulty" in result
    assert "optimal_interval_days" in result
    assert "next_retrievability" in result


def test_scheduler_get_retrievability():
    """get_retrievability returns expected value."""
    sched = FSRSScheduler()
    r = sched.get_retrievability(stability_days=1.0, last_reviewed=0)
    # 0 timestamp = 1970, many days ago => retrievability ~ 0
    assert r < 0.1


def test_scheduler_get_retrievability_no_review():
    """No last_reviewed returns 1.0."""
    sched = FSRSScheduler()
    r = sched.get_retrievability(stability_days=1.0, last_reviewed=None)
    assert r == 1.0


def test_scheduler_status():
    """get_status returns diagnostic info."""
    sched = FSRSScheduler()
    status = sched.get_status()
    assert "parameters" in status
    assert len(status["parameters"]) == 19
    assert "min_difficulty" in status
    assert "max_difficulty" in status


def test_scheduler_custom_weights():
    """Custom weights are used correctly."""
    custom_w = [0.5, 1.5, 4.0, 20.0, 8.0, 0.6, 1.5, 0.005, 1.6, 0.12, 1.1, 2.0, 0.12, 0.3, 2.3, 0.25, 3.0, 0.55, 0.7]
    sched = FSRSScheduler(w=custom_w)
    state = sched.init_memory(grade=3)
    assert state["stability"] > 0
    assert sched.w == custom_w
