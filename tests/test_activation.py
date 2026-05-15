"""Test ACT-R activation math and Bayesian confidence."""

import time

import pytest

from myelin.core.activation import (
    agent_similarity,
    base_level_activation,
    bayesian_confidence_update,
    calibration_offset,
    ebbinghaus_decay,
    should_promote,
    transfer_confidence,
)


class TestBaseActivation:
    def test_empty_returns_neg_inf(self):
        assert base_level_activation([]) == float("-inf")

    def test_single_recent_access(self):
        now = time.time()
        result = base_level_activation([now - 0.5], now=now)
        # ln(0.5^(-0.5)) = ln(sqrt(2)) > 0
        assert result > 0

    def test_recent_higher_than_old(self):
        now = time.time()
        recent = base_level_activation([now - 10], now=now)
        old = base_level_activation([now - 10000], now=now)
        assert recent > old

    def test_frequent_higher_than_rare(self):
        now = time.time()
        frequent = base_level_activation([now - 100, now - 200, now - 300], now=now)
        rare = base_level_activation([now - 100], now=now)
        assert frequent > rare

    def test_recency_and_frequency_combined(self):
        now = time.time()
        recent_frequent = base_level_activation([now - 10, now - 20, now - 30, now - 40], now=now)
        old_frequent = base_level_activation(
            [now - 10000, now - 20000, now - 30000, now - 40000], now=now
        )
        assert recent_frequent > old_frequent


class TestShouldPromote:
    def test_below_min_episodes(self):
        now = time.time()
        assert not should_promote([now], min_episodes=2)

    def test_above_threshold(self):
        now = time.time()
        times = [now - i for i in range(1, 20)]
        assert should_promote(times, threshold=1.0)

    def test_below_threshold(self):
        now = time.time()
        assert not should_promote([now - 100000], threshold=1.0, min_episodes=1)


class TestBayesianConfidence:
    def test_success_increases(self):
        assert bayesian_confidence_update(0.5, True) > 0.5

    def test_failure_decreases(self):
        assert bayesian_confidence_update(0.5, False) < 0.5

    def test_bounded_above(self):
        c = 0.99
        for _ in range(100):
            c = bayesian_confidence_update(c, True)
        assert c <= 1.0

    def test_bounded_below(self):
        c = 0.01
        for _ in range(100):
            c = bayesian_confidence_update(c, False)
        assert c >= 0.0

    def test_asymptotic_approach(self):
        c = 0.5
        prev_delta = 1.0
        for _ in range(10):
            new_c = bayesian_confidence_update(c, True)
            delta = new_c - c
            assert delta < prev_delta
            prev_delta = delta
            c = new_c

    def test_mature_procedures_stable(self):
        high_conf = bayesian_confidence_update(0.95, False)
        low_conf = bayesian_confidence_update(0.3, False)
        high_drop = 0.95 - high_conf
        low_drop = 0.3 - low_conf
        assert high_drop > low_drop


class TestCalibration:
    def test_overconfident(self):
        predicted = [0.9, 0.9, 0.9]
        outcomes = [True, False, False]
        offset = calibration_offset(predicted, outcomes)
        assert offset > 0

    def test_underconfident(self):
        predicted = [0.3, 0.3, 0.3]
        outcomes = [True, True, True]
        offset = calibration_offset(predicted, outcomes)
        assert offset < 0

    def test_well_calibrated(self):
        predicted = [0.5, 0.5]
        outcomes = [True, False]
        offset = calibration_offset(predicted, outcomes)
        assert abs(offset) < 0.01


class TestEbbinghausDecay:
    def test_no_decay_at_zero_hours(self):
        assert ebbinghaus_decay(0.8, 0.0) == 0.8

    def test_decays_over_time(self):
        assert ebbinghaus_decay(0.8, 48.0) < 0.8

    def test_high_stability_decays_slower(self):
        fast = ebbinghaus_decay(0.8, 24.0, stability=1.0)
        slow = ebbinghaus_decay(0.8, 24.0, stability=100.0)
        assert slow > fast


class TestAgentSimilarity:
    def test_identical_agents(self):
        tools = {"bash", "file_edit", "web_search"}
        score = agent_similarity(tools, tools, "mcp_stdio", "mcp_stdio", "claude", "claude")
        assert score == 1.0

    def test_completely_different(self):
        score = agent_similarity(
            {"bash"},
            {"web_browse"},
            "mcp_stdio",
            "custom",
            "claude",
            "gpt",
        )
        assert score < 0.5

    def test_partial_overlap(self):
        score = agent_similarity(
            {"bash", "file_edit", "web_search"},
            {"bash", "file_edit", "code_review"},
            "mcp_stdio",
            "mcp_sse",
            "claude",
            "claude",
        )
        assert 0.5 < score < 1.0


class TestTransferConfidence:
    def test_perfect_transfer(self):
        assert transfer_confidence(0.9, 1.0) == 0.9

    def test_discounted_transfer(self):
        result = transfer_confidence(0.9, 0.5)
        assert result == pytest.approx(0.45)

    def test_zero_similarity(self):
        assert transfer_confidence(0.9, 0.0) == 0.0
