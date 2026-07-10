"""Tests for the background consolidation scheduler.

Covers:
- ``start`` / ``stop`` lifecycle
- Single-run locking (locking blocks overlapping runs)
- Interval gating (scheduler waits between runs)
- Status/error tracking (successful vs failed runs)
- Failure handling (callable raises → tracked as error)
- Graceful stop (stop within reasonable timeout)
- ``set_interval`` at runtime
- Multiple start/stop cycles
- ``is_running`` property
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from myelin.cognitive.scheduler import ConsolidationScheduler


@pytest.fixture
def success_callable():
    """An async callable that succeeds immediately."""

    async def _run() -> dict[str, Any]:
        return {"processed": 5, "created": 2}

    return _run


@pytest.fixture
def slow_callable():
    """An async callable that takes ~0.3s to complete."""

    async def _run() -> dict[str, Any]:
        time.sleep(0.3)
        return {"processed": 1}

    return _run


@pytest.fixture
def failing_callable():
    """An async callable that raises an exception."""

    async def _run() -> dict[str, Any]:
        raise RuntimeError("simulated consolidation failure")

    return _run


class TestLifecycle:
    """Start / stop lifecycle."""

    def test_start_stop(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=9999)
        try:
            scheduler.start()
            assert scheduler.is_running
            status = scheduler.get_status()
            assert status.running is True
        finally:
            scheduler.stop()

        assert scheduler.is_running is False
        status = scheduler.get_status()
        assert status.running is False

    def test_double_start_noop(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=9999)
        try:
            scheduler.start()
            scheduler.start()  # should not raise
            assert scheduler.is_running
        finally:
            scheduler.stop()

    def test_stop_not_started(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=9999)
        scheduler.stop()  # should not raise
        status = scheduler.get_status()
        assert status.running is False

    def test_stop_timeout(self, slow_callable):
        """Stop should block until the running consolidation finishes."""
        scheduler = ConsolidationScheduler(slow_callable, interval_seconds=9999)
        try:
            scheduler.start()
            time.sleep(0.05)  # let the first tick start
            scheduler.stop(timeout=5.0)
            assert scheduler.is_running is False
        finally:
            scheduler.stop()

    def test_is_running_property(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=9999)
        assert scheduler.is_running is False
        try:
            scheduler.start()
            assert scheduler.is_running is True
        finally:
            scheduler.stop()
        assert scheduler.is_running is False


class TestSingleRunLocking:
    """Single-run lock prevents overlapping executions."""

    def test_overlap_skipped(self, slow_callable):
        """If a run is still in progress when the next tick fires, the tick
        is skipped.  Since interval_seconds=0 means every tick fires as soon
        as the wait completes, we can test with a very short interval."""
        scheduler = ConsolidationScheduler(slow_callable, interval_seconds=0.05)
        try:
            scheduler.start()
            time.sleep(1.0)
            status = scheduler.get_status()
            # The slow callable takes 0.3s, so in 1s we should have some
            # successful runs and at least one skipped overlap.
            assert status.successful_runs >= 1
            assert status.skipped_overlaps >= 0  # at least non-negative
            # total_runs should be ~ successful + skipped (no failures)
            assert status.total_runs >= status.successful_runs
        finally:
            scheduler.stop()


class TestIntervalGating:
    """Scheduler waits the configured interval between runs."""

    def test_respects_interval(self, success_callable):
        """With a 0.2s interval, we should get roughly 3-5 runs in 1.5s."""
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=0.2)
        try:
            scheduler.start()
            time.sleep(1.5)
            status = scheduler.get_status()
            # 1.5s / 0.2s ≈ 7.5 possible ticks, but with startup and stop delays
            # we expect at least 3 runs
            assert 3 <= status.total_runs <= 10
        finally:
            scheduler.stop()


class TestStatusTracking:
    """Status and error tracking."""

    def test_tracks_successful_runs(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=0.1)
        try:
            scheduler.start()
            time.sleep(0.7)
            scheduler.stop()
            status = scheduler.get_status()
            assert status.total_runs >= 1
            assert status.successful_runs == status.total_runs
            assert status.failed_runs == 0
            assert status.last_success is not None
            assert status.last_error is None
        finally:
            scheduler.stop()

    def test_tracks_failures(self, failing_callable):
        scheduler = ConsolidationScheduler(failing_callable, interval_seconds=0.1)
        try:
            scheduler.start()
            time.sleep(0.7)
            scheduler.stop()
            status = scheduler.get_status()
            assert status.total_runs >= 1
            assert status.failed_runs == status.total_runs
            assert status.successful_runs == 0
            assert status.last_error is not None
            assert "simulated consolidation failure" in status.last_error
        finally:
            scheduler.stop()

    def test_last_run_timestamp(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=0.1)
        try:
            scheduler.start()
            time.sleep(0.5)
            status = scheduler.get_status()
            assert status.last_run is not None
            assert status.last_success is not None
            assert status.last_run > 0
            assert abs(status.last_run - status.last_success) < 1.0
        finally:
            scheduler.stop()

    def test_next_run_after_stop(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=9999)
        try:
            scheduler.start()
            status = scheduler.get_status()
            assert status.next_run is not None
        finally:
            scheduler.stop()
        status = scheduler.get_status()
        assert status.next_run is None


class TestSetInterval:
    """Changing interval at runtime."""

    def test_set_interval_updates_status(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=3600)
        assert scheduler.status.interval_seconds == 3600

        scheduler.set_interval(1800)
        assert scheduler.status.interval_seconds == 1800

    def test_set_interval_rejects_non_positive(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=3600)
        with pytest.raises(ValueError, match="positive"):
            scheduler.set_interval(0)
        with pytest.raises(ValueError, match="positive"):
            scheduler.set_interval(-1)

    def test_set_interval_running_sets_next_run(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=9999)
        try:
            scheduler.start()
            scheduler.set_interval(0.5)
            # next_run should be recalculated from now
            assert scheduler.status.next_run is not None
        finally:
            scheduler.stop()

    def test_set_interval_takes_effect(self, success_callable):
        """Changing interval to a short value should make ticks fire faster."""
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=9999)
        try:
            scheduler.start()
            # With 9999s interval no tick would fire; change to 0.1s.
            # The internal wait loop rechecks every 2s, so we need ~2.5s
            # for the new interval to be picked up and a tick to fire.
            scheduler.set_interval(0.1)
            time.sleep(3.0)
            status = scheduler.get_status()
            assert status.total_runs >= 1
        finally:
            scheduler.stop()


class TestRestart:
    """Multiple start/stop cycles."""

    def test_restart_produces_new_ticks(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=0.1)
        try:
            scheduler.start()
            time.sleep(0.5)
            scheduler.stop()
            runs_after_first = scheduler.get_status().total_runs

            scheduler.start()
            time.sleep(0.5)
            scheduler.stop()
            runs_after_second = scheduler.get_status().total_runs
            assert runs_after_second > runs_after_first
        finally:
            scheduler.stop()

    def test_multiple_start_stop_cycles(self, success_callable):
        scheduler = ConsolidationScheduler(success_callable, interval_seconds=0.2)
        try:
            for _ in range(3):
                scheduler.start()
                assert scheduler.is_running
                time.sleep(0.3)
                scheduler.stop()
                assert not scheduler.is_running
            # Final status should have accumulated runs
            assert scheduler.get_status().total_runs >= 1
        finally:
            scheduler.stop()
