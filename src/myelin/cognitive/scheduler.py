"""Opt-in background consolidation scheduler.

Provides a lightweight ``ConsolidationScheduler`` that runs a user-provided
consolidation callable (typically ``SleepCycle.execute``) on a background thread
at a configurable interval.

Design goals
------------
- **Dependency-light**: pure Python standard library — only ``threading``,
  ``dataclasses``, ``time``, and ``logging``.
- **Opt-in**: the scheduler is NOT started by default.  Users opt in by calling
  ``scheduler.start()``.  This keeps the cognitive process layer pure async and
  free of background thread assumptions.
- **Single-run locking**: uses ``threading.Lock`` to prevent overlapping
  executions (if the callable takes longer than the interval, the next tick is
  skipped rather than queued).
- **Graceful stop**: ``.stop()`` signals a ``threading.Event`` and joins the
  thread (with timeout).
- **Status/error tracking**: the scheduler tracks total runs, failures, last run
  time, next run time, and the error message from the most recent failure.
- **No DB schema changes**: status is kept in-memory as a dataclass; the
  scheduler does not write to or delete any database table.

Usage::

    import asyncio

    from myelin.cognitive.scheduler import ConsolidationScheduler

    async def consolidate():
        await sleep_cycle.execute()

    scheduler = ConsolidationScheduler(callable=consolidate, interval_seconds=3600)
    scheduler.start()

    # ... later ...
    scheduler.stop()

    # Inspect status
    status = scheduler.get_status()
    print(status.total_runs, status.last_run, status.errors)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

log = logging.getLogger("myelin.scheduler")


@dataclass
class SchedulerStatus:
    """In-memory status of the consolidation scheduler.

    All fields are updated atomically and do not touch the database.
    """

    running: bool = False
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    last_run: float | None = None  # time.time() of the last completed run
    last_success: float | None = None  # time.time() of the last successful run
    last_error: str | None = None  # error message from the most recent failure
    next_run: float | None = None  # time.time() of the next scheduled run
    skipped_overlaps: int = 0  # ticks skipped because previous run still active
    interval_seconds: float = 3600.0


class ConsolidationScheduler:
    """Background scheduler that periodically runs a consolidation callable.

    The callable is an async function (typically ``SleepCycle.execute`` or a
    wrapper around ``CognitiveOrchestrator.check_triggers``).
    """

    def __init__(
        self,
        callable: Callable[[], Coroutine[Any, Any, dict[str, Any]]],
        interval_seconds: float = 3600.0,
        name: str = "consolidation",
    ):
        self._callable = callable
        self._interval = interval_seconds
        self._name = name
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._run_lock = Lock()
        self._status_lock = Lock()

        self.status = SchedulerStatus(interval_seconds=interval_seconds)

        log.info(
            "ConsolidationScheduler '%s' initialized (interval=%.0fs)",
            name,
            interval_seconds,
        )

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler thread.

        Safe to call multiple times — subsequent calls are no-ops if the
        scheduler is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            log.warning("Scheduler '%s' is already running", self._name)
            return

        self._stop_event.clear()
        self._thread = Thread(
            target=self._loop,
            name=f"myelin-scheduler-{self._name}",
            daemon=True,
        )
        self._thread.start()

        with self._status_lock:
            self.status.running = True
            self.status.next_run = time.time() + self._interval

        log.info("ConsolidationScheduler '%s' started (thread=%s)", self._name, self._thread.name)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the scheduler to stop and wait for the thread.

        Args:
            timeout: Maximum seconds to wait for the thread to finish. A
                     currently-running consolidation will be allowed to
                     complete up to this timeout.
        """
        if self._thread is None or not self._thread.is_alive():
            with self._status_lock:
                self.status.running = False
            return

        self._stop_event.set()
        self._thread.join(timeout=timeout)

        with self._status_lock:
            self.status.running = False
            self.status.next_run = None

        if self._thread.is_alive():
            log.warning(
                "Scheduler '%s' thread did not stop within %.0fs timeout",
                self._name,
                timeout,
            )
        else:
            log.info("ConsolidationScheduler '%s' stopped cleanly", self._name)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Status ───────────────────────────────────────────────────

    def get_status(self) -> SchedulerStatus:
        """Return a snapshot of the current scheduler status."""
        with self._status_lock:
            return SchedulerStatus(
                running=self.status.running,
                total_runs=self.status.total_runs,
                successful_runs=self.status.successful_runs,
                failed_runs=self.status.failed_runs,
                last_run=self.status.last_run,
                last_success=self.status.last_success,
                last_error=self.status.last_error,
                next_run=self.status.next_run,
                skipped_overlaps=self.status.skipped_overlaps,
                interval_seconds=self.status.interval_seconds,
            )

    def set_interval(self, interval_seconds: float) -> None:
        """Change the interval between runs. Takes effect after the current run."""
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval = interval_seconds
        with self._status_lock:
            self.status.interval_seconds = interval_seconds
            # Recalculate next_run from now + new interval
            if self.status.running:
                self.status.next_run = time.time() + interval_seconds
        log.info("Scheduler '%s' interval updated to %.0fs", self._name, interval_seconds)

    # ── Internal loop ────────────────────────────────────────────

    def _loop(self) -> None:
        """Main scheduler loop running on the background thread."""
        log.debug("Scheduler '%s' loop started", self._name)

        while not self._stop_event.is_set():
            # Wait for the interval (checking stop_event every 2s for
            # responsive shutdown during long intervals).
            # Re-read self._interval each iteration in case set_interval
            # was called while we were sleeping.
            remaining = self._interval
            while remaining > 0 and not self._stop_event.is_set():
                wait = min(remaining, 2.0)
                self._stop_event.wait(wait)
                remaining -= wait
                # If interval was changed mid-sleep, recalculate remaining
                if self._interval != remaining + wait:
                    remaining = self._interval

            if self._stop_event.is_set():
                break

            self._tick()

        log.debug("Scheduler '%s' loop ended", self._name)

    def _tick(self) -> None:
        """Execute one consolidation run if the lock is available."""
        # Single-run locking — skip if previous run still in progress
        if not self._run_lock.acquire(blocking=False):
            log.warning(
                "Scheduler '%s': previous run still active — skipping tick",
                self._name,
            )
            with self._status_lock:
                self.status.skipped_overlaps += 1
                self.status.next_run = time.time() + self._interval
            return

        try:
            log.debug("Scheduler '%s': running consolidation tick", self._name)

            with self._status_lock:
                self.status.total_runs += 1

            # Run the async callable from the sync thread via a new event loop
            _ = self._run_async()

            with self._status_lock:
                self.status.successful_runs += 1
                self.status.last_run = time.time()
                self.status.last_success = time.time()
                self.status.last_error = None
                self.status.next_run = time.time() + self._interval

            log.info(
                "Scheduler '%s': consolidation completed (run %d)",
                self._name,
                self.status.total_runs,
            )
        except Exception as e:
            log.error(
                "Scheduler '%s': consolidation failed: %s",
                self._name,
                e,
                exc_info=True,
            )
            with self._status_lock:
                self.status.failed_runs += 1
                self.status.last_run = time.time()
                self.status.last_error = str(e)
                self.status.next_run = time.time() + self._interval
        finally:
            self._run_lock.release()

    def _run_async(self) -> dict[str, Any]:
        """Run the async callable from the synchronous background thread.

        Creates a temporary event loop if one doesn't exist on this thread.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're on the main event loop thread — schedule and wait
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._callable())
                return future.result()

        # We're on a non-event-loop thread — create a temporary loop
        return asyncio.run(self._callable())
