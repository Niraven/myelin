"""FSRS-5 scheduler for optimal memory review timing.

Free Spaced Repetition Scheduler v5 — the DSR (Difficulty, Stability,
Retrievability) model from Anki's modern scheduler.

Integrates with Myelin's ACT-R activation and Ebbinghaus decay to
determine the optimal time to review each memory for maximum retention.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Any

log = logging.getLogger("myelin.fsrs")

# ── FSRS-5 Default Parameters ───────────────────────────────────────
# From py-fsrs v5 reference implementation
DEFAULT_W: list[float] = [
    0.40255,
    1.18385,
    3.173,
    15.69105,
    1.0,  # w0-w4 (w4=1 for gentler diff)
    0.3,  # w5
    1.4604,
    0.0046,
    1.54575,
    0.1192,  # w6-w9
    0.3,
    0.2,
    0.15,
    0.29605,
    2.2698,  # w10-w14 (fail params)
    0.2315,
    2.9898,
    0.51655,
    0.6621,  # w15-w18
]

MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 10.0
DEFAULT_DIFFICULTY = 5.0
DEFAULT_STABILITY_DAYS = 1.0
MAX_STABILITY_DAYS = 365.0
GRADE_MAP_THRESHOLD = 0.5  # min success rate for grade 3+


# ── Pure Functions (testable) ────────────────────────────────────────


def forgetting_curve(stability_days: float, elapsed_days: float) -> float:
    """FSRS forgetting curve: R = exp(ln(0.9) * elapsed / stability).

    Returns retrievability R ∈ (0, 1] where 0.9 = 90% recall probability
    at exactly `stability` days.
    """
    if stability_days <= 0 or elapsed_days < 0:
        return 0.0
    return math.exp(math.log(0.9) * elapsed_days / stability_days)


def initial_stability(grade: int, w: list[float] | None = None) -> float:
    """Initial stability after first review.

    S_0 = w[0] + w[1] * (1 - grade/4)
    Higher grade = higher initial stability.
    """
    if w is None:
        w = DEFAULT_W
    g = _clamp_grade(grade)
    return max(0.1, w[0] + w[1] * (g / 4.0))


def initial_difficulty(grade: int, w: list[float] | None = None) -> float:
    """Initial difficulty after first review.

    D_0 = w[2] + w[3] * (1 - grade/4)
    Clamped to [1, 10].
    """
    if w is None:
        w = DEFAULT_W
    g = _clamp_grade(grade)
    return _clamp_difficulty(w[2] + w[3] * (1.0 - g / 4.0))


def stability_after_review(
    stability: float,
    difficulty: float,
    retrievability: float,
    grade: int,
    w: list[float] | None = None,
) -> float:
    """Compute new stability after a successful review (grade 3-4).

    Simplified FSRS-5: higher grade → higher stability multiplier.
    Grade 3 (good): slight increase. Grade 4 (easy): strong increase.
    """
    if w is None:
        w = DEFAULT_W
    g = _clamp_grade(grade)
    d = _clamp_difficulty(difficulty)
    r = max(0.01, min(0.99, retrievability))

    # Grade boost: g=3 → 1.0, g=4 → ~2.5x
    grade_mult = 1.0 + (w[6] + w[7]) * (g - 3.0)

    # Difficulty modulation: harder memories get less boost
    diff_mod = 1.0 + w[8] * (1.0 - d / 10.0)

    # Retrievability modulation: well-retrieved memories get less boost
    ret_mod = 1.0 + w[9] * (1.0 - r)

    factor = grade_mult * diff_mod * ret_mod
    new_s = stability * max(0.1, factor)
    return min(MAX_STABILITY_DAYS, new_s)


def stability_after_fail(
    stability: float,
    difficulty: float,
    retrievability: float,
    grade: int,
    w: list[float] | None = None,
) -> float:
    """Compute new stability after a failed review (grade 1-2).

    Simplified FSRS-5: resets to a fraction of current stability.
    Failures always decrease or maintain stability.
    """
    if w is None:
        w = DEFAULT_W
    d = _clamp_difficulty(difficulty)
    r = max(0.01, min(0.99, retrievability))

    # Fail multiplier: always < 1.0
    fail_mult = w[10] + w[11] * (d / 10.0) + w[12] * (1.0 - r)
    fail_mult = max(0.1, min(1.0, fail_mult))

    new_s = stability * fail_mult
    return max(0.1, new_s)


def difficulty_after_review(difficulty: float, grade: int, w: list[float] | None = None) -> float:
    """Compute new difficulty after a review.

    D' = D + w[4] * (3 - G) + w[5] * (1 - D/10) * (3 - G)

    Higher grade (success) → lower difficulty.
    Lower grade (failure) → higher difficulty.
    Clamped to [1, 10].
    """
    if w is None:
        w = DEFAULT_W
    d = _clamp_difficulty(difficulty)
    g = _clamp_grade(grade)

    delta = w[4] * (3.0 - g) + w[5] * (1.0 - d / 10.0) * (3.0 - g)
    new_d = d + delta
    return _clamp_difficulty(new_d)


def myelin_signals_to_grade(
    success: bool | None,
    prediction_error: float | None,
    surprise: float | None,
    confidence: float | None,
) -> int:
    """Map Myelin's signals to FSRS grade (1-4).

    1 = complete failure (failed, high surprise)
    2 = partial failure (failed, low surprise, or mixed)
    3 = partial success (succeeded but uncertain)
    4 = complete success (succeeded confidently)
    """
    if success is None:
        # Unknown — use confidence as guide
        if confidence and confidence < 0.4:
            return 3  # cautiously optimistic
        return 4  # assume success

    if not success:
        # Failed
        if surprise and surprise > 0.5:
            return 1  # complete failure, very surprising
        return 2  # partial failure

    # Succeeded
    pe = abs(prediction_error or 0.0)
    if pe > 0.5:
        return 3  # succeeded despite high prediction error
    if confidence and confidence > 0.8:
        return 4  # confidently succeeded
    return 3  # default partial success


def optimal_review_interval(stability_days: float, target_retention: float = 0.9) -> float:
    """Compute optimal interval to achieve target retention.

    I = S * ln(target_retention) / ln(0.9)
    """
    if stability_days <= 0:
        return 0.0
    return stability_days * math.log(target_retention) / math.log(0.9)


def hybrid_activation(
    actr_activation: float,
    fsrs_retrievability: float,
    fsrs_stability: float,
    actr_weight: float = 0.3,
    fsrs_weight: float = 0.7,
) -> float:
    """Blend ACT-R activation with FSRS retrievability.

    ACT-R captures frequency/recency patterns.
    FSRS captures optimal schedule patterns.
    """
    # Normalize ACT-R to [0, 1] range
    actr_norm = max(0.0, min(1.0, (actr_activation + 3.0) / 6.0))
    return actr_weight * actr_norm + fsrs_weight * fsrs_retrievability


# ── Helpers ──────────────────────────────────────────────────────────


def _clamp_grade(grade: int) -> int:
    return max(1, min(4, grade))


def _clamp_difficulty(d: float) -> float:
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, d))


# ── Schedule Priority ────────────────────────────────────────────────


def review_priority(
    retrievability: float,
    difficulty: float,
    importance: float,
    days_since_review: float,
    optimal_interval: float,
) -> float:
    """Compute review priority.

    Higher = more urgent to review.
    Combines: low retrievability, high difficulty,
    high importance, overdue status.
    """
    # Low retrievability = urgent
    ret_factor = 1.0 - retrievability

    # Overdue ratio: how far past optimal interval
    if optimal_interval > 0:
        overdue = max(0.0, days_since_review - optimal_interval) / optimal_interval
    else:
        overdue = 0.0
    overdue = min(1.0, overdue)

    return (
        0.35 * ret_factor
        + 0.15 * (difficulty / MAX_DIFFICULTY)
        + 0.25 * importance
        + 0.25 * overdue
    )


# ── FSRS State Management ───────────────────────────────────────────


class FSRSScheduler:
    """Manages FSRS review schedules for memories.

    Tracks per-memory Difficulty, Stability, and last review time.
    Computes optimal review intervals and priority scores.
    Does NOT store to DB directly — returns computed values for
    the caller to persist.
    """

    def __init__(self, w: list[float] | None = None):
        self.w = w or DEFAULT_W.copy()

    def get_retrievability(self, stability_days: float, last_reviewed: str | float | None) -> float:
        """Compute current retrievability for a memory."""
        if stability_days <= 0 or last_reviewed is None:
            return 1.0

        elapsed = self._days_since(last_reviewed)
        return forgetting_curve(stability_days, elapsed)

    def schedule_next_review(
        self,
        current_stability: float,
        current_difficulty: float,
        current_retrievability: float,
        grade: int,
    ) -> dict[str, Any]:
        """Compute next state after a review.

        Returns:
        {
            'new_stability': float,
            'new_difficulty': float,
            'optimal_interval_days': float,
            'next_retrievability': float,
        }
        """
        is_failure = grade <= 2

        if is_failure:
            new_s = stability_after_fail(
                current_stability, current_difficulty, current_retrievability, grade, self.w
            )
        else:
            new_s = stability_after_review(
                current_stability, current_difficulty, current_retrievability, grade, self.w
            )

        new_d = difficulty_after_review(current_difficulty, grade, self.w)
        optimal_interval = optimal_review_interval(new_s)
        next_r = forgetting_curve(new_s, optimal_interval)

        return {
            "new_stability": round(new_s, 4),
            "new_difficulty": round(new_d, 4),
            "optimal_interval_days": round(optimal_interval, 2),
            "next_retrievability": round(next_r, 4),
        }

    def init_memory(self, grade: int = 4) -> dict[str, Any]:
        """Initialize FSRS state for a new memory.

        Returns initial stability, difficulty, and optimal interval.
        """
        s = initial_stability(grade, self.w)
        d = initial_difficulty(grade, self.w)
        interval = optimal_review_interval(s)
        return {
            "stability": round(s, 4),
            "difficulty": round(d, 4),
            "optimal_interval_days": round(interval, 2),
            "retrievability": 1.0,
        }

    def record_review(
        self,
        stability: float,
        difficulty: float,
        retrievability: float,
        grade: int,
    ) -> dict[str, Any]:
        """Record a review and compute new state.

        Shortcut: takes current state and grade, returns next state.
        """
        return self.schedule_next_review(stability, difficulty, retrievability, grade)

    def get_overdue_memories(
        self,
        memories: list[dict[str, Any]],
        max_count: int = 20,
        min_priority: float = 0.3,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Get memories that need review, sorted by priority.

        memories: list of dicts with keys:
            stability, difficulty, last_reviewed, retrievability, importance (0-1)

        Returns list of (priority, memory) tuples, highest priority first.
        """
        candidates: list[tuple[float, dict[str, Any]]] = []
        time.time()

        for mem in memories:
            mem.get("stability", DEFAULT_STABILITY_DAYS)
            difficulty = mem.get("difficulty", DEFAULT_DIFFICULTY)
            retrievability = mem.get("retrievability", 1.0)
            importance = mem.get("importance", 0.5)
            last_reviewed = mem.get("last_reviewed")
            optimal_int = mem.get("optimal_interval_days", 30.0)

            days_since = self._days_since(last_reviewed) if last_reviewed else 999.0
            priority = review_priority(
                retrievability, difficulty, importance, days_since, optimal_int
            )

            if priority >= min_priority:
                candidates.append((priority, mem))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:max_count]

    @staticmethod
    def _days_since(timestamp: str | float | None) -> float:
        if timestamp is None:
            return 999.0
        if isinstance(timestamp, (int, float)):
            ts = timestamp
        else:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                ts = dt.timestamp()
            except (ValueError, TypeError):
                return 999.0
        return (time.time() - ts) / 86400.0

    def get_status(self) -> dict[str, Any]:
        """Return FSRS status for diagnostics."""
        return {
            "parameters": [round(w_, 5) for w_ in self.w],
            "min_difficulty": MIN_DIFFICULTY,
            "max_difficulty": MAX_DIFFICULTY,
            "default_stability_days": DEFAULT_STABILITY_DAYS,
        }
