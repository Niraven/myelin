# Task: FSRS Spaced Repetition

**Status:** Complete
**Priority:** P1
**Assignee:** builder
**Dependencies:** 04-importance-scoring
**Domain:** development/mlops

## Goal

Implement the FSRS-5 (Free Spaced Repetition Scheduler v5) algorithm — the state-of-the-art spaced repetition system used by Anki and Memrise. FSRS schedules optimal review times based on a 19-parameter DSR (Difficulty, Stability, Retrievability) model.

## Why

Without spaced repetition, Myelin reviews everything at roughly the same frequency. Things the agent knows well get checked as often as things it's about to forget. FSRS optimizes review timing so Myelin focuses on memories at the edge of forgetting — maximizing learning efficiency per review.

## Architecture

```
Memory review → compute retrievability R(t) from last review
             → R(t) < 0.9? → schedule review now
             → Update difficulty (D) based on grade (4=perfect, 1=fail)
             → Update stability (S) based on D, grade, past S
             → Compute next review date from S and requested R
             → FSRS-5 uses 19 configurable parameters (w[0]..w[18])
```

## Key Design Decisions

1. **FSRS-5 algorithm** — implements the canonical DSR model: S_new = f(S_old, D, grade, R), D_new = g(D, grade), next_date = h(S_new, R_requested). Parameters from Anki's published defaults.
2. **Difficulty model** — starts at 5.0 (easy) or 7.0 (hard). Adjusts by grade: grade 4 → -0.08, grade 3 → +0.06, grade 2 → +0.40, grade 1 → +0.60. Clamped to [1.0, 10.0].
3. **Stability model** — retrievability-based: R(t) = e^(-t/S). When R < 0.9, it's time to review.
4. **Hybrid with ACT-R blending** — FSRS is the primary scheduler, but ACT-R's decay equation provides a fallback for new memories with no history.
5. **Batching** — reviews are batched and run during sleep cycles, not in real-time. Reduces overhead.

## Implementation Details

- `FSRSScheduler` in `cognitive/fsrs_scheduler.py` — 409 lines
- 19 FSRS-5 parameters with Anki-compatible defaults
- Grade system: 4 (perfect) → 3 (good) → 2 (hard) → 1 (fail)
- Min stability: 0.1 (preventing zero-division)
- Max difficulty: 10.0 (upper bound)
- 10 tests covering: DSR model, next date computation, grade effects, edge cases (grade=1 review scheduling, extreme difficulties)

## Files

- `src/myelin/cognitive/fsrs_scheduler.py`
- `tests/test_fsrs_scheduler.py` — 321 lines
