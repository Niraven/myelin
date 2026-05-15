# Task: Prediction Error Learning

**Status:** Complete
**Priority:** P0
**Assignee:** builder
**Dependencies:** 01-hybrid-entity-extraction
**Domain:** development/mlops

## Goal

Build a forward model that predicts the outcome of procedures, then computes prediction error (surprise) when reality diverges. This surprise signal drives reconsolidation, replay prioritization, and exploration.

## Why

Prediction error is the fundamental learning signal in neuroscience and reinforcement learning. Without it, Myelin has no way to know which memories are wrong, which procedures need updating, or which situations are genuinely novel. PE turns raw observation into prioritized learning.

## Architecture

```
Procedure execution → observe outcome
                   → run forward model → predict outcome
                   → compare actual vs predicted → TD-error
                   → high TD-error? → flag for reconsolidation
                   → low TD-error? → increment stability
                   → surprise = abs(TD-error) → used for replay priority
```

## Key Design Decisions

1. **Forward model per procedure** — each procedure builds its own prediction model based on past outcomes. No cross-procedure interference.
2. **TD-error (temporal difference)** — difference between predicted outcome and actual outcome. Clipped to [0,1] range.
3. **Modulated learning rate** — learning rate adjusts based on recent volatility. High volatility = higher learning rate.
4. **Surprise metric** — exponentially weighted moving average of recent TD-errors. Used by curiosity engine and prioritized replay.

## Implementation Details

- `PredictionLearner` in `cognitive/prediction_learner.py`
- Maintains per-procedure outcome history (last 20)
- Computes rolling mean prediction from history
- TD-error = |actual - predicted| clipped to [0,1]
- Surprise = EMA of recent TD-errors (α=0.3)
- 29 tests covering: prediction accuracy, TD-error bounds, surprise computation, edge cases (empty history, single sample, NaN inputs)

## Files

- `src/myelin/cognitive/prediction_learner.py` — 373 lines
- `tests/test_prediction_learner.py` — 351 lines
