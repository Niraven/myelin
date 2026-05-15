# Task: Self-Model & Introspection

**Status:** Complete
**Priority:** P1
**Assignee:** builder
**Dependencies:** 10-prediction-error-learning, 13-schema-learning
**Domain:** development/mlops

## Goal

Give Myelin a model of itself — the ability to measure its own confidence, detect its own biases, map its competence areas, and know what it doesn't know. Self-model is the foundation of meta-cognition.

## Why

A system that can't assess its own confidence will be overconfident in unfamiliar situations and underconfident in familiar ones. Self-model lets Myelin calibrate its outputs, detect systematic errors (biases), and focus improvement efforts on genuine weaknesses. This is the difference between blind execution and calibrated judgment.

## Architecture

```
Per procedure:
  → Track: confidence (predicted success), actual success, trial count
  → Calibration: confidence vs actual accuracy over last 20 trials
  → Bias detection per dimension:
     • Confirmation bias: favor evidence that confirms existing beliefs
     • Recency bias: overweight recent events
     • Overconfidence: confidence > accuracy persistently

Knowledge map:
  → Domain → procedures → confidence levels
  → Competence clusters: mastered, practiced, novice, absent
  → Uncertainty hot spots: domains with high variance but low confidence
```

## Key Design Decisions

1. **Confidence calibration** — Brier score per procedure. Perfect calibration = 0. Overconfident = positive. Underconfident = negative.
2. **Three bias detectors** — confirmation, recency, overconfidence. Each returns a bias score [0,1]. Threshold > 0.5 flags the bias.
3. **Competence map** — 4 tiers: mastered (confidence > 0.8, > 20 trials), practiced (0.6-0.8, > 10 trials), novice (< 0.6 or < 5 trials), absent (no data).
4. **Uncertainty tracking** — prediction variance + confidence deficit = uncertainty score. High uncertainty regions are flagged for curiosity engine.
5. **Non-invasive** — self-model reads performance data, it doesn't modify it. All calibration is observational.

## Implementation Details

- `SelfModel` in `cognitive/self_model.py` — 425 lines
- Bias detection runs every 10 sleep cycles (configurable)
- Confidence calibration outputs: average_brier, overconfident_pct, underconfident_pct
- Competence map serializable for export (useful for dashboards)
- 19 tests covering: calibration calculation, bias detection (all 3 types), knowledge map building, uncertainty scoring, edge cases (no data, single trial, perfect calibration)

## Files

- `src/myelin/cognitive/self_model.py`
- `tests/test_self_model.py` — 196 lines
