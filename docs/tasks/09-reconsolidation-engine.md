# Task: Reconsolidation Engine

**Status:** Complete
**Priority:** P0
**Assignee:** builder
**Dependencies:** 03-query-synthesis, 04-importance-scoring
**Domain:** development/mlops

## Goal

Implement memory reconsolidation — the biological process where existing memories become labile (modifiable) when retrieved, then re-stabilize. This replaces naive overwrite with a theory-grounded update mechanism.

## Why

Pure consolidation (Phase 1) only creates new memories. Reconsolidation lets Myelin **update existing memories** when new evidence contradicts them — without deleting or blindly appending. This is the difference between a system that accumulates and a system that learns.

## Architecture

```
Episode retrieval → lability check (within 6h window?)
                 → prediction error calculation
                 → PE below threshold? → strengthen existing
                 → PE moderate? → update with new info (reconsolidation)
                 → PE high? → block update, flag contradiction (β penalty)
                 → stability upgrade if reinforced consistently
```

## Key Design Decisions

1. **6-hour lability window** — memories are modifiable only within 6 hours of retrieval (based on rodent reconsolidation research). Outside this window they're stable and skipped.
2. **Prediction-error gating** — PE < 0.1: strengthen. PE 0.1-0.7: reconsolidate (update). PE > 0.7: block, flag contradiction.
3. **Stability tracking** — each memory has a stability counter. Repeated successful predictions increment it. High-stability memories become resistant to PE-modest updates.
4. **β contradiction penalty** — memories flagged as contradictions get a confidence penalty. After 3 contradictions, the memory is deprioritized.

## Implementation Details

- `ReconsolidationEngine` in `cognitive/reconsolidator.py`
- Integrates with `PredictionLearner` for PE computation
- Uses `MemoryStore.get_by_id()` + `MemoryStore.update()`
- Lability tracked via `episodic.lability_window_end` column in schema V4
- 62 tests covering: PE thresholds, lability enforcement, stability protection, β penalty accumulation, edge cases (null PE, already-stable, etc.)

## Files

- `src/myelin/cognitive/reconsolidator.py` — 921 lines
- `tests/test_reconsolidation.py` — 676 lines
- `src/myelin/core/schema.py` (V4) — added `lability_window_end`, `contradiction_count`, `stability` columns
