# Task: Two-Phase Sleep (NREM → REM)

**Status:** Complete
**Priority:** P0
**Assignee:** builder
**Dependencies:** 04-importance-scoring, 09-reconsolidation-engine, 10-prediction-error-learning
**Domain:** development/mlops

## Goal

Replace single-phase batch consolidation with mammalian-style two-phase sleep: **NREM (non-rapid eye movement)** followed by **REM (rapid eye movement)** . NREM strengthens and prunes. REM recombines and generates novelty.

## Why

Sleep is not one process — it's two complementary phases. NREM consolidates by replaying strong patterns (Hebbian strengthening + synaptic downscaling). REM integrates by forming novel connections between disparate memories. Together they produce both stability and creativity. Single-phase batch does neither well.

## Architecture

```
Sleep cycle:
  Phase 1: NREM (50% of cycle time)
    → Hebbian strengthening: co-activated memories → strengthen mutual connections
    → Synaptic downscaling: weak connections (below threshold) → prune
    → Temporal substates: order memories by recency within replay
  Phase 2: REM (50% of cycle time)
    → Random walk dreaming: pick seed memory, traverse graph, form novel pairs
    → Counterfactual generation: "what if A had happened instead of B?"
    → TAG scoring: Track-Adjust-Grow — retain novel connections that pass utility threshold
  Output: consolidated memories + novel schemas + updated importance scores
```

## Key Design Decisions

1. **NREM first, REM second** — biological order. NREM stabilizes the strong signals; REM explores what's left.
2. **50/50 split per cycle** — configurable via `nrem_ratio` parameter.
3. **Multiple cycles** — default 3 cycles per sleep session, each NREM→REM.
4. **Hebbian plasticity** — memories that fire together in replay get their connection weights increased.
5. **Synaptic downscaling** — connections below 0.15 weight are pruned. Prevents memory bloat.
6. **Random walk dreaming** — BFS-like traversal from seed memory, limited depth 3, capped at 50 nodes.

## Implementation Details

- `NREMSleepPhase` in `cognitive/nrem_sleep.py` — 457 lines
- `REMSleepPhase` in `cognitive/rem_sleep.py` — 525 lines
- `SleepConsolidator` in `cognitive/sleep.py` — orchestrates both phases
- 21 NREM tests + REM tests merged into `test_two_phase_sleep.py` (582 lines)
- Priorities: important (>0.7), routine (0.3-0.7), ephemeral (<0.3) — processed differently

## Files

- `src/myelin/cognitive/nrem_sleep.py`
- `src/myelin/cognitive/rem_sleep.py`
- `src/myelin/cognitive/sleep.py` (rewritten)
- `tests/test_two_phase_sleep.py`
