# Task: Prioritized Replay

**Status:** Complete
**Priority:** P0
**Assignee:** builder
**Dependencies:** 10-prediction-error-learning
**Domain:** development/mlops

## Goal

Replace FIFO replay buffer with prioritized experience replay (PER) — sample memories in order of how much the system can learn from them, not in arrival order.

## Why

Not all memories are equally valuable for learning. A failed deployment is more informative than a routine success. PER ensures Myelin replays the most instructive experiences first, accelerating learning by focusing compute on high-surprise, high-TD-error memories.

## Architecture

```
Memory → compute priority = f(surprise, recency, importance, TD-error)
       → store in priority queue (max-heap)
       → sample: rank-based with IS (importance sampling) weights
       → replay: feed to reconsolidation + NREM sleep
       → update priority after replay
       → FreshPER: penalize staleness (memories not replayed for N cycles)
```

## Key Design Decisions

1. **Rank-based PER** — priority = 1/(rank + α)^β. Simpler and more stable than proportional PER.
2. **IS weights** — importance sampling (w = (1/N · 1/P(i))^β_anneal) to correct for non-uniform sampling bias. β anneals from 0.4 to 1.0 over training.
3. **FreshPER** — memories not replayed within 10 cycles get a staleness bonus to their priority. Prevents tail neglect.
4. **Priorities decay** — every 24 hours, all priorities decay by 0.95. Old surprises become less relevant.

## Implementation Details

- `PrioritizedReplay` in `cognitive/prioritized_replay.py` — 324 lines
- Priority formula: `P(i) = (rank(i)^(-α)) / sum(rank(j)^(-α))`
- IS weight formula: `w_i = (1/N * 1/P(i))^β` where β anneals from 0.4→1.0
- FreshPER: `priority *= 1.5` if not replayed in 10 cycles
- 25 tests covering: rank distribution, IS weight correctness, FreshPER staleness, priority decay, edge cases (single memory, duplicate priorities)

## Files

- `src/myelin/cognitive/prioritized_replay.py`
- `tests/test_prioritized_replay.py` — 438 lines
