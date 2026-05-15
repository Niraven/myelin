# Task: Curiosity & Active Learning

**Status:** Complete
**Priority:** P1
**Assignee:** builder
**Dependencies:** 10-prediction-error-learning, 12-prioritized-replay
**Domain:** development/mlops

## Goal

Give Myelin intrinsic motivation — the ability to identify what it doesn't know, formulate learning goals, and actively seek information to fill knowledge gaps. This transforms Myelin from a passive observer to an active learner.

## Why

Without curiosity, Myelin only learns what it happens to observe. Active learning lets it **choose what to learn next** based on knowledge gaps, prediction uncertainty, and coverage holes. This is what separates a learner from a logger.

## Architecture

```
Knowledge gap detection:
  → Low-confidence procedures (< 0.5) → exploration candidates
  → High surprise regions (recent PE > 0.6) → investigation targets
  → Unvisited entity clusters → coverage gaps
  → Rare entity in dense graph → novelty bonus

Exploration vs exploitation:
  → Epsilon-greedy: explore with probability ε (decays from 0.3→0.05)
  → Exploration → generate "what happens if X?" queries
  → Exploitation → replay high-value memories

Learning goals:
  → Rate-limited to 5 active goals max
  → Goal types: understand, verify, explore
  → Completion criteria: confidence > 0.8 or 3 consistent observations
```

## Key Design Decisions

1. **Epsilon-greedy exploration** — simple, proven, effective. ε starts at 0.3 and decays by 0.99 per cycle to 0.05 minimum.
2. **Knowledge gap = low confidence + high uncertainty** — a procedure with confidence < 0.5 and prediction variance > 0.2 is a "known unknown."
3. **Novelty bonus** — entities that appear rarely (< 5 times) in dense clusters get a novelty score boost. Encourages exploring fringe connections.
4. **Learning goals are rate-limited** — max 5 active goals prevents thrashing. Goals auto-complete when confidence threshold is met.

## Implementation Details

- `CuriosityEngine` in `cognitive/curiosity_engine.py` — 1,576 lines (!)
- Generates learning queries from knowledge gaps
- Maintains exploration history to avoid repeat discovery
- Integrated with orchestrator for pre-sleep curiosity phase
- 24 tests covering: gap detection, epsilon-greedy, learning goal lifecycle, novelty bonus, edge cases

## Files

- `src/myelin/cognitive/curiosity_engine.py`
- `tests/test_curiosity_engine.py` — 895 lines
