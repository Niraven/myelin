# Task: Schema Learning

**Status:** Complete
**Priority:** P0
**Assignee:** builder
**Dependencies:** 03-query-synthesis, 11-two-phase-sleep
**Domain:** development/mlops

## Goal

Automatically discover behavioral schemas — reusable patterns abstracted from multiple related episodes. A schema is "deploy with confidence check" derived from 5 successful deployments, or "troubleshoot API error" from 3 failed API calls.

## Why

Individual memories are atomic. Schemas are the next level of abstraction — they let Myelin reason about **classes of situations**, not just individual events. Without schemas, every new situation looks novel. With schemas, Myelin recognizes "oh, this is another instance of the deployment pattern" and applies accumulated knowledge.

## Architecture

```
Episodic memories → Jaccard similarity matrix (ε=0.30 threshold)
                 → cluster similar episodes
                 → 3+ episodes in cluster? → promote to schema
                 → Schema lifecycle: nascent → active → mature
                 → Schema extraction: common steps, entities, outcomes
                 → Duplicate merge: if new schema is 85%+ similar to existing, merge
```

## Key Design Decisions

1. **Jaccard similarity** — `|intersection|/|union|` of entity sets between two episodes. Threshold ε=0.30 (tuned for 10-20 entity episodes).
2. **3-episode minimum** — a schema needs at least 3 episodes to be promoted from nascent to active. Prevents overfitting to 1-2 coincidental matches.
3. **Schema lifecycle** — nascent (3+ episodes) → active (10+ episodes) → mature (50+ episodes, locked structure). Mature schemas can still be updated but only by high-confidence events.
4. **Duplicate merge** — if a new schema overlaps 85%+ with an existing one by entity Jaccard, merge instead of creating duplicate. Keeps the schema count manageable.

## Implementation Details

- `SchemaLearner` in `cognitive/schema_learner.py` — 465 lines
- Clustering runs as part of sleep cycle, not in real-time
- Schema stores: entity set, episodes, outcome distribution, confidence
- 30 tests covering: Jaccard computation, clustering, lifecycle transitions, duplicate merge, edge cases (1 episode, identical episodes, empty sets)

## Files

- `src/myelin/cognitive/schema_learner.py`
- `tests/test_schema_learner.py` — 487 lines
- `src/myelin/core/schema.py` (V4) — added `schemas` table
