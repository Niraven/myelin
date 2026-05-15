# Task: Importance Scoring

**Status:** Planned
**Priority:** P2
**Assignee:** builder
**Dependencies:** None
**Domain:** development

## Goal

Add a three-signal importance score to every episode. Feed it into the multi-signal retriever so important episodes (production fixes, frequent workflows) rank higher than ephemeral noise (one-off tests, exploratory commands).

## Why

Currently every episode is equal. A critical production fix and a throwaway `ls` command are scored identically. This means the retriever can't tell "this matters" from "this is noise."

## Three Signals

### 1. Frequency

Episodes that are part of a large cluster (similar actions, same domain) score higher. An episode about "running deployment" that's been done 20 times is more important than "testing a random command."

**Implementation:** During sleep, the clusterer groups similar episodes. Cluster size → frequency score (0-1).

```python
frequency_score = min(1.0, cluster_size / 10)  # 10+ episodes = max score
```

### 2. Consequence

Episodes that are followed by high-success-rate follow-up actions score higher. If "deploy" is always followed by "verify" at 95% success, the deploy episode is consequential.

**Implementation:** Look at the success rate of the 5 episodes immediately following this one in the same session.

```python
follow_up_successes = count_success_follow_ups(episode_id, window=5)
consequence_score = follow_up_successes / 5 if follow_up_successes > 0 else 0.3
```

### 3. Recency (existing)

Already implemented in the retriever via temporal scoring. Standard decay curve.

## Integration

Importance scores are computed during sleep (batch) and stored on the episode record. The retriever picks them up as an additional signal in the composite score.

### Storage

Add an `importance_score` column to the episodes table (default: 0.5):
```sql
ALTER TABLE episodes ADD COLUMN importance_score REAL NOT NULL DEFAULT 0.5;
```

### Computation (during sleep)

```python
class ImportanceComputer:
    def compute(self, db, episodes, clusters):
        for ep in episodes:
            freq = self._frequency(ep, clusters)
            conseq = self._consequence(ep, db)
            recency = self._recency(ep)
            importance = 0.4 * freq + 0.4 * conseq + 0.2 * recency
            db.update("episodes", ep["id"], {"importance_score": importance})
```

### Retrieval Integration

In `retriever.py`, add importance as a signal weight:

```python
weights = weights or {
    "text": 0.25,
    "vector": 0.25,
    "entity": 0.20,
    "temporal": 0.10,
    "activation": 0.10,
    "importance": 0.10,  # NEW
}
```

## Acceptance Criteria

- [ ] Frequency computed from cluster size during sleep
- [ ] Consequence computed from follow-up success rate
- [ ] Recency reuses existing temporal scoring
- [ ] Importance score stored on each episode (0.0-1.0)
- [ ] Retriever accepts `importance` in weights dict
- [ ] Default weight: 0.10 (configurable)
- [ ] Zero regression on existing queries (default weight is low)
- [ ] All new episodes get default importance 0.5 until sleep runs

## Files Changed

| File | Change |
|------|--------|
| `core/schema.py` | Add `importance_score` column to episodes table |
| `cognitive/sleep.py` | Add importance computation phase |
| `cognitive/importance.py` | New — ImportanceComputer class |
| `memory/retriever.py` | Add importance to signal weights |
| `memory/episodic.py` | Add importance update method |

## Estimated Effort

- Schema migration: ~30 min
- Importance computation: ~2 hours
- Retrieval integration: ~1 hour
- Testing: ~1 hour
