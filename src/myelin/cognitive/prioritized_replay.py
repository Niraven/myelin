"""Prioritized Experience Replay (PER) with importance sampling and staleness prevention.

Implements Schaul et al. 2015 PER with FreshPER adaptations:

1. COMPOSITE PRIORITY SCORING
2. RANK-BASED SAMPLING with stratification
3. IMPORTANCE SAMPLING (IS) WEIGHTS with beta annealing
4. STALENESS PREVENTION (FreshPER)
5. REPLAY LOOP with entity strengthening
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import ProcessName
from ..knowledge.entities import EntityStore, extract_entities_from_text
from ..knowledge.graph import KnowledgeGraph
from .base import CognitiveProcess

# ── Constants ──────────────────────────────────────────────────────

W_TD_ERROR = 0.35
W_SURPRISE = 0.30
W_IMPORTANCE = 0.35
IMPORTANCE_HALF_LIFE_HOURS = 168  # 1 week
PER_ALPHA = 0.6
IS_BETA_START = 0.4
IS_BETA_END = 1.0
FRESHPER_DECAY = 0.95
PRIORITY_FLOOR = 0.1
MAX_REPLAY_COUNT = 10
BATCH_SIZE = 20
SCHEMA_HINT_THRESHOLD = 5


def _new_id() -> str:
    return uuid4().hex[:16]


def _hours_since(timestamp_str: str) -> float:
    try:
        ep_ts = datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return 0.0
    delta = datetime.utcnow() - ep_ts
    return max(0.0, delta.total_seconds() / 3600.0)


class PrioritizedReplay(CognitiveProcess):
    """Prioritized experience replay process."""

    name = ProcessName.PRIORITIZED_REPLAY

    def __init__(
        self,
        db: Database,
        entity_store: EntityStore | None = None,
        graph: KnowledgeGraph | None = None,
    ):
        super().__init__(db)
        self.entities = entity_store or EntityStore(db)
        self.graph = graph or KnowledgeGraph(db)

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        # Get cycle count from previous run
        prev = self.db.fetchone(
            "SELECT details FROM process_runs "
            "WHERE process_name = ? AND status = 'completed' "
            "ORDER BY started_at DESC LIMIT 1",
            (self.name.value,),
        )
        cycle_count = 0
        if prev and prev.get("details"):
            details = prev["details"]
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}
            if isinstance(details, dict):
                cycle_count = details.get("cycle_count", 0) or 0
        cycle_count += 1

        # Step 1: Compute priorities
        scored = self._compute_priorities()

        # Step 2: Sample batch
        batch = self._sample_batch(cycle_count)
        if not batch:
            return {
                "cycle_count": cycle_count,
                "sampled": 0,
                "replayed": 0,
                "entities_strengthened": 0,
                "schema_hints": 0,
                "beta": self._beta(cycle_count),
                "total_scored": len(scored),
                "note": "No available episodes for replay",
            }

        # Step 3: Replay
        replayed = 0
        entities_strengthened = 0
        schema_hints = 0
        cluster_counts: dict[str, int] = defaultdict(int)

        for ep in batch:
            result = self._replay_episode(ep)
            if result["replayed"]:
                replayed += 1
            entities_strengthened += result["entities_strengthened"]

            cid = ep.get("cluster_id")
            if cid:
                cluster_counts[cid] += 1
                if cluster_counts[cid] >= SCHEMA_HINT_THRESHOLD:
                    schema_hints += 1
                    cluster_counts[cid] = 0

        self.db.commit()

        return {
            "cycle_count": cycle_count,
            "sampled": len(batch),
            "replayed": replayed,
            "entities_strengthened": entities_strengthened,
            "schema_hints": schema_hints,
            "beta": self._beta(cycle_count),
            "total_scored": len(scored),
        }

    # ── Priority Scoring ───────────────────────────────────────────

    def _compute_priority(self, ep: dict[str, Any]) -> float:
        """Composite priority: 0.35*|td_error| + 0.30*surprise + 0.35*(importance*exp(-Δ/τ))."""
        td_error = ep.get("td_error")
        surprise = ep.get("surprise_score")
        importance = ep.get("importance_score", 0.5)
        timestamp = ep.get("timestamp", "")

        td_term = 0.0
        if td_error is not None:
            td_term = W_TD_ERROR * min(abs(float(td_error)), 1.0)

        surprise_term = 0.0
        if surprise is not None:
            surprise_term = W_SURPRISE * min(float(surprise), 1.0)

        hours = _hours_since(timestamp)
        decay = math.exp(-hours / IMPORTANCE_HALF_LIFE_HOURS)
        importance_term = W_IMPORTANCE * min(float(importance), 1.0) * decay

        raw = td_term + surprise_term + importance_term

        # FreshPER: penalize repeatedly-replayed episodes
        replay_count = int(ep.get("replay_count", 0))
        stale_penalty = math.pow(FRESHPER_DECAY, replay_count)
        priority = raw * stale_penalty

        # Apply floor — never let priority drop below minimum
        return max(priority, PRIORITY_FLOOR)

    def _compute_priorities(self) -> list[dict[str, Any]]:
        """Score all eligible episodes and update priority in DB."""
        episodes = self.db.fetchall(
            "SELECT * FROM episodes "
            "WHERE replay_count < ? OR replay_count IS NULL",
            (MAX_REPLAY_COUNT,),
        )
        scored: list[dict[str, Any]] = []
        for ep in episodes:
            priority = self._compute_priority(ep)
            self.db.update("episodes", ep["id"], {"priority_score": priority})
            ep["priority_score"] = priority
            scored.append(ep)
        self.db.commit()
        # Sort descending by priority
        scored.sort(key=lambda e: e.get("priority_score", 0) or 0, reverse=True)
        return scored

    # ── Rank-Based Stratified Sampling ─────────────────────────────

    def _beta(self, cycle_count: int) -> float:
        """Annealed beta: IS_BETA_START at cycle 1, IS_BETA_END asymptote."""
        return min(IS_BETA_END, IS_BETA_START + (IS_BETA_END - IS_BETA_START) * cycle_count / 100)

    def _sample_batch(self, cycle_count: int) -> list[dict[str, Any]]:
        """Rank-based stratified sampling with IS weight computation."""
        # Get all episodes sorted by priority, excluding over-replayed ones
        candidates = self.db.fetchall(
            "SELECT * FROM episodes "
            "WHERE (replay_count < ? OR replay_count IS NULL) "
            "ORDER BY priority_score DESC",
            (MAX_REPLAY_COUNT,),
        )
        if not candidates:
            return []

        n = len(candidates)
        batch_size = min(BATCH_SIZE, n)
        if batch_size < 1:
            return []

        # Assign ranks (1-indexed)
        # P(i) = (1/rank)^alpha / sum((1/k)^alpha)
        denominator = sum(math.pow(1.0 / (i + 1), PER_ALPHA) for i in range(n))

        # Stratified: partition into batch_size buckets, sample 1 per bucket
        bucket_size = n / batch_size
        batch: list[dict[str, Any]] = []
        import random

        rng = random.Random(time.time() + hash(str(candidates[:1])))

        for b in range(batch_size):
            start = int(b * bucket_size)
            end = int((b + 1) * bucket_size) if b < batch_size - 1 else n
            if start >= end:
                continue
            idx = rng.randrange(start, end)
            ep = dict(candidates[idx])

            # Compute IS weight
            rank = idx + 1
            prob = math.pow(1.0 / rank, PER_ALPHA) / denominator
            beta = self._beta(cycle_count)
            is_weight = math.pow(1.0 / (n * prob) if prob > 0 else 0.0, beta)
            ep["_is_weight"] = is_weight
            ep["_rank"] = rank
            ep["_prob"] = prob
            batch.append(ep)

        return batch

    # ── Replay Loop ────────────────────────────────────────────────

    def _replay_episode(self, ep: dict[str, Any]) -> dict[str, Any]:
        """Replay a single episode: increment count, strengthen entities."""
        result: dict[str, Any] = {"replayed": False, "entities_strengthened": 0}

        ep_id = ep["id"]
        current_count = int(ep.get("replay_count", 0))

        # Increment replay_count
        self.db.update("episodes", ep_id, {"replay_count": current_count + 1})
        result["replayed"] = True

        # Replay action sequence if procedure reference exists
        procedure_id = ep.get("procedure_id")
        if procedure_id:
            try:
                from ..memory.procedural import ProceduralMemory

                proc_mem = ProceduralMemory(self.db)
                proc = proc_mem.get(procedure_id)
                if proc:
                    self.db.update(
                        "procedures",
                        procedure_id,
                        {"activation_score": float(proc.get("activation_score", 0)) + 0.1},
                    )
            except Exception:
                pass

        # Strengthen entity relationships
        content = ep.get("content_text", "") or ""
        action = ep.get("action", "") or ""
        raw_entities = extract_entities_from_text(content, action)
        if len(raw_entities) >= 2:
            entity_ids: list[str] = []
            for raw in raw_entities:
                found = self.entities.find_by_canonical(raw["canonical_name"], raw["entity_type"])
                if found:
                    entity_ids.append(found["id"])

            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    src_id = entity_ids[i]
                    tgt_id = entity_ids[j]
                    existing = self.db.fetchone(
                        "SELECT id, strength, evidence_count, evidence_episodes "
                        "FROM relationships "
                        "WHERE ((source_entity_id = ? AND target_entity_id = ?) "
                        "   OR (source_entity_id = ? AND target_entity_id = ?)) "
                        "AND relation_type = ?",
                        (src_id, tgt_id, tgt_id, src_id, "related_to"),
                    )
                    if existing:
                        new_strength = min(float(existing["strength"]) + 0.15, 10.0)
                        ep_list = json.loads(existing["evidence_episodes"] or "[]")
                        if ep_id not in ep_list:
                            ep_list.append(ep_id)
                        self.db.update(
                            "relationships",
                            existing["id"],
                            {
                                "strength": new_strength,
                                "evidence_count": int(existing["evidence_count"]) + 1,
                                "evidence_episodes": ep_list,
                                "last_observed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            },
                        )
                    else:
                        self.graph.add_relationship(
                            source_entity_id=src_id,
                            target_entity_id=tgt_id,
                            relation_type="related_to",
                            episode_id=ep_id,
                            strength=1.15,
                        )
                    result["entities_strengthened"] += 1

        return result
