"""NREM sleep phase: Hebbian strengthening, synaptic downscaling, temporal substates.

NREM (non-rapid eye movement) sleep is the first phase of the two-phase sleep
cycle. It performs:

1. Hebbian strengthening (SCM): strengthen co-occurring concept pairs
   Δs = η * I(c_i) * I(c_j), η=0.15

2. Synaptic downscaling (SHY): s ← 0.85 * s on ALL relationship strengths,
   preventing unbounded growth while co-occurring pairs maintain relative strength.

3. Temporal substate separation (Nature 2025): recent (<2 days) memories cluster
   & strengthen in first pass, older (>2 days) memories integrate with existing
   clusters in second pass — mirroring the contracted/dilated pupil microstructure.

4. Veridical replay: sample high-priority episodes (priority_score > 0.5) and
   replay their action sequences through graph strengthening.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import ProcessName, RelationType
from ..knowledge.entities import (
    EntityStore,
    extract_entities_from_text,
)
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.temporal import TemporalIndex
from .base import CognitiveProcess

HEBBIAN_ETA = 0.15
SYNAPTIC_SCALE = 0.85
RECENT_CUTOFF_DAYS = 2.0
PRIORITY_REPLAY_THRESHOLD = 0.5
NREM_RECENT_LIMIT = 500
NREM_OLD_LIMIT = 500
REPLAY_LIMIT = 20
CLUSTER_RECENT_LABEL = "nrem_recent"
CLUSTER_OLD_LABEL = "nrem_old"


def _new_id() -> str:
    return uuid4().hex[:16]


class NREMPhase(CognitiveProcess):
    """NREM sleep: strengthening, downscaling, temporal substates, veridical replay."""

    name = ProcessName.NREM_SLEEP

    def __init__(
        self,
        db: Database,
        entity_store: EntityStore | None = None,
        graph: KnowledgeGraph | None = None,
        temporal: TemporalIndex | None = None,
    ):
        super().__init__(db)
        self.entities = entity_store or EntityStore(db)
        self.graph = graph or KnowledgeGraph(db)
        self.temporal = temporal or TemporalIndex(db)

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "hebbian_strengthened": 0,
            "synaptic_downscaled": 0,
            "recent_cluster_strengthened": 0,
            "old_integrated": 0,
            "veridical_replays": 0,
            "entities_processed": 0,
        }

        # Step 1: Synaptic downscaling — applied to ALL relationships first
        downscaled = self._apply_synaptic_downscaling()
        results["synaptic_downscaled"] = downscaled

        # Step 2: Temporal substate separation (two-pass)
        recent_eps, old_eps = self._split_by_temporal_age()

        # Pass 1: recent memories → cluster + strengthen
        if recent_eps:
            recent_clusters = self._cluster_episodes(recent_eps, label=CLUSTER_RECENT_LABEL)
            strengthened_recent = self._hebbian_strengthen(recent_eps, recent_clusters)
            results["recent_cluster_strengthened"] = strengthened_recent

        # Pass 2: older memories → integrate with existing clusters
        if old_eps:
            integrated = self._integrate_old_memories(old_eps)
            results["old_integrated"] = integrated

        # Step 3: Veridical replay of high-priority episodes
        replayed = self._veridical_replay()
        results["veridical_replays"] = replayed

        # Summary
        results["entities_processed"] = len(recent_eps) + len(old_eps)

        self.db.commit()
        return results

    # ── Synaptic Downscaling ────────────────────────────────────

    def _apply_synaptic_downscaling(self) -> int:
        """Reduce ALL relationship strengths by SYNAPTIC_SCALE (0.85).

        s_new = max(0.01, s_old * 0.85)
        This prevents unbounded growth. Hebbian strengthening will
        selectively restore strength only to co-occurring pairs.
        """
        rows = self.db.fetchall("SELECT id, strength FROM relationships")
        updated = 0
        for row in rows:
            new_strength = max(0.01, float(row["strength"]) * SYNAPTIC_SCALE)
            # Use direct SQL for batch efficiency
            self.db.execute(
                "UPDATE relationships SET strength = ? WHERE id = ?",
                (new_strength, row["id"]),
            )
            updated += 1
        return updated

    # ── Hebbian Strengthening ───────────────────────────────────

    def _hebbian_strengthen(
        self,
        episodes: list[dict[str, Any]],
        clusters: list[list[dict[str, Any]]],
    ) -> int:
        """Hebbian strengthening: Δs = η * I(c_i) * I(c_j).

        For co-occurring concept pairs in the same episode, strengthen
        the relationship between their entities.
        """
        total_strengthened = 0

        # Build entity-to-episode mapping from current episodes
        for ep in episodes:
            ep_id = ep["id"]
            content = ep.get("content_text", "") or ""
            action = ep.get("action", "") or ""

            # Extract entities from this episode
            raw_entities = extract_entities_from_text(content, action)
            if not raw_entities:
                continue

            # Resolve to entity IDs
            entity_ids: list[str] = []
            for raw in raw_entities:
                found = self.entities.find_by_canonical(raw["canonical_name"], raw["entity_type"])
                if found:
                    entity_ids.append(found["id"])

            # Hebbian: for every pair of entities co-occurring in this episode
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    src_id = entity_ids[i]
                    tgt_id = entity_ids[j]
                    delta = HEBBIAN_ETA * 1.0 * 1.0  # I(c_i) * I(c_j) = 1 for co-occurring

                    # Try all undirected relation types, strengthen the best match
                    existing = self.db.fetchone(
                        "SELECT id, strength, relation_type, evidence_count, evidence_episodes "
                        "FROM relationships "
                        "WHERE ((source_entity_id = ? AND target_entity_id = ?) "
                        "   OR (source_entity_id = ? AND target_entity_id = ?)) "
                        "AND relation_type = ?",
                        (src_id, tgt_id, tgt_id, src_id, RelationType.RELATED_TO.value),
                    )
                    if existing:
                        new_strength = min(
                            float(existing["strength"]) + delta,
                            10.0,  # Hard cap
                        )
                        episodes_list = json.loads(existing["evidence_episodes"] or "[]")
                        if ep_id not in episodes_list:
                            episodes_list.append(ep_id)
                        self.db.update(
                            "relationships",
                            existing["id"],
                            {
                                "strength": new_strength,
                                "evidence_count": int(existing["evidence_count"]) + 1,
                                "evidence_episodes": episodes_list,
                                "last_observed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            },
                        )
                    else:
                        # Create new relationship with base strength + Hebbian boost
                        self.graph.add_relationship(
                            source_entity_id=src_id,
                            target_entity_id=tgt_id,
                            relation_type=RelationType.RELATED_TO.value,
                            episode_id=ep_id,
                            strength=min(1.0 + delta, 10.0),
                        )
                    total_strengthened += 1

        return total_strengthened

    # ── Temporal Substate Separation ────────────────────────────

    def _split_by_temporal_age(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split episodes into recent (< 2 days) and older (> 2 days).

        First pass: recent → cluster + strengthen.
        Second pass: old → integrate with existing clusters.
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(days=RECENT_CUTOFF_DAYS)
        cutoff_str = cutoff.isoformat()

        recent = self.db.fetchall(
            "SELECT * FROM episodes WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
            (cutoff_str, NREM_RECENT_LIMIT),
        )

        old = self.db.fetchall(
            "SELECT * FROM episodes WHERE timestamp < ? ORDER BY timestamp DESC LIMIT ?",
            (cutoff_str, NREM_OLD_LIMIT),
        )

        return recent, old

    def _cluster_episodes(
        self,
        episodes: list[dict[str, Any]],
        label: str = CLUSTER_RECENT_LABEL,
    ) -> list[list[dict[str, Any]]]:
        """Simple domain-based clustering for temporal pass.

        Groups episodes by domain. Each domain group is a cluster.
        Weak entities (< 2 mentions) in each cluster get merged.
        """
        domain_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ep in episodes:
            domain = ep.get("domain") or "general"
            domain_groups[domain].append(ep)

        clusters: list[list[dict[str, Any]]] = list(domain_groups.values())

        # Strengthen within each cluster: update cluster_id on episodes
        for cluster in clusters:
            if not cluster:
                continue
            cluster_id = _new_id()
            for ep in cluster:
                self.db.update("episodes", ep["id"], {"cluster_id": cluster_id})

        return clusters

    def _integrate_old_memories(self, old_eps: list[dict[str, Any]]) -> int:
        """Second temporal pass: integrate older memories with existing clusters.

        For each old episode, find an existing recent cluster by domain match
        and link it via relationship strengthening. If no cluster exists,
        create a new one for the old memory.
        """
        integrated = 0

        # Find existing clusters from recent pass
        existing_clusters = self.db.fetchall(
            "SELECT DISTINCT cluster_id FROM episodes "
            "WHERE cluster_id IS NOT NULL AND cluster_id != ''"
            "LIMIT 50"
        )

        cluster_domains: dict[str, str] = {}
        for row in existing_clusters:
            cid = row["cluster_id"]
            sample = self.db.fetchone(
                "SELECT domain FROM episodes WHERE cluster_id = ? LIMIT 1",
                (cid,),
            )
            if sample:
                cluster_domains[cid] = sample.get("domain") or "general"

        for ep in old_eps:
            ep_domain = ep.get("domain") or "general"
            content = ep.get("content_text", "") or ""
            action = ep.get("action", "") or ""

            # Find a matching cluster by domain
            matched_cluster = None
            for cid, cdomain in cluster_domains.items():
                if cdomain == ep_domain:
                    matched_cluster = cid
                    break

            if matched_cluster:
                # Link old memory entities to cluster entities
                raw_entities = extract_entities_from_text(content, action)
                for raw in raw_entities:
                    old_entity = self.entities.find_by_canonical(
                        raw["canonical_name"], raw["entity_type"]
                    )
                    if not old_entity:
                        continue
                    # Find cluster entities in the same domain
                    cluster_entities = self.db.fetchall(
                        "SELECT e.id FROM entities e "
                        "JOIN entity_mentions em ON em.entity_id = e.id "
                        "JOIN episodes ep2 ON ep2.id = em.source_id "
                        "WHERE ep2.cluster_id = ? AND e.id != ? "
                        "AND (e.domain = ? OR e.domain IS NULL) "
                        "LIMIT 5",
                        (matched_cluster, old_entity["id"], ep_domain),
                    )
                    for ce in cluster_entities:
                        existing = self.db.fetchone(
                            "SELECT id FROM relationships "
                            "WHERE ((source_entity_id = ? AND target_entity_id = ?) "
                            "   OR (source_entity_id = ? AND target_entity_id = ?)) "
                            "AND relation_type = ?",
                            (
                                old_entity["id"],
                                ce["id"],
                                ce["id"],
                                old_entity["id"],
                                RelationType.RELATED_TO.value,
                            ),
                        )
                        if not existing:
                            self.graph.add_relationship(
                                source_entity_id=old_entity["id"],
                                target_entity_id=ce["id"],
                                relation_type=RelationType.RELATED_TO.value,
                                episode_id=ep["id"],
                                strength=0.6,  # Integration gets moderate strength
                            )
                            integrated += 1

                # Assign old episode to the cluster
                self.db.update("episodes", ep["id"], {"cluster_id": matched_cluster})
            else:
                # No matching cluster: create one
                new_cluster_id = _new_id()
                self.db.update("episodes", ep["id"], {"cluster_id": new_cluster_id})
                integrated += 1  # Count as integrated into new cluster

        return integrated

    # ── Veridical Replay ────────────────────────────────────────

    def _veridical_replay(self) -> int:
        """Sample high-priority episodes and replay action sequences.

        Priority threshold: priority_score > 0.5.
        Each replay: find entities mentioned in the episode and strengthen
        their relationships (tools/services mentioned together).
        """
        replay_candidates = self.db.fetchall(
            "SELECT * FROM episodes "
            "WHERE priority_score > ? AND priority_score IS NOT NULL "
            "ORDER BY priority_score DESC "
            "LIMIT ?",
            (PRIORITY_REPLAY_THRESHOLD, REPLAY_LIMIT),
        )

        if not replay_candidates:
            # Fallback to importance_score if no priority scores exist
            replay_candidates = self.db.fetchall(
                "SELECT * FROM episodes "
                "WHERE importance_score > ? AND importance_score IS NOT NULL "
                "ORDER BY importance_score DESC "
                "LIMIT ?",
                (PRIORITY_REPLAY_THRESHOLD, REPLAY_LIMIT),
            )

        replayed = 0
        for ep in replay_candidates:
            content = ep.get("content_text", "") or ""
            action = ep.get("action", "") or ""
            ep_id = ep["id"]

            raw_entities = extract_entities_from_text(content, action)
            if len(raw_entities) < 2:
                continue  # Need at least 2 entities to strengthen a relationship

            # Resolve to entity IDs
            entity_ids: list[str] = []
            for raw in raw_entities:
                found = self.entities.find_by_canonical(raw["canonical_name"], raw["entity_type"])
                if found:
                    entity_ids.append(found["id"])

            # Strengthen all pairs (tools/services used together in this replay)
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
                        (src_id, tgt_id, tgt_id, src_id, RelationType.RELATED_TO.value),
                    )
                    if existing:
                        new_strength = min(
                            float(existing["strength"]) + HEBBIAN_ETA,
                            10.0,
                        )
                        episodes_list = json.loads(existing["evidence_episodes"] or "[]")
                        if ep_id not in episodes_list:
                            episodes_list.append(ep_id)
                        self.db.update(
                            "relationships",
                            existing["id"],
                            {
                                "strength": new_strength,
                                "evidence_count": int(existing["evidence_count"]) + 1,
                                "evidence_episodes": episodes_list,
                                "last_observed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            },
                        )
                    else:
                        self.graph.add_relationship(
                            source_entity_id=src_id,
                            target_entity_id=tgt_id,
                            relation_type=RelationType.RELATED_TO.value,
                            episode_id=ep_id,
                            strength=1.0 + HEBBIAN_ETA,
                        )

            # Increment replay count
            current_count = int(ep.get("replay_count", 0))
            self.db.update(
                "episodes",
                ep_id,
                {"replay_count": current_count + 1},
            )
            replayed += 1

        return replayed
