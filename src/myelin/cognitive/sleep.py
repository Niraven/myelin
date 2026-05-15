"""Sleep cycle — two-phase NREM + REM orchestration.

Replaces the single-phase batch consolidation with a proper two-phase sleep
system inspired by neuroscience (Frankland & Bontempi 2005, Nature 2025):

PHASE 1 — NREM (non-rapid eye movement):
  - Hebbian strengthening: Δs = η * I(c_i) * I(c_j) for co-occurring pairs
  - Synaptic downscaling (SHY): s ← 0.85 * s on all relationships
  - Temporal substate separation: recent (<2 days) cluster + strengthen,
    older (>2 days) integrate with existing clusters
  - Veridical replay: high-priority episodes replayed through graph

PHASE 2 — REM (rapid eye movement):
  - Random walk dreaming: BFS random walks through knowledge graph,
    creating weak 'dreamed_connection' edges
  - Counterfactual generation: "what if" alternatives for failed episodes
  - Novel connection discovery: cross-domain linking via shared attributes
  - TAG importance-weighted replay selection for next cycle

Trigger: session end (alongside other cognitive processes) or manual.
"""

from __future__ import annotations

from typing import Any

from ..core.database import Database
from ..core.models import ProcessName
from ..knowledge.entities import (
    EntityStore,
    HybridEntityExtractor,
    extract_entities_from_text,
    extract_relations_from_sequence,
)
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.temporal import TemporalIndex
from .base import CognitiveProcess
from .importance import ImportanceWeights, score_clusters
from .nrem_sleep import NREMPhase
from .rem_sleep import REMPhase


class ImportanceComputer:
    """Batch-compute per-episode importance scores from cluster stats."""

    def compute(
        self,
        db: Database,
        episodes: list[dict[str, Any]],
        weights: dict[str, float] | ImportanceWeights | None = None,
    ) -> dict[str, float]:
        """Compute importance for each episode based on its cluster stats.

        Returns a mapping of episode_id -> importance_score.
        """
        if not episodes:
            return {}

        cluster_scores = score_clusters(episodes, weights=weights, use_normalized_default=True)
        episode_scores: dict[str, float] = {}
        for ep in episodes:
            cid = ep.get("cluster_id") or ep.get("cluster")
            if cid and cid in cluster_scores:
                episode_scores[ep["id"]] = float(cluster_scores[cid])
            else:
                episode_scores[ep["id"]] = 0.5
        return episode_scores

    def persist(self, db: Database, episode_scores: dict[str, float]) -> int:
        """Write computed scores back to the episodes table."""
        updated = 0
        for episode_id, score in episode_scores.items():
            db.update("episodes", episode_id, {"importance_score": score})
            updated += 1
        return updated


class SleepCycle(CognitiveProcess):
    """Two-phase sleep orchestrator: NREM + REM consolidation.

    Runs NREM phase first (strengthening, downscaling, replay),
    then REM phase (dreaming, counterfactuals, novel connections).

    Also retains the original graph consolidation steps for backward
    compatibility (entity extraction, relationship inference, temporal
    updates, cross-domain linking, staleness detection).
    """

    name = ProcessName.SLEEP

    def __init__(
        self,
        db: Database,
        entity_store: EntityStore | None = None,
        graph: KnowledgeGraph | None = None,
        temporal: TemporalIndex | None = None,
        hybrid_extractor: HybridEntityExtractor | None = None,
    ):
        super().__init__(db)
        self.entities = entity_store or EntityStore(db)
        self.graph = graph or KnowledgeGraph(db)
        self.temporal = temporal or TemporalIndex(db)
        self.hybrid_extractor = hybrid_extractor

        # Create phase modules (share same dependencies)
        self.nrem = NREMPhase(db, self.entities, self.graph, self.temporal)
        self.rem = REMPhase(db, self.entities, self.graph, self.temporal)

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "entities_extracted": 0,
            "relationships_created": 0,
            "entities_merged": 0,
            "temporal_states_updated": 0,
            "cross_domain_links": 0,
            "stale_flagged": 0,
            "importance_scores_updated": 0,
            # Two-phase sleep results
            "nrem": {},
            "rem": {},
        }

        # ── Pre-sleep: legacy graph maintenance ────────────────
        # LLM-based concept extraction during sleep (gated by --llm-extraction)
        if self.hybrid_extractor:
            unprocessed = self._get_unprocessed_episodes()
            if unprocessed:
                candidates = self.hybrid_extractor.extract_concepts(
                    [ep["content_text"] for ep in unprocessed]
                )
                for candidate in candidates:
                    entity_id = self.entities.upsert_entity(
                        name=candidate["name"],
                        entity_type=candidate.get("entity_type", "concept"),
                        canonical_name=candidate.get(
                            "canonical_name", candidate["name"].lower().strip()
                        ),
                    )
                    candidate_lower = candidate["name"].lower()
                    for ep in unprocessed:
                        if candidate_lower in ep.get("content_text", "").lower():
                            self.entities.add_mention(
                                entity_id=entity_id,
                                source_type="episode",
                                source_id=ep["id"],
                                context_snippet=ep["content_text"][:200],
                            )
                    results["entities_extracted"] += 1

        # Relationship inference from session sequences
        recent_episodes = self.db.fetchall(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 1000"
        )
        if recent_episodes:
            raw_relations = extract_relations_from_sequence(recent_episodes)
            for rel in raw_relations:
                source_entity = self.entities.find_by_canonical(rel["source"], "tool")
                if not source_entity:
                    source_entity = self.entities.find_by_canonical(rel["source"], "service")
                target_entity = self.entities.find_by_canonical(rel["target"], "tool")
                if not target_entity:
                    target_entity = self.entities.find_by_canonical(rel["target"], "service")

                if source_entity and target_entity:
                    self.graph.add_relationship(
                        source_entity_id=source_entity["id"],
                        target_entity_id=target_entity["id"],
                        relation_type=rel["relation_type"],
                    )
                    results["relationships_created"] += 1

        # Graph consolidation: merge weak duplicate entities
        results["entities_merged"] += self._merge_weak_entities()

        # Temporal state updates from recent episodes
        results["temporal_states_updated"] += self._update_temporal_states(
            recent_episodes[:100]
        )

        # Cross-domain linking (legacy approach)
        results["cross_domain_links"] += self._find_cross_domain_links()

        # Staleness detection
        results["stale_flagged"] += self._flag_stale_entities()

        # Importance scoring
        importance_computer = ImportanceComputer()
        episode_scores = importance_computer.compute(
            self.db,
            recent_episodes,
            weights=ImportanceWeights(frequency=0.4, consequence=0.4, recency=0.2),
        )
        results["importance_scores_updated"] = importance_computer.persist(
            self.db, episode_scores
        )

        # ── PHASE 1: NREM Sleep ───────────────────────────────
        results["nrem"] = await self.nrem.execute()

        # ── PHASE 2: REM Sleep ────────────────────────────────
        results["rem"] = await self.rem.execute()

        return results

    def _get_unprocessed_episodes(self, limit: int = 500) -> list[dict[str, Any]]:
        """Get episodes that haven't had entity extraction yet."""
        return self.db.fetchall(
            """
            SELECT e.* FROM episodes e
            LEFT JOIN entity_mentions em ON em.source_id = e.id AND em.source_type = 'episode'
            WHERE em.id IS NULL
            ORDER BY e.timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

    def _merge_weak_entities(self, min_mentions: int = 1) -> int:
        """Merge entities that appear only once and have similar names."""
        weak = self.db.fetchall(
            "SELECT * FROM entities WHERE mention_count <= ? ORDER BY canonical_name",
            (min_mentions,),
        )
        merged = 0
        seen_canonicals: dict[str, str] = {}

        for entity in weak:
            canonical = entity["canonical_name"].lower().strip()
            if canonical in seen_canonicals and seen_canonicals[canonical] != entity["id"]:
                self.db.execute(
                    "UPDATE entity_mentions SET entity_id = ? WHERE entity_id = ?",
                    (seen_canonicals[canonical], entity["id"]),
                )
                self.db.delete("entities", entity["id"])
                merged += 1
            else:
                seen_canonicals[canonical] = entity["id"]

        if merged:
            self.db.commit()
        return merged

    def _update_temporal_states(self, episodes: list[dict[str, Any]]) -> int:
        """Create temporal states from episodes that indicate state changes."""
        updated = 0
        state_keywords = {
            "deploy", "migrate", "update", "upgrade",
            "install", "configure", "fix", "break", "fail",
        }

        for ep in episodes:
            action = ep.get("action", "").lower()
            content = ep.get("content_text", "").lower()
            combined = f"{action} {content}"

            if any(kw in combined for kw in state_keywords):
                entities = extract_entities_from_text(
                    ep.get("content_text", ""),
                    ep.get("action", ""),
                )
                for ent_data in entities:
                    found = self.entities.find_by_canonical(
                        ent_data["canonical_name"],
                        ent_data["entity_type"],
                    )
                    if found:
                        description = f"{ep.get('action', 'unknown action')} on {ent_data['name']}"
                        if not ep.get("success", True):
                            description += " (FAILED)"
                        self.temporal.record_state(
                            state_description=description,
                            entity_id=found["id"],
                            source_episode_id=ep["id"],
                            domain=ep.get("domain"),
                            confidence=0.7 if ep.get("success", True) else 0.3,
                        )
                        updated += 1

        return updated

    def _find_cross_domain_links(self) -> int:
        """Find entities that appear in multiple domains and link them."""
        cross_domain = self.db.fetchall(
            """
            SELECT e.id, e.canonical_name, GROUP_CONCAT(DISTINCT em_ep.domain) as domains,
                   COUNT(DISTINCT em_ep.domain) as domain_count
            FROM entities e
            JOIN entity_mentions em ON em.entity_id = e.id AND em.source_type = 'episode'
            JOIN episodes em_ep ON em_ep.id = em.source_id
            WHERE em_ep.domain IS NOT NULL
            GROUP BY e.id
            HAVING domain_count > 1
            """
        )

        links_created = 0
        for entity in cross_domain:
            domains = entity["domains"].split(",") if entity["domains"] else []
            for i, d1 in enumerate(domains):
                for _d2 in domains[i + 1:]:
                    domain_entities_1 = self.db.fetchall(
                        "SELECT id FROM entities WHERE domain = ? AND id != ? LIMIT 5",
                        (d1, entity["id"]),
                    )
                    for de1 in domain_entities_1:
                        existing = self.db.fetchone(
                            "SELECT id FROM relationships "
                            "WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = 'related_to'",
                            (entity["id"], de1["id"]),
                        )
                        if not existing:
                            self.graph.add_relationship(
                                source_entity_id=entity["id"],
                                target_entity_id=de1["id"],
                                relation_type="related_to",
                                domain=d1,
                            )
                            links_created += 1
        return links_created

    def _flag_stale_entities(self, stale_days: int = 14) -> int:
        """Count entities not seen in recent episodes."""
        from datetime import datetime, timedelta

        cutoff = (datetime.utcnow() - timedelta(days=stale_days)).isoformat()
        row = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM entities WHERE last_seen < ?",
            (cutoff,),
        )
        return row["cnt"] if row else 0
