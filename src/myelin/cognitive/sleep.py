"""Sleep cycle cognitive process.

Inspired by hermes-cashew's sleep synthesis (9-phase consolidation,
7K nodes in ~4s) but deeply integrated with Myelin's knowledge graph,
entity extraction, and temporal reasoning.

Hermes-cashew runs: cross-linking, dedup, GC, permanence evaluation,
core memory promotion, and dream generation.

Our sleep cycle runs:
1. Entity extraction: batch-extract entities from unprocessed episodes
2. Relationship inference: learn typed edges from action sequences
3. Graph consolidation: merge weak entities, strengthen frequent edges
4. Temporal state updates: close stale states, detect transitions
5. Cross-domain linking: find entities that bridge domains
6. Staleness detection: flag entities/facts not seen in recent sessions
7. Stats: report what changed

Trigger: session end (alongside other cognitive processes) or manual.
"""

from __future__ import annotations

from typing import Any

from ..core.database import Database
from ..core.models import ProcessName
from ..knowledge.entities import (
    EntityStore,
    extract_entities_from_text,
    extract_relations_from_sequence,
)
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.temporal import TemporalIndex
from .base import CognitiveProcess


class SleepCycle(CognitiveProcess):
    """Batch consolidation process that builds the knowledge graph."""

    name = ProcessName.SLEEP

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
        results = {
            "entities_extracted": 0,
            "relationships_created": 0,
            "entities_merged": 0,
            "temporal_states_updated": 0,
            "cross_domain_links": 0,
            "stale_flagged": 0,
        }

        # 1. Entity extraction from unprocessed episodes
        unprocessed = self._get_unprocessed_episodes(limit=500)
        for ep in unprocessed:
            entity_ids = self.entities.process_episode(
                episode_id=ep["id"],
                content_text=ep.get("content_text", ""),
                action=ep.get("action", ""),
                action_type=ep.get("action_type", ""),
                domain=ep.get("domain"),
            )
            results["entities_extracted"] += len(entity_ids)
            self._mark_entity_processed(ep["id"])

        # 2. Relationship inference from session sequences
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

        # 3. Graph consolidation: merge weak duplicate entities
        results["entities_merged"] += self._merge_weak_entities()

        # 4. Temporal state updates from recent episodes
        results["temporal_states_updated"] += self._update_temporal_states(recent_episodes[:100])

        # 5. Cross-domain linking
        results["cross_domain_links"] += self._find_cross_domain_links()

        # 6. Staleness detection
        results["stale_flagged"] += self._flag_stale_entities()

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

    def _mark_entity_processed(self, episode_id: str) -> None:
        pass

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
            "deploy",
            "migrate",
            "update",
            "upgrade",
            "install",
            "configure",
            "fix",
            "break",
            "fail",
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
                for _d2 in domains[i + 1 :]:
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
