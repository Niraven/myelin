"""Consolidator: merge similar episodes into semantic summaries.

Inspired by sleep consolidation in neuroscience.
Trigger: every 50 writes or session end.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import NodeType, ProcessName, SemanticNode, SourceType
from ..memory.episodic import EpisodicMemory
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess

CONSOLIDATION_BATCH = 50


class Consolidator(CognitiveProcess):
    name = ProcessName.CONSOLIDATOR

    def __init__(self, db: Database, episodic: EpisodicMemory, semantic: SemanticMemory):
        super().__init__(db)
        self.episodic = episodic
        self.semantic = semantic

    def should_run(self) -> bool:
        unconsolidated = self.episodic.get_unconsolidated(limit=1)
        count = self.episodic.db.fetchone(
            "SELECT COUNT(*) as cnt FROM episodes WHERE consolidated = 0"
        )
        return (count["cnt"] if count else 0) >= CONSOLIDATION_BATCH

    async def execute(self) -> dict[str, Any]:
        episodes = self.episodic.get_unconsolidated(limit=CONSOLIDATION_BATCH)
        if not episodes:
            return {"processed": 0, "created": 0}

        clusters = self._cluster_by_domain(episodes)
        created = 0

        for domain, group in clusters.items():
            cluster_id = uuid4().hex[:16]
            episode_ids = [ep["id"] for ep in group]

            summary = self._summarize_cluster(group)
            node = SemanticNode(
                node_type=NodeType.FACT,
                content=summary,
                source_type=SourceType.OBSERVATION,
                source_ids=episode_ids,
                domain=domain,
                confidence=0.6,
            )
            self.semantic.store(node)
            self.episodic.mark_consolidated(episode_ids, cluster_id)
            created += 1

        return {"processed": len(episodes), "created": created}

    def _cluster_by_domain(
        self, episodes: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        clusters: dict[str, list] = {}
        for ep in episodes:
            domain = ep.get("domain") or "general"
            clusters.setdefault(domain, []).append(ep)
        return clusters

    def _summarize_cluster(self, episodes: list[dict[str, Any]]) -> str:
        """Basic extractive summary. Phase 1 will add LLM-based summarization."""
        actions = [ep["action"] for ep in episodes]
        domains = set(ep.get("domain", "unknown") for ep in episodes)
        success_rate = sum(1 for ep in episodes if ep.get("success")) / len(episodes)

        return (
            f"Cluster of {len(episodes)} episodes in {', '.join(domains)}. "
            f"Actions: {'; '.join(actions[:5])}{'...' if len(actions) > 5 else ''}. "
            f"Success rate: {success_rate:.0%}."
        )
