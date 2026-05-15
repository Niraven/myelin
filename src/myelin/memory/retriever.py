"""Multi-signal hybrid retriever.

Fuses 5 retrieval signals into a single ranked result set:
1. FTS5 text search (keyword matching)
2. Vector similarity (semantic matching)
3. Entity graph boost (entities mentioned in query boost connected memories)
4. Temporal recency (recent states weighted higher)
5. ACT-R activation (frequently accessed memories rank higher)

mem0 uses "multi-signal retrieval" (semantic + BM25 + entity). We add
temporal reasoning and ACT-R activation on top of that. Supermemory
claims <300ms for their retrieval. Ours is SQLite-native, so it's even
faster for local deployments.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.activation import base_level_activation
from ..core.database import Database
from ..knowledge.entities import EntityStore, extract_entities_from_text
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.temporal import TemporalIndex


class MultiSignalRetriever:
    """Retrieves memories using fused signals from all memory subsystems."""

    def __init__(
        self,
        db: Database,
        entity_store: EntityStore | None = None,
        graph: KnowledgeGraph | None = None,
        temporal: TemporalIndex | None = None,
    ):
        self.db = db
        self.entities = entity_store or EntityStore(db)
        self.graph = graph or KnowledgeGraph(db)
        self.temporal = temporal or TemporalIndex(db)

    def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        domain: str | None = None,
        limit: int = 10,
        weights: dict[str, float] | None = None,
        include_procedures: bool = True,
        include_semantic: bool = True,
        include_episodes: bool = True,
    ) -> list[dict[str, Any]]:
        """Multi-signal retrieval across all memory types.

        Returns ranked results with source type and composite score.
        """
        w = weights or {
            "text": 0.25,
            "vector": 0.25,
            "entity": 0.20,
            "temporal": 0.15,
            "activation": 0.15,
        }

        candidates: dict[str, dict[str, Any]] = {}

        if include_episodes:
            self._add_episode_candidates(query, query_embedding, domain, candidates, limit * 3)

        if include_semantic:
            self._add_semantic_candidates(query, query_embedding, domain, candidates, limit * 3)

        if include_procedures:
            self._add_procedure_candidates(query, query_embedding, domain, candidates, limit * 3)

        query_entities = extract_entities_from_text(query)
        entity_boost_ids = set()
        for qe in query_entities:
            found = self.entities.find_by_canonical(qe["canonical_name"], qe["entity_type"])
            if found:
                entity_boost_ids.add(found["id"])
                neighbors = self.graph.get_neighbors(found["id"], limit=5)
                for n in neighbors:
                    entity_boost_ids.add(n["id"])

        scored: list[tuple[float, dict]] = []
        for _cid, candidate in candidates.items():
            text_score = candidate.get("_text_score", 0.0)
            vec_score = candidate.get("_vec_score", 0.0)

            entity_score = self._compute_entity_score(candidate, entity_boost_ids)
            temporal_score = self._compute_temporal_score(candidate)
            activation_score = self._compute_activation_score(candidate)

            composite = (
                w["text"] * text_score
                + w["vector"] * vec_score
                + w["entity"] * entity_score
                + w["temporal"] * temporal_score
                + w["activation"] * activation_score
            )

            result = {k: v for k, v in candidate.items() if not k.startswith("_") or k in ("_source_type",)}
            result["_composite_score"] = composite
            result["_scores"] = {
                "text": text_score,
                "vector": vec_score,
                "entity": entity_score,
                "temporal": temporal_score,
                "activation": activation_score,
            }
            scored.append((composite, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def _add_episode_candidates(
        self,
        query: str,
        embedding: list[float] | None,
        domain: str | None,
        candidates: dict[str, dict],
        limit: int,
    ) -> None:
        fts = self.db.fts_search("episodes", "episodes_fts", query, limit=limit)
        for i, row in enumerate(fts):
            rid = row["id"]
            if rid not in candidates:
                candidates[rid] = dict(row)
                candidates[rid]["_source_type"] = "episode"
            candidates[rid]["_text_score"] = 1.0 - (i / max(len(fts), 1))

        if embedding and self.db.vec_available:
            vec = self.db.vec_search("episodes", "embedding", embedding, limit=limit)
            for i, row in enumerate(vec):
                rid = row["id"]
                if rid not in candidates:
                    candidates[rid] = dict(row)
                    candidates[rid]["_source_type"] = "episode"
                candidates[rid]["_vec_score"] = 1.0 - (i / max(len(vec), 1))

    def _add_semantic_candidates(
        self,
        query: str,
        embedding: list[float] | None,
        domain: str | None,
        candidates: dict[str, dict],
        limit: int,
    ) -> None:
        fts = self.db.fts_search("semantic_nodes", "semantic_fts", query, limit=limit)
        for i, row in enumerate(fts):
            rid = row["id"]
            if rid not in candidates:
                candidates[rid] = dict(row)
                candidates[rid]["_source_type"] = "semantic"
            candidates[rid]["_text_score"] = 1.0 - (i / max(len(fts), 1))

    def _add_procedure_candidates(
        self,
        query: str,
        embedding: list[float] | None,
        domain: str | None,
        candidates: dict[str, dict],
        limit: int,
    ) -> None:
        fts = self.db.fts_search("procedures", "procedures_fts", query, limit=limit)
        for i, row in enumerate(fts):
            rid = row["id"]
            if rid not in candidates:
                candidates[rid] = dict(row)
                candidates[rid]["_source_type"] = "procedure"
            candidates[rid]["_text_score"] = 1.0 - (i / max(len(fts), 1))

    def _compute_entity_score(
        self,
        candidate: dict[str, Any],
        boosted_entity_ids: set[str],
    ) -> float:
        """Boost candidates that mention entities from the query."""
        if not boosted_entity_ids:
            return 0.0

        candidate_id = candidate.get("id", "")
        source_type = candidate.get("_source_type", "episode")

        mentions = self.db.fetchall(
            "SELECT entity_id FROM entity_mentions WHERE source_type = ? AND source_id = ?",
            (source_type, candidate_id),
        )

        if not mentions:
            return 0.0

        mention_ids = {m["entity_id"] for m in mentions}
        overlap = mention_ids & boosted_entity_ids
        return len(overlap) / max(len(boosted_entity_ids), 1)

    def _compute_temporal_score(self, candidate: dict[str, Any]) -> float:
        """Score based on recency. More recent = higher score."""
        timestamp = (
            candidate.get("timestamp") or candidate.get("created_at") or candidate.get("updated_at")
        )
        if not timestamp:
            return 0.5

        try:
            from datetime import datetime

            created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            age_hours = (datetime.utcnow() - created.replace(tzinfo=None)).total_seconds() / 3600
            return 1.0 / (1.0 + age_hours / 168.0)
        except (ValueError, TypeError):
            return 0.5

    def _compute_activation_score(self, candidate: dict[str, Any]) -> float:
        """Score based on ACT-R base-level activation."""
        access_times_raw = candidate.get("access_times", "[]")
        if isinstance(access_times_raw, str):
            try:
                access_times = json.loads(access_times_raw)
            except (json.JSONDecodeError, TypeError):
                access_times = []
        elif isinstance(access_times_raw, list):
            access_times = access_times_raw
        else:
            access_times = []

        if not access_times:
            return 0.0

        activation = base_level_activation(access_times)
        return min(1.0, max(0.0, (activation + 2.0) / 4.0))
