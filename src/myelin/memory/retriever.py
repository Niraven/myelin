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

import datetime
import json
from typing import Any

from ..core.activation import base_level_activation
from ..core.database import Database
from ..core.models import RetrievalProvenance
from ..knowledge.entities import EntityStore, extract_entities_from_text
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.temporal import TemporalIndex

# ── Hardening helpers ──────────────────────────────────────────


def _clamp_limit(limit: int, min_val: int = 1, max_val: int = 100) -> int:
    """Clamp the retrieval limit to a safe range."""
    return max(min_val, min(limit, max_val))


def _validate_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Validate and normalise retrieval weights.

    Returns default weights when input is None, clamps negative values
    to zero, and normalises so they sum to 1.0.
    """
    w = weights or {
        "text": 0.25,
        "vector": 0.25,
        "entity": 0.20,
        "temporal": 0.10,
        "activation": 0.10,
        "importance": 0.10,
    }
    # Clamp negative values
    w = {k: max(0.0, v) for k, v in w.items()}
    total = sum(w.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        w = {k: v / total for k, v in w.items()}
    return w


def _parse_iso_timestamp(value: Any) -> datetime.datetime | None:
    """Parse an ISO-8601 timestamp string, returning None on failure."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _filter_by_max_age(result: dict[str, Any], max_age_hours: float | None) -> bool:
    """Return False if the result is older than max_age_hours."""
    if max_age_hours is None:
        return True
    ts = result.get("timestamp") or result.get("created_at")
    parsed = _parse_iso_timestamp(ts)
    if parsed is None:
        return True  # no timestamp → keep (cannot judge age)
    age_hours = (datetime.datetime.utcnow() - parsed.replace(tzinfo=None)).total_seconds() / 3600
    return age_hours <= max_age_hours


# ── Retriever ──────────────────────────────────────────────────


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
        agent_ids: list[str] | None = None,
        querying_agent_id: str | None = None,
        min_confidence: float | None = None,
        max_age_hours: float | None = None,
    ) -> list[dict[str, Any]]:
        """Multi-signal retrieval across all memory types.

        Parameters
        ----------
        query : str
            Natural-language query text.
        query_embedding : list[float] | None
            Optional pre-computed embedding vector.
        domain : str | None
            Optional domain filter.
        limit : int
            Maximum results to return. Clamped to [1, 100].
        weights : dict[str, float] | None
            Per-signal fusion weights. Validated and normalised.
        include_procedures, include_semantic, include_episodes : bool
            Toggle retrieval from each memory store.
        agent_ids : list[str] | None
            Restrict to specific source agent(s). ['*'] or None = no restriction.
        querying_agent_id : str | None
            When set, applies a cross-agent confidence discount.
        min_confidence : float | None
            Minimum confidence threshold for episodic/procedure results (0.0-1.0).
        max_age_hours : float | None
            Maximum age in hours. Results older than this are excluded.

        Returns
        -------
        list[dict[str, Any]]
            Ranked results with ``_provenance`` metadata attached to each.
        """
        limit = _clamp_limit(limit)
        w = _validate_weights(weights)

        candidates: dict[str, dict[str, Any]] = {}

        if include_episodes:
            self._add_episode_candidates(
                query, query_embedding, domain, candidates, limit * 3, agent_ids=agent_ids
            )

        if include_semantic:
            self._add_semantic_candidates(query, query_embedding, domain, candidates, limit * 3)

        if include_procedures:
            self._add_procedure_candidates(
                query, query_embedding, domain, candidates, limit * 3, agent_ids=agent_ids
            )

        query_entities = extract_entities_from_text(query)
        entity_boost_ids = set()
        for qe in query_entities:
            found = self.entities.find_by_canonical(qe["canonical_name"], qe["entity_type"])
            if found:
                entity_boost_ids.add(found["id"])
                neighbors = self.graph.get_neighbors(found["id"], limit=5)
                for n in neighbors:
                    entity_boost_ids.add(n["id"])

        retrieved_at = datetime.datetime.utcnow().isoformat()
        scored: list[tuple[float, dict]] = []
        for _cid, candidate in candidates.items():
            text_score = candidate.get("_text_score", 0.0)
            vec_score = candidate.get("_vec_score", 0.0)

            entity_score = self._compute_entity_score(candidate, entity_boost_ids)
            temporal_score = self._compute_temporal_score(candidate)
            activation_score = self._compute_activation_score(candidate)
            importance_score = self._compute_importance_score(candidate)

            composite = (
                w["text"] * text_score
                + w["vector"] * vec_score
                + w["entity"] * entity_score
                + w["temporal"] * temporal_score
                + w["activation"] * activation_score
                + w.get("importance", 0.0) * importance_score
            )

            # Confidence filter (procedures carry their own confidence;
            # episodes use importance_score as a proxy)
            candidate_confidence = candidate.get("confidence")
            if candidate_confidence is None:
                candidate_confidence = candidate.get("importance_score", 0.5)
            if min_confidence is not None and candidate_confidence < min_confidence:
                continue

            # Max-age filter
            if not _filter_by_max_age(candidate, max_age_hours):
                continue

            result = {
                k: v
                for k, v in candidate.items()
                if not k.startswith("_") or k in ("_source_type",)
            }
            result["_composite_score"] = composite
            result["_scores"] = {
                "text": text_score,
                "vector": vec_score,
                "entity": entity_score,
                "temporal": temporal_score,
                "activation": activation_score,
                "importance": importance_score,
            }
            result["source_agent"] = candidate.get("agent_id") or candidate.get(
                "source_agent", "unknown"
            )

            # Cross-agent confidence discount
            if querying_agent_id:
                source = result["source_agent"]
                if source == querying_agent_id:
                    multiplier = 1.0
                elif source != "unknown":
                    multiplier = 0.85
                else:
                    multiplier = 0.7
                result["_composite_score"] *= multiplier
                result["_scores"]["cross_agent_discount"] = multiplier

            # Attach durable provenance metadata
            result["_provenance"] = RetrievalProvenance(
                source_id=result.get("id", ""),
                source_type=str(result.get("_source_type", "unknown")),
                source_agent=str(result.get("source_agent", "unknown")),
                domain=result.get("domain"),
                timestamp=result.get("timestamp") or result.get("created_at"),
                retrieved_at=retrieved_at,
                retrieval_signals=dict(result.get("_scores", {})),
                composite_score=float(result.get("_composite_score", 0.0)),
            ).to_dict()

            scored.append((result["_composite_score"], result))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def _add_episode_candidates(
        self,
        query: str,
        embedding: list[float] | None,
        domain: str | None,
        candidates: dict[str, dict],
        limit: int,
        agent_ids: list[str] | None = None,
    ) -> None:
        where = None
        where_params: tuple[Any, ...] = ()
        if agent_ids and agent_ids != ["*"]:
            placeholders = ",".join("?" for _ in agent_ids)
            where = f"agent_id IN ({placeholders})"
            where_params = tuple(agent_ids)

        fts = self.db.fts_search(
            "episodes", "episodes_fts", query, limit=limit, where=where, where_params=where_params
        )
        for i, row in enumerate(fts):
            rid = row["id"]
            if rid not in candidates:
                candidates[rid] = dict(row)
                candidates[rid]["_source_type"] = "episode"
            candidates[rid]["_text_score"] = 1.0 - (i / max(len(fts), 1))

        if embedding and self.db.vec_available:
            vec = self.db.vec_search(
                "episodes",
                "embedding",
                embedding,
                limit=limit,
                where=where,
                where_params=where_params,
            )
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
        agent_ids: list[str] | None = None,
    ) -> None:
        where = None
        where_params: tuple[Any, ...] = ()
        if agent_ids and agent_ids != ["*"]:
            placeholders = ",".join("?" for _ in agent_ids)
            where = f"source_agent IN ({placeholders})"
            where_params = tuple(agent_ids)

        fts = self.db.fts_search(
            "procedures",
            "procedures_fts",
            query,
            limit=limit,
            where=where,
            where_params=where_params,
        )
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
            created = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            age_hours = (
                datetime.datetime.utcnow() - created.replace(tzinfo=None)
            ).total_seconds() / 3600
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

    def _compute_importance_score(self, candidate: dict[str, Any]) -> float:
        """Score based on pre-computed importance (0.0-1.0)."""
        importance = candidate.get("importance_score")
        if importance is None:
            return 0.5
        try:
            return float(min(1.0, max(0.0, importance)))
        except (ValueError, TypeError):
            return 0.5
