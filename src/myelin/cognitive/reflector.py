"""Reflector: generate higher-order insights from episode clusters.

Inspired by Stanford Generative Agents (Park et al., 2023).
Trigger: session end.

Observations -> Reflections -> Higher-order Reflections
"Nino ran npm test 5x" -> "Nino always tests before deploying"
  -> "Nino is cautious about breaking production"
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, DefaultDict, Dict, Iterable, Mapping, MutableMapping

from ..core.database import Database
from ..core.models import NodeType, ProcessName, SemanticNode, SourceType
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess


# ============================================================================
# Reflector cognitive process
# ============================================================================

class Reflector(CognitiveProcess):
    name = ProcessName.REFLECTOR

    def __init__(self, db: Database, semantic: SemanticMemory):
        super().__init__(db)
        self.semantic = semantic

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        """Generate reflections from recent semantic nodes.

        Phase 0: pattern-based reflection (no LLM).
        Phase 1: LLM-powered insight generation.
        """
        recent_facts = self.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE node_type = ? "
            "AND created_at > datetime('now', '-24 hours') "
            "ORDER BY created_at DESC LIMIT 20",
            (NodeType.FACT.value,),
        )

        if len(recent_facts) < 3:
            return {"processed": 0, "created": 0}

        domain_groups: dict[str, list] = {}
        for fact in recent_facts:
            d = fact.get("domain") or "general"
            domain_groups.setdefault(d, []).append(fact)

        created = 0
        for domain, facts in domain_groups.items():
            if len(facts) < 2:
                continue

            reflection_text = self._generate_reflection(domain, facts)
            source_ids = [f["id"] for f in facts]

            node = SemanticNode(
                node_type=NodeType.REFLECTION,
                content=reflection_text,
                source_type=SourceType.REFLECTION,
                source_ids=source_ids,
                domain=domain,
                confidence=0.5,
            )
            self.semantic.store(node)
            created += 1

        return {"processed": len(recent_facts), "created": created}

    def _generate_reflection(self, domain: str, facts: list[dict]) -> str:
        """Pattern-based reflection. Will be replaced by LLM call in Phase 1."""
        contents = [f["content"] for f in facts]
        return (
            f"Reflection on {domain} ({len(facts)} observations): "
            f"Pattern observed across: {'; '.join(c[:80] for c in contents[:3])}. "
            f"This suggests a consistent behavior in the {domain} domain."
        )


# ============================================================================
# Episode reflector utilities for importance scoring
#
# Provides deterministic cluster-level metrics extracted from a collection of
# episode records used by ``importance.py``.
# ============================================================================

Episode = Mapping[str, Any]


@dataclass(frozen=True)
class ClusterStats:
    """Pre-aggregated metrics for one cluster."""

    cluster_id: str
    frequency: int
    success_rate: float
    last_seen_ts: float


def _as_iso_timestamp(value: Any) -> float | None:
    """Return a unix timestamp from a datetime, int/float, or ISO-8601 string."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Support a common RFC3339 / ISO flavor used in payloads.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None

    return None


def _cluster_id(episode: Episode) -> str | None:
    """Resolve cluster field from multiple common key aliases."""

    return (
        episode.get("cluster")
        or episode.get("cluster_id")
        or episode.get("topic")
        or episode.get("clusterId")
    )


def _to_success(value: Any) -> float:
    """Parse success-like values into a 0.0..1.0 score."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        if 0 <= float(value) <= 1:
            return float(value)
        if value > 1:
            # Backward-compatible fallback for count-style signals.
            return 1.0 if value > 0 else 0.0
        return 0.0

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "1", "yes", "y", "success", "succeeded", "pass", "passed"}:
            return 1.0
        if text in {"false", "f", "0", "no", "n", "failure", "failed", "fail"}:
            return 0.0

    return 0.0


def build_cluster_stats(episodes: Iterable[Episode]) -> Dict[str, ClusterStats]:
    """Aggregate episodes into per-cluster frequency/success/recency stats.

    Parameters
    ----------
    episodes:
        Iterable of episode dictionaries.

    Required keys per episode
    ------------------------
    * ``cluster`` / ``cluster_id`` / ``topic`` / ``clusterId``

    Optional keys
    -------------
    * ``success`` / ``success_rate`` / ``succeeded`` / ``outcome``
    * ``timestamp`` / ``ts`` / ``seen_at`` / ``created_at``
    """

    raw: DefaultDict[str, list[float]] = defaultdict(list)
    success_counts: MutableMapping[str, float] = defaultdict(float)
    seen: MutableMapping[str, float] = {}

    for episode in episodes:
        cid = _cluster_id(episode)
        if not cid:
            continue

        raw[cid].append(1.0)

        success = episode.get("success")
        if success is None:
            # Keep compatibility with alternate source fields.
            success = episode.get("success_rate", episode.get("succeeded", episode.get("outcome")))
        success_counts[cid] += _to_success(success)

        for key in ("timestamp", "ts", "seen_at", "created_at", "observed_at"):
            ts = _as_iso_timestamp(episode.get(key))
            if ts is not None:
                seen[cid] = max(seen.get(cid, ts), ts)
                break

    cluster_stats: Dict[str, ClusterStats] = {}
    for cluster_id, occurrences in raw.items():
        frequency = len(occurrences)
        success_rate = success_counts[cluster_id] / frequency if frequency else 0.0
        cluster_last_seen = seen.get(cluster_id)
        if cluster_last_seen is None:
            # Deterministic fallback keeps scoring stable across missing timestamps.
            cluster_last_seen = 0.0
        cluster_stats[cluster_id] = ClusterStats(
            cluster_id=cluster_id,
            frequency=frequency,
            success_rate=success_rate,
            last_seen_ts=cluster_last_seen,
        )

    return cluster_stats
