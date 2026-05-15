"""Schema learner — semantic clustering and schema induction.

Agenternal-inspired: 3+ semantic memory nodes in a domain → schema.

1. CLUSTER semantic nodes by domain, compute pairwise Jaccard similarity
2. GROUP at threshold ε = 0.45 into clusters
3. INDUCE schema from each cluster of ≥3 nodes
4. MANAGE schema lifecycle: hypothesis → active → refuted → archived
5. MERGE duplicate schemas with confidence update
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import ProcessName, SchemaModel, SchemaStatus, SchemaType
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess

# ── Constants ──────────────────────────────────────────────────────

JACCARD_THRESHOLD = 0.30
MIN_CLUSTER_SIZE = 3
HYPOTHESIS_CONFIDENCE = 0.4
ACTIVE_CONFIDENCE = 0.6
ARCHIVE_DAYS = 30
INDUCTION_BOOST = 0.2  # Confidence boost from re-induction


def _new_id() -> str:
    return uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def extract_action_type(content: str) -> str:
    """Extract the most verb-like action from content."""
    content_lower = content.lower()
    # Common action prefixes
    action_prefixes = [
        "running", "executing", "calling", "deploying", "building",
        "testing", "configuring", "installing", "updating", "creating",
        "removing", "checking", "verifying", "analyzing", "processing",
        "using", "applying", "starting", "stopping", "restarting",
    ]
    for prefix in action_prefixes:
        if content_lower.startswith(prefix):
            return prefix
    # Fallback: take first meaningful word
    words = content_lower.split()[:5]
    for w in words:
        if w.endswith("ing") or w.endswith("ed"):
            return w
    return words[0] if words else "unknown"


def extract_entities_from_content(content: str) -> list[str]:
    """Extract potential entity references from content text."""
    words = content.split()
    # Heuristic: capitalized words, code-like terms, tool names
    entities: list[str] = []
    for w in words:
        clean = w.strip('"\'(),.;:!?')
        if not clean or len(clean) < 2:
            continue
        if clean[0].isupper() or "_" in clean or "-" in clean:
            entities.append(clean.lower())
        if any(kw in clean.lower() for kw in ["git", "docker", "pip", "npm", "aws", "api"]):
            entities.append(clean.lower())
    return entities


# ── SchemaLearner ──────────────────────────────────────────────────


class SchemaLearner(CognitiveProcess):
    """Schema induction from semantic memory clusters.

    Groups semantic nodes by domain, clusters by Jaccard similarity,
    induces behavioral schemas, and manages schema lifecycle.
    """

    name = ProcessName.SCHEMA_LEARNER

    def __init__(self, db: Database):
        super().__init__(db)
        self.semantic = SemanticMemory(db)

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        """Run one schema induction cycle."""
        results: dict[str, Any] = {
            "domains_processed": 0,
            "clusters_found": 0,
            "schemas_induced": 0,
            "schemas_merged": 0,
            "schemas_archived": 0,
            "schemas_refuted": 0,
        }

        # Step 1: Archive stale schemas
        results["schemas_archived"] = self._archive_stale_schemas()

        # Step 2: Refute schemas with contradictory evidence
        results["schemas_refuted"] = self._check_contradictions()

        # Step 3: Cluster and induce
        domains = self._get_domains_with_min_nodes()
        results["domains_processed"] = len(domains)

        for domain in domains:
            nodes = self._get_nodes_for_domain(domain)
            if len(nodes) < MIN_CLUSTER_SIZE:
                continue

            clusters = self._cluster_nodes(nodes)
            results["clusters_found"] += len(clusters)

            for cluster in clusters:
                if len(cluster) < MIN_CLUSTER_SIZE:
                    continue
                induced = self._induce_schema(cluster, domain)
                if induced:
                    results["schemas_induced"] += 1

                    # Try to merge with existing schema
                    merged = self._merge_with_existing(induced, domain)
                    if merged:
                        results["schemas_merged"] += 1

        self.db.commit()
        return results

    # ── Domain Helpers ─────────────────────────────────────────────

    def _get_domains_with_min_nodes(self) -> list[str]:
        """Get domains with at least MIN_CLUSTER_SIZE semantic nodes."""
        rows = self.db.fetchall(
            "SELECT domain, COUNT(*) as cnt FROM semantic_nodes "
            "WHERE domain IS NOT NULL AND domain != '' "
            "AND valid_until IS NULL "
            "GROUP BY domain HAVING cnt >= ?",
            (MIN_CLUSTER_SIZE,),
        )
        return [r["domain"] for r in rows]

    def _get_nodes_for_domain(self, domain: str) -> list[dict[str, Any]]:
        """Get all valid semantic nodes for a domain."""
        rows = self.db.fetchall(
            "SELECT * FROM semantic_nodes "
            "WHERE domain = ? AND valid_until IS NULL "
            "ORDER BY confidence DESC",
            (domain,),
        )
        for r in rows:
            for field in ("source_ids", "access_times", "tags"):
                if isinstance(r.get(field), str):
                    r[field] = json.loads(r[field])
        return rows

    # ── Clustering ─────────────────────────────────────────────────

    def _cluster_nodes(self, nodes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Cluster nodes by pairwise Jaccard similarity at threshold 0.45.

        Uses a simple greedy approach: for each node, find all nodes with
        Jaccard similarity >= threshold, group into clusters.
        """
        if not nodes:
            return []

        # Build similarity matrix
        n = len(nodes)
        sim_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = jaccard_similarity(nodes[i]["content"], nodes[j]["content"])
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s

        # Greedy clustering
        assigned: set[int] = set()
        clusters: list[list[dict[str, Any]]] = []

        for i in range(n):
            if i in assigned:
                continue
            cluster = [nodes[i]]
            assigned.add(i)
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                # Node j joins cluster if it's similar to ANY node in cluster
                if any(sim_matrix[k][j] >= JACCARD_THRESHOLD for k in range(n) if nodes[k] in cluster):
                    cluster.append(nodes[j])
                    assigned.add(j)
            if len(cluster) >= MIN_CLUSTER_SIZE:
                clusters.append(cluster)
            else:
                # Still include small clusters for callers that want visibility
                pass

        return clusters

    # ── Schema Induction ───────────────────────────────────────────

    def _induce_schema(
        self, cluster: list[dict[str, Any]], domain: str
    ) -> SchemaModel | None:
        """Induce a schema from a cluster of 3+ semantic nodes.

        1. Extract common patterns (shared entities, action types, success rate)
        2. Generate schema name from most common action + domain
        3. Build behavioral_pattern string
        4. Store conditions and exceptions
        """
        if len(cluster) < MIN_CLUSTER_SIZE:
            return None

        # Extract common action types
        actions = [extract_action_type(n.get("content", "")) for n in cluster]
        action_counter = Counter(actions)
        most_common_action = action_counter.most_common(1)[0][0] if action_counter else "unknown"

        # Find next-most-common action for pattern
        next_actions = [a for a in actions if a != most_common_action]
        next_common_action = (
            Counter(next_actions).most_common(1)[0][0] if next_actions else most_common_action
        )

        # Shared entities
        all_entities: list[str] = []
        for n in cluster:
            all_entities.extend(extract_entities_from_content(n.get("content", "")))
        entity_counter = Counter(all_entities)
        shared_entities = [e for e, c in entity_counter.most_common(5) if c >= len(cluster) * 0.5]

        # Shared action types (node_type)
        node_types = [n.get("node_type", "") for n in cluster]
        shared_types = [
            t for t, c in Counter(node_types).most_common() if c >= len(cluster) * 0.5
        ]

        # Average confidence
        avg_confidence = sum(float(n.get("confidence", 0.5)) for n in cluster) / len(cluster)

        # Success rate from cluster
        success_count = sum(1 for n in cluster if "success" in n.get("content", "").lower())
        success_rate = success_count / len(cluster)

        # Generate name
        name = f"{most_common_action}_{domain}".replace(" ", "_")

        # Generate behavioral pattern
        behavioral_pattern = (
            f"When performing {domain} tasks: {most_common_action} is typically "
            f"followed by {next_common_action}"
        )

        # Conditions (what must be true for this schema to apply)
        conditions: list[str] = []
        if shared_entities:
            conditions.append(f"Entities {', '.join(shared_entities[:3])} are involved")
        if shared_types:
            conditions.append(f"Node types: {', '.join(shared_types)}")
        if domain:
            conditions.append(f"Domain is {domain}")

        # Exceptions (when the pattern does NOT apply)
        exceptions: list[str] = []
        failed_nodes = [n for n in cluster if "fail" in n.get("content", "").lower()]
        if failed_nodes and success_rate < 0.5:
            exceptions.append("Previous attempts in this domain have failed")

        # Source IDs
        source_ids = [n["id"] for n in cluster]

        # Determine status and confidence
        # Check if this domain+pattern already exists
        existing = self.db.fetchone(
            "SELECT id, induction_count, confidence, status, semantic_source_ids FROM schemas "
            "WHERE domain = ? AND behavioral_pattern = ? AND status IN ('hypothesis', 'active')",
            (domain, behavioral_pattern),
        )
        if existing:
            # Re-induction: update existing
            induction_count = int(existing["induction_count"]) + 1
            old_confidence = float(existing["confidence"])
            new_confidence = 1.0 - (1.0 - old_confidence) * (1.0 - avg_confidence)

            new_status = existing["status"]
            if induction_count >= 2:
                new_status = SchemaStatus.ACTIVE.value

            old_sources = json.loads(existing.get("semantic_source_ids", "[]") or "[]")
            merged_sources = list(set(old_sources + source_ids))

            self.db.update(
                "schemas",
                existing["id"],
                {
                    "confidence": new_confidence,
                    "induction_count": induction_count,
                    "semantic_source_ids": merged_sources,
                    "status": new_status,
                    "updated_at": _now_iso(),
                    "conditions": json.dumps(conditions),
                    "exceptions": json.dumps(exceptions),
                },
            )
            return None  # Already merged in-place

        # First induction
        schema = SchemaModel(
            name=name,
            description=f"Schema induced from {len(cluster)} semantic nodes in {domain}",
            behavioral_pattern=behavioral_pattern,
            schema_type=SchemaType.BEHAVIORAL,
            semantic_source_ids=source_ids,
            confidence=HYPOTHESIS_CONFIDENCE,
            induction_count=1,
            domain=domain,
            conditions=conditions,
            exceptions=exceptions,
            status=SchemaStatus.HYPOTHESIS,
        )
        data = schema.model_dump()
        data["conditions"] = json.dumps(data["conditions"])
        data["exceptions"] = json.dumps(data["exceptions"])
        data["semantic_source_ids"] = json.dumps(data["semantic_source_ids"])
        data["episode_source_ids"] = json.dumps(data["episode_source_ids"])
        self.db.insert("schemas", data)
        return schema

    # ── Merge Duplicates ───────────────────────────────────────────

    def _merge_with_existing(self, schema: SchemaModel, domain: str) -> bool:
        """Check if a newly-induced schema matches an existing one and merge.

        Matches by domain + LHS of behavioral pattern (before 'typically').
        Merged confidence: c = 1 - (1-c_old)*(1-c_new)
        Returns True if merge happened.
        """
        if not schema:
            return False

        # Extract pattern key (before " is typically ")
        pattern_key = schema.behavioral_pattern.split(" is typically ")[0]

        existing = self.db.fetchone(
            "SELECT id, induction_count, confidence, status, semantic_source_ids, episode_source_ids "
            "FROM schemas WHERE domain = ? AND status NOT IN ('refuted', 'archived') "
            "AND behavioral_pattern LIKE ?",
            (domain, f"{pattern_key}%"),
        )
        if not existing:
            return False

        # Merge confidence: c = 1 - (1-c_old)*(1-c_new)
        old_conf = float(existing["confidence"])
        merged_conf = 1.0 - (1.0 - old_conf) * (1.0 - schema.confidence)

        new_induction_count = int(existing["induction_count"]) + 1

        # Merge source IDs
        old_sources = json.loads(existing.get("semantic_source_ids", "[]") or "[]")
        new_sources = list(set(old_sources + schema.semantic_source_ids))

        # Update status based on induction count
        new_status = existing["status"]
        if new_induction_count >= 2 and new_status == SchemaStatus.HYPOTHESIS.value:
            new_status = SchemaStatus.ACTIVE.value

        self.db.update(
            "schemas",
            existing["id"],
            {
                "confidence": min(merged_conf, 0.99),
                "induction_count": new_induction_count,
                "semantic_source_ids": json.dumps(new_sources),
                "status": new_status,
                "updated_at": _now_iso(),
            },
        )
        return True

    # ── Lifecycle Management ───────────────────────────────────────

    def _archive_stale_schemas(self, stale_days: int = ARCHIVE_DAYS) -> int:
        """Archive schemas not reinforced in N days."""
        cutoff = (datetime.utcnow() - timedelta(days=stale_days)).isoformat()
        stale = self.db.fetchall(
            "SELECT id FROM schemas "
            "WHERE updated_at < ? AND status NOT IN ('archived', 'refuted')",
            (cutoff,),
        )
        for s in stale:
            self.db.update("schemas", s["id"], {
                "status": SchemaStatus.ARCHIVED.value,
                "updated_at": _now_iso(),
            })
        return len(stale)

    def _check_contradictions(self) -> int:
        """Check for schemas with contradictory evidence.

        A schema is refuted if:
        - It has low confidence (< 0.3) after multiple inductions
        - Semantic sources have been superseded
        """
        refuted = 0
        candidates = self.db.fetchall(
            "SELECT s.id, s.confidence, s.induction_count, s.semantic_source_ids, s.name "
            "FROM schemas s "
            "WHERE s.status NOT IN ('refuted', 'archived') "
            "AND s.induction_count > 0",
        )
        for s in candidates:
            should_refute = False

            # Low confidence after multiple inductions
            if float(s["confidence"]) < 0.3 and int(s["induction_count"]) >= 2:
                should_refute = True

            # All source nodes superseded
            source_ids = json.loads(s.get("semantic_source_ids", "[]") or "[]")
            if source_ids:
                placeholders = ",".join("?" for _ in source_ids)
                alive = self.db.fetchone(
                    f"SELECT COUNT(*) as cnt FROM semantic_nodes "
                    f"WHERE id IN ({placeholders}) AND valid_until IS NULL",
                    tuple(source_ids),
                )
                if alive and alive["cnt"] == 0:
                    should_refute = True

            if should_refute:
                self.db.update("schemas", s["id"], {
                    "status": SchemaStatus.REFUTED.value,
                    "updated_at": _now_iso(),
                })
                refuted += 1

        return refuted
