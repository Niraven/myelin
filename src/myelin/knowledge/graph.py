"""Knowledge graph with typed, weighted edges.

Supermemory has "ontology-aware edges" but they're just similarity-based.
Ours are learned from behavioral sequences: if entity A consistently
appears before entity B in agent workflows, we learn A -> triggers -> B.
If A and B always fail together, we learn A -> causes -> B_error.

The graph supports:
- Typed edges: uses, requires, produces, causes, contradicts, supersedes,
  related_to, part_of, triggers
- Evidence-weighted strength: edges grow stronger with more observations
- Neighborhood queries: find all related entities within N hops
- Subgraph extraction: pull the relevant subgraph for a query
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import Relationship, RelationType


def _new_id() -> str:
    return uuid4().hex[:16]


class KnowledgeGraph:
    """Graph operations over the entities/relationships tables."""

    def __init__(self, db: Database):
        self.db = db

    def add_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        episode_id: str | None = None,
        domain: str | None = None,
        strength: float = 1.0,
    ) -> str:
        """Add or strengthen a relationship between entities."""
        existing = self.db.fetchone(
            "SELECT id, strength, evidence_count, evidence_episodes "
            "FROM relationships "
            "WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?",
            (source_entity_id, target_entity_id, relation_type),
        )

        if existing:
            rel_id = existing["id"]
            episodes = json.loads(existing["evidence_episodes"] or "[]")
            if episode_id and episode_id not in episodes:
                episodes.append(episode_id)
            new_strength = min(existing["strength"] + 0.1, 10.0)
            self.db.update("relationships", rel_id, {
                "strength": new_strength,
                "evidence_count": existing["evidence_count"] + 1,
                "evidence_episodes": episodes,
                "last_observed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            return rel_id

        rel_id = _new_id()
        rel = Relationship(
            id=rel_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=RelationType(relation_type),
            strength=strength,
            evidence_episodes=[episode_id] if episode_id else [],
            domain=domain,
        )
        self.db.insert("relationships", rel.model_dump())
        return rel_id

    def get_neighbors(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
        direction: str = "both",
        min_strength: float = 0.0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get neighboring entities connected by edges."""
        results = []

        if direction in ("out", "both"):
            sql = """
                SELECT e.*, r.relation_type, r.strength, r.evidence_count, 'outgoing' as direction
                FROM relationships r
                JOIN entities e ON e.id = r.target_entity_id
                WHERE r.source_entity_id = ? AND r.strength >= ?
            """
            params: list[Any] = [entity_id, min_strength]
            if relation_types:
                placeholders = ",".join("?" * len(relation_types))
                sql += f" AND r.relation_type IN ({placeholders})"
                params.extend(relation_types)
            sql += " ORDER BY r.strength DESC LIMIT ?"
            params.append(limit)
            results.extend(self.db.fetchall(sql, tuple(params)))

        if direction in ("in", "both"):
            sql = """
                SELECT e.*, r.relation_type, r.strength, r.evidence_count, 'incoming' as direction
                FROM relationships r
                JOIN entities e ON e.id = r.source_entity_id
                WHERE r.target_entity_id = ? AND r.strength >= ?
            """
            params = [entity_id, min_strength]
            if relation_types:
                placeholders = ",".join("?" * len(relation_types))
                sql += f" AND r.relation_type IN ({placeholders})"
                params.extend(relation_types)
            sql += " ORDER BY r.strength DESC LIMIT ?"
            params.append(limit)
            results.extend(self.db.fetchall(sql, tuple(params)))

        return results

    def bfs_subgraph(
        self,
        start_entity_id: str,
        max_depth: int = 2,
        min_strength: float = 0.5,
        max_nodes: int = 50,
    ) -> dict[str, Any]:
        """BFS traversal to extract a subgraph around an entity.

        Returns {nodes: [...], edges: [...]} where each node is an entity
        dict and each edge is a relationship dict.
        """
        visited: set[str] = set()
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        queue: deque[tuple[str, int]] = deque([(start_entity_id, 0)])

        while queue and len(nodes) < max_nodes:
            current_id, depth = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            node = self.db.fetchone("SELECT * FROM entities WHERE id = ?", (current_id,))
            if node:
                nodes.append(node)

            if depth >= max_depth:
                continue

            rels = self.db.fetchall(
                "SELECT * FROM relationships "
                "WHERE (source_entity_id = ? OR target_entity_id = ?) AND strength >= ?",
                (current_id, current_id, min_strength),
            )

            for rel in rels:
                edges.append(rel)
                neighbor = (
                    rel["target_entity_id"]
                    if rel["source_entity_id"] == current_id
                    else rel["source_entity_id"]
                )
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return {"nodes": nodes, "edges": edges}

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
    ) -> list[list[dict[str, Any]]]:
        """Find all paths between two entities up to max_depth."""
        paths: list[list[dict]] = []
        self._dfs_paths(source_id, target_id, max_depth, [], set(), paths)
        return paths

    def _dfs_paths(
        self,
        current: str,
        target: str,
        remaining: int,
        path: list[dict],
        visited: set[str],
        results: list[list[dict]],
    ) -> None:
        if current == target and path:
            results.append(list(path))
            return
        if remaining <= 0:
            return
        visited.add(current)

        rels = self.db.fetchall(
            "SELECT * FROM relationships WHERE source_entity_id = ?",
            (current,),
        )
        for rel in rels:
            neighbor = rel["target_entity_id"]
            if neighbor not in visited:
                path.append(rel)
                self._dfs_paths(neighbor, target, remaining - 1, path, visited, results)
                path.pop()

        visited.discard(current)

    def get_domain_subgraph(
        self,
        domain: str,
        min_strength: float = 0.5,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get the knowledge graph for a specific domain."""
        entities = self.db.fetchall(
            "SELECT * FROM entities WHERE domain = ? ORDER BY mention_count DESC LIMIT ?",
            (domain, limit),
        )
        entity_ids = {e["id"] for e in entities}

        if not entity_ids:
            return {"nodes": [], "edges": []}

        placeholders = ",".join("?" * len(entity_ids))
        edges = self.db.fetchall(
            f"SELECT * FROM relationships "
            f"WHERE source_entity_id IN ({placeholders}) "
            f"AND target_entity_id IN ({placeholders}) "
            f"AND strength >= ?",
            tuple(list(entity_ids) + list(entity_ids) + [min_strength]),
        )

        return {"nodes": entities, "edges": edges}

    def get_relationship_stats(self) -> dict[str, int]:
        rows = self.db.fetchall(
            "SELECT relation_type, COUNT(*) as cnt FROM relationships GROUP BY relation_type"
        )
        return {r["relation_type"]: r["cnt"] for r in rows}

    def count_relationships(self) -> int:
        row = self.db.fetchone("SELECT COUNT(*) as cnt FROM relationships")
        return row["cnt"] if row else 0
