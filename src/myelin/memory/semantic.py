"""Semantic memory: facts, reflections, and higher-order knowledge."""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.database import Database
from ..core.json_utils import deserialize_row
from ..core.models import NodeType, SemanticNode


class SemanticMemory:
    def __init__(self, db: Database):
        self.db = db

    def store(self, node: SemanticNode) -> str:
        data = node.model_dump()
        if data.get("embedding"):
            from ..core.database import _serialize_f32

            data["embedding"] = _serialize_f32(data["embedding"])
        self.db.insert("semantic_nodes", data)
        return node.id

    def get(self, node_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM semantic_nodes WHERE id = ?", (node_id,))
        if row:
            deserialize_row(row)
        return row

    def access(self, node_id: str) -> None:
        row = self.db.fetchone(
            "SELECT access_count, access_times FROM semantic_nodes WHERE id = ?",
            (node_id,),
        )
        if not row:
            return
        times = json.loads(row["access_times"])
        times.append(time.time())
        self.db.update(
            "semantic_nodes",
            node_id,
            {
                "access_count": row["access_count"] + 1,
                "access_times": times,
                "last_accessed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def search_text(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.db.fts_search("semantic_nodes", "semantic_fts", query, limit=limit)

    def search_hybrid(
        self,
        text_query: str,
        query_vec: list[float] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self.db.hybrid_search(
            "semantic_nodes", "semantic_fts", text_query, query_vec, limit=limit
        )

    def get_reflections(self, domain: str | None = None) -> list[dict[str, Any]]:
        if domain:
            return self.db.fetchall(
                "SELECT * FROM semantic_nodes WHERE node_type IN (?, ?) AND domain = ? ORDER BY created_at DESC",
                (NodeType.REFLECTION.value, NodeType.META_REFLECTION.value, domain),
            )
        return self.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE node_type IN (?, ?) ORDER BY created_at DESC",
            (NodeType.REFLECTION.value, NodeType.META_REFLECTION.value),
        )

    def get_facts(self, domain: str | None = None) -> list[dict[str, Any]]:
        if domain:
            return self.db.fetchall(
                "SELECT * FROM semantic_nodes WHERE node_type = ? AND domain = ? AND valid_until IS NULL ORDER BY confidence DESC",
                (NodeType.FACT.value, domain),
            )
        return self.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE node_type = ? AND valid_until IS NULL ORDER BY confidence DESC",
            (NodeType.FACT.value,),
        )

    def supersede(self, old_id: str, new_node: SemanticNode) -> str:
        """Replace a semantic node with a newer version."""
        self.db.update(
            "semantic_nodes",
            old_id,
            {
                "valid_until": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "superseded_by": new_node.id,
            },
        )
        return self.store(new_node)

    def count(self, node_type: NodeType | None = None) -> int:
        if node_type:
            row = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM semantic_nodes WHERE node_type = ?",
                (node_type.value,),
            )
        else:
            row = self.db.fetchone("SELECT COUNT(*) as cnt FROM semantic_nodes")
        return row["cnt"] if row else 0
