"""Episodic memory: raw observations of agent behavior."""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.activation import base_level_activation
from ..core.database import Database
from ..core.models import ActionType, Episode


class EpisodicMemory:
    def __init__(self, db: Database):
        self.db = db

    def record(self, episode: Episode) -> str:
        """Record a new episode. Returns the episode ID."""
        data = episode.model_dump()
        if data.get("embedding"):
            from ..core.database import _serialize_f32
            data["embedding"] = _serialize_f32(data["embedding"])
        self.db.insert("episodes", data)
        return episode.id

    def get(self, episode_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        if row:
            row["access_times"] = json.loads(row["access_times"])
            row["tags"] = json.loads(row["tags"])
        return row

    def access(self, episode_id: str) -> None:
        """Record an access (retrieval) of this episode for ACT-R activation."""
        row = self.db.fetchone(
            "SELECT access_count, access_times FROM episodes WHERE id = ?",
            (episode_id,),
        )
        if not row:
            return
        times = json.loads(row["access_times"])
        times.append(time.time())
        self.db.update("episodes", episode_id, {
            "access_count": row["access_count"] + 1,
            "access_times": times,
            "last_accessed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    def search_text(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.db.fts_search("episodes", "episodes_fts", query, limit=limit)

    def search_vector(
        self, query_vec: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        return self.db.vec_search("episodes", "embedding", query_vec, limit=limit)

    def search_hybrid(
        self,
        text_query: str,
        query_vec: list[float] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self.db.hybrid_search(
            "episodes", "episodes_fts", text_query, query_vec, limit=limit
        )

    def get_unconsolidated(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM episodes WHERE consolidated = 0 ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        )

    def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM episodes WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        )

    def get_by_domain(self, domain: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM episodes WHERE domain = ? ORDER BY timestamp DESC LIMIT ?",
            (domain, limit),
        )

    def get_cluster(self, cluster_id: str) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM episodes WHERE cluster_id = ? ORDER BY timestamp ASC",
            (cluster_id,),
        )

    def mark_consolidated(self, episode_ids: list[str], cluster_id: str) -> None:
        placeholders = ",".join("?" for _ in episode_ids)
        self.db.execute(
            f"UPDATE episodes SET consolidated = 1, cluster_id = ? WHERE id IN ({placeholders})",
            (cluster_id, *episode_ids),
        )
        self.db.commit()

    def count(self, agent_id: str | None = None) -> int:
        if agent_id:
            row = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM episodes WHERE agent_id = ?", (agent_id,)
            )
        else:
            row = self.db.fetchone("SELECT COUNT(*) as cnt FROM episodes")
        return row["cnt"] if row else 0

    def get_activation_scores(
        self, domain: str | None = None, min_activation: float = 0.0
    ) -> list[dict[str, Any]]:
        """Get episodes grouped by cluster with ACT-R activation scores."""
        where = "WHERE cluster_id IS NOT NULL"
        params: list[Any] = []
        if domain:
            where += " AND domain = ?"
            params.append(domain)

        clusters = self.db.fetchall(
            f"SELECT DISTINCT cluster_id FROM episodes {where}", tuple(params)
        )

        results = []
        for cluster in clusters:
            cid = cluster["cluster_id"]
            episodes = self.get_cluster(cid)
            all_times: list[float] = []
            for ep in episodes:
                all_times.extend(json.loads(ep["access_times"]) if isinstance(ep["access_times"], str) else ep["access_times"])

            activation = base_level_activation(all_times)
            if activation > min_activation:
                results.append({
                    "cluster_id": cid,
                    "activation": activation,
                    "episode_count": len(episodes),
                    "episodes": episodes,
                })

        results.sort(key=lambda x: x["activation"], reverse=True)
        return results
