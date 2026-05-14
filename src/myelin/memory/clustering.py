"""Episode clustering engine for procedure promotion.

Uses a combination of domain grouping, action sequence similarity,
and optional embedding-based cosine similarity to group related episodes
into clusters that can be promoted to procedures.

Phase 0 used simple domain grouping.
Phase 1 uses hierarchical agglomerative clustering with multiple signals.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def action_sequence_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """Normalized LCS length between two action sequences."""
    if not seq_a or not seq_b:
        return 0.0
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    return 2.0 * lcs_len / (m + n)


def episode_similarity(
    ep_a: dict[str, Any],
    ep_b: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """Multi-signal similarity between two episodes.

    Signals:
    1. Domain match (binary)
    2. Action type match (binary)
    3. Action text similarity (Jaccard on tokens)
    4. Embedding cosine similarity (if available)
    """
    w = weights or {
        "domain": 0.2,
        "action_type": 0.1,
        "action_text": 0.3,
        "embedding": 0.4,
    }

    scores: dict[str, float] = {}

    # Domain match
    scores["domain"] = 1.0 if ep_a.get("domain") == ep_b.get("domain") else 0.0

    # Action type match
    scores["action_type"] = 1.0 if ep_a.get("action_type") == ep_b.get("action_type") else 0.0

    # Action text similarity (Jaccard on tokens)
    tokens_a = set(ep_a.get("content_text", "").lower().split())
    tokens_b = set(ep_b.get("content_text", "").lower().split())
    if tokens_a and tokens_b:
        scores["action_text"] = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        scores["action_text"] = 0.0

    # Embedding similarity
    emb_a = ep_a.get("embedding")
    emb_b = ep_b.get("embedding")
    if emb_a and emb_b:
        if isinstance(emb_a, bytes):
            import struct
            dim = len(emb_a) // 4
            emb_a = list(struct.unpack(f"{dim}f", emb_a))
        if isinstance(emb_b, bytes):
            import struct
            dim = len(emb_b) // 4
            emb_b = list(struct.unpack(f"{dim}f", emb_b))
        scores["embedding"] = cosine_similarity(emb_a, emb_b)
    else:
        # Redistribute embedding weight to action_text
        w = dict(w)
        w["action_text"] += w.get("embedding", 0)
        w["embedding"] = 0.0
        scores["embedding"] = 0.0

    total = sum(w[k] * scores[k] for k in scores)
    weight_sum = sum(w[k] for k in scores if w[k] > 0)
    return total / weight_sum if weight_sum > 0 else 0.0


class EpisodeClusterer:
    """Hierarchical agglomerative clustering for episodes.

    Uses average-linkage with multi-signal similarity.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        min_cluster_size: int = 2,
        max_cluster_size: int = 50,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def cluster(self, episodes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Cluster episodes using hierarchical agglomerative clustering.

        Returns list of clusters, each cluster is a list of episodes.
        """
        if len(episodes) < self.min_cluster_size:
            return []

        n = len(episodes)

        # Pre-group by domain for efficiency
        domain_groups: dict[str, list[int]] = defaultdict(list)
        for i, ep in enumerate(episodes):
            domain = ep.get("domain") or "general"
            domain_groups[domain].append(i)

        all_clusters: list[list[dict]] = []

        # Cluster within each domain group
        for domain, indices in domain_groups.items():
            if len(indices) < self.min_cluster_size:
                continue

            domain_episodes = [episodes[i] for i in indices]
            clusters = self._agglomerative_cluster(domain_episodes)
            all_clusters.extend(clusters)

        return all_clusters

    def _agglomerative_cluster(
        self, episodes: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Average-linkage agglomerative clustering."""
        n = len(episodes)
        if n < self.min_cluster_size:
            return []

        # Compute pairwise similarity matrix
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = episode_similarity(episodes[i], episodes[j])
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        # Initialize: each episode is its own cluster
        clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
        active = set(range(n))

        while len(active) > 1:
            # Find most similar pair of clusters
            best_sim = -1.0
            best_pair = (-1, -1)

            active_list = sorted(active)
            for idx_i, ci in enumerate(active_list):
                for cj in active_list[idx_i + 1:]:
                    # Average linkage
                    total_sim = 0.0
                    count = 0
                    for ei in clusters[ci]:
                        for ej in clusters[cj]:
                            total_sim += sim_matrix[ei][ej]
                            count += 1
                    avg_sim = total_sim / count if count > 0 else 0.0

                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (ci, cj)

            if best_sim < self.similarity_threshold:
                break

            ci, cj = best_pair
            merged_size = len(clusters[ci]) + len(clusters[cj])
            if merged_size > self.max_cluster_size:
                break

            # Merge cj into ci
            clusters[ci].extend(clusters[cj])
            del clusters[cj]
            active.remove(cj)

        # Convert index clusters to episode clusters
        result = []
        for indices in clusters.values():
            if len(indices) >= self.min_cluster_size:
                cluster = [episodes[i] for i in indices]
                result.append(cluster)

        return result

    def cluster_by_session_sequences(
        self, episodes: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Group episodes by session, then cluster sessions with similar action sequences.

        This is specifically for procedure promotion: we want to find sessions
        where the agent did similar things in a similar order.
        """
        # Group by session
        sessions: dict[str, list[dict]] = defaultdict(list)
        for ep in episodes:
            sessions[ep.get("session_id", "unknown")].append(ep)

        # Sort each session by timestamp
        for sid in sessions:
            sessions[sid].sort(key=lambda e: e.get("timestamp", ""))

        session_list = list(sessions.values())
        if len(session_list) < self.min_cluster_size:
            return []

        # Compute session similarity using action sequence alignment
        n = len(session_list)
        sim_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            actions_i = [ep.get("action", "") for ep in session_list[i]]
            for j in range(i + 1, n):
                actions_j = [ep.get("action", "") for ep in session_list[j]]
                sim = action_sequence_similarity(actions_i, actions_j)
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        # Cluster sessions
        clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
        active = set(range(n))

        while len(active) > 1:
            best_sim = -1.0
            best_pair = (-1, -1)
            active_list = sorted(active)

            for idx_i, ci in enumerate(active_list):
                for cj in active_list[idx_i + 1:]:
                    total_sim = 0.0
                    count = 0
                    for si in clusters[ci]:
                        for sj in clusters[cj]:
                            total_sim += sim_matrix[si][sj]
                            count += 1
                    avg_sim = total_sim / count if count > 0 else 0.0
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (ci, cj)

            if best_sim < self.similarity_threshold:
                break

            ci, cj = best_pair
            clusters[ci].extend(clusters[cj])
            del clusters[cj]
            active.remove(cj)

        # Flatten: each cluster of sessions -> all episodes from those sessions
        result = []
        for session_indices in clusters.values():
            if len(session_indices) >= self.min_cluster_size:
                all_episodes = []
                for si in session_indices:
                    all_episodes.extend(session_list[si])
                result.append(all_episodes)

        return result
