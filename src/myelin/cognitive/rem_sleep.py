"""REM sleep phase: random walk dreaming, counterfactual generation, TAG scoring.

REM (rapid eye movement) sleep is the second phase of the two-phase sleep
cycle. It performs:

1. Random walk dreaming (SCM Algo 1): pick top-3 entities by mention_count,
   run 5-step BFS random walks through the knowledge graph, create
   'dreamed_connection' edges for novel pairs or boost sub-0.5 edges.

2. Counterfactual generation (ZenBrain CA3): find failed episodes, generate
   "what if" alternatives by replacing the failing action, store as dreams
   in the semantic layer.

3. Novel connection discovery: find cross-domain entity pairs where the
   shortest path > 3 hops, link via shared attributes with weak strength.

4. Importance-weighted replay selection using TAG score:
   TAG(e) = 0.4 * |delta_TD| + 0.35 * R_e + 0.25 * N_e
   where R_e = consequence (importance_score) and N_e = novelty
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import NodeType, ProcessName, RelationType, SemanticNode, SourceType
from ..knowledge.entities import EntityStore, extract_entities_from_text
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.temporal import TemporalIndex
from .base import CognitiveProcess

DREAM_WALK_STEPS = 5
DREAM_WALK_START = 3  # top-N entities by mention_count
DREAM_WALK_STRENGTH = 0.2
DREAM_BOOST_THRESHOLD = 0.5
DREAM_BOOST_DELTA = 0.1
COUNTERFACTUAL_BATCH = 10
NOVEL_DOMAIN_LIMIT = 10
NOVEL_CONNECTION_STRENGTH = 0.15
NOVEL_HOPS_THRESHOLD = 3
TAG_TD_WEIGHT = 0.4
TAG_R_WEIGHT = 0.35
TAG_N_WEIGHT = 0.25
REM_SAMPLE_LIMIT = 20


def _new_id() -> str:
    return uuid4().hex[:16]


class REMPhase(CognitiveProcess):
    """REM sleep: random walk dreaming, counterfactuals, novel connections, TAG scoring."""

    name = ProcessName.REM_SLEEP

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
        results: dict[str, Any] = {
            "dreamed_connections_created": 0,
            "dreamed_connections_boosted": 0,
            "counterfactuals_generated": 0,
            "novel_connections_created": 0,
            "tag_scores_computed": 0,
            "tag_replay_selected": 0,
            "dream_walks_completed": 0,
        }

        # Step 1: Random walk dreaming
        walked_created, walked_boosted, count = self._random_walk_dreaming()
        results["dreamed_connections_created"] = walked_created
        results["dreamed_connections_boosted"] = walked_boosted
        results["dream_walks_completed"] = count

        # Step 2: Counterfactual generation
        counterfactuals = self._generate_counterfactuals()
        results["counterfactuals_generated"] = counterfactuals

        # Step 3: Novel cross-domain connection discovery
        novel = self._discover_novel_connections()
        results["novel_connections_created"] = novel

        # Step 4: TAG importance-weighted replay selection
        tagged, sampled = self._tag_scoring_and_sample()
        results["tag_scores_computed"] = tagged
        results["tag_replay_selected"] = sampled

        self.db.commit()
        return results

    # ── Random Walk Dreaming ────────────────────────────────────

    def _random_walk_dreaming(self) -> tuple[int, int, int]:
        """SCM Algo 1: pick top entities and run random walks.

        For each of the top-3 entities by mention_count:
          5-step random walk through the knowledge graph (BFS with random
          neighbor selection). At each step, if the neighbor pair has no
          existing relationship, create 'dreamed_connection' with strength 0.2.
          If a relationship exists and strength < 0.5, boost by 0.1.
        """
        top_entities = self.db.fetchall(
            "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
            (DREAM_WALK_START,),
        )
        if not top_entities:
            return 0, 0, 0

        created = 0
        boosted = 0
        walks = 0

        for start_entity in top_entities:
            entity_id = start_entity["id"]
            current_id = entity_id

            for step in range(DREAM_WALK_STEPS):
                # Get neighbors (both directions)
                neighbors = self.graph.get_neighbors(
                    entity_id=current_id,
                    direction="both",
                    min_strength=0.0,
                    limit=20,
                )

                if not neighbors:
                    break  # Dead end — terminate this walk

                # Random neighbor selection (BFS style: uniform random)
                chosen = random.choice(neighbors)
                neighbor_id = chosen["id"]

                # Check existing relationship between the *walk origin* and this neighbor
                if current_id != entity_id:
                    # Cross-edge: origin ↔ neighbor (not just current step)
                    passes = [(entity_id, neighbor_id), (neighbor_id, entity_id)]
                else:
                    passes = [(entity_id, neighbor_id), (neighbor_id, entity_id)]

                existing_rel = None
                for src, tgt in passes:
                    row = self.db.fetchone(
                        "SELECT id, strength FROM relationships "
                        "WHERE source_entity_id = ? AND target_entity_id = ? "
                        "AND relation_type = ?",
                        (src, tgt, RelationType.DREAMED_CONNECTION.value),
                    )
                    if row:
                        existing_rel = row
                        break

                if existing_rel:
                    # Relationship exists — boost if weak
                    current_strength = float(existing_rel["strength"])
                    if current_strength < DREAM_BOOST_THRESHOLD:
                        new_strength = min(
                            current_strength + DREAM_BOOST_DELTA, 10.0
                        )
                        self.db.update(
                            "relationships",
                            existing_rel["id"],
                            {
                                "strength": new_strength,
                                "last_observed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            },
                        )
                        boosted += 1
                else:
                    # No relationship — create dreamed_connection
                    # Also check if there's any non-dreamed relationship
                    any_rel = self.db.fetchone(
                        "SELECT id FROM relationships "
                        "WHERE ((source_entity_id = ? AND target_entity_id = ?) "
                        "   OR (source_entity_id = ? AND target_entity_id = ?)) "
                        "AND relation_type != ?",
                        (entity_id, neighbor_id, neighbor_id, entity_id,
                         RelationType.DREAMED_CONNECTION.value),
                    )
                    if not any_rel:
                        self.graph.add_relationship(
                            source_entity_id=entity_id,
                            target_entity_id=neighbor_id,
                            relation_type=RelationType.DREAMED_CONNECTION.value,
                            strength=DREAM_WALK_STRENGTH,
                        )
                        created += 1

                # Move to neighbor for next step
                current_id = neighbor_id
                walks += 1

        return created, boosted, walks

    # ── Counterfactual Generation ───────────────────────────────

    def _generate_counterfactuals(self) -> int:
        """Find failed episodes and generate "what if" alternatives.

        For each failed episode (success=0) in recent history:
          1. Extract the action sequence from the episode
          2. Identify the "failing action" (the action that failed)
          3. Generate an alternative action
          4. Store as semantic_node with type='DREAM'
        """
        failed_eps = self.db.fetchall(
            "SELECT * FROM episodes WHERE success = 0 "
            "ORDER BY timestamp DESC LIMIT ?",
            (COUNTERFACTUAL_BATCH,),
        )

        if not failed_eps:
            return 0

        generated = 0
        for ep in failed_eps:
            content = ep.get("content_text", "") or ""
            action = ep.get("action", "") or ""
            domain = ep.get("domain")
            ep_id = ep["id"]

            # Build the counterfactual
            alternative = self._build_counterfactual(action, content)
            if not alternative:
                continue

            # Store as a semantic dream node
            node = SemanticNode(
                node_type=NodeType.DREAM,
                content=json.dumps({
                    "type": "counterfactual_dream",
                    "original_episode_id": ep_id,
                    "original_action": action,
                    "alternative": alternative,
                    "domain": domain,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }),
                source_type=SourceType.OBSERVATION,
                source_ids=[ep_id],
                domain=domain,
                confidence=0.3,  # Low confidence — these are speculative
                tags=["dream", "counterfactual"],
            )
            self.db.insert("semantic_nodes", node.model_dump())
            generated += 1

        return generated

    def _build_counterfactual(
        self,
        action: str,
        content: str,
    ) -> str | None:
        """Generate a counterfactual alternative for a failed action.

        Strategy: prepend "Try instead: " with a plausible alternative.
        Uses keyword-based heuristics to generate meaningful alternatives.
        """
        action_lower = action.lower().strip()

        # Map common failure patterns to alternatives
        alternatives = {
            "git push": "git push --force-with-lease",
            "git merge": "git rebase",
            "npm install": "npm install --legacy-peer-deps",
            "pip install": "pip install --no-deps",
            "docker build": "docker build --no-cache",
            "kubectl apply": "kubectl apply --validate=false",
            "deploy": "rollback and retry deploy with --no-restart",
            "migrate": "migrate with --skip-lock",
            "test": "test with --retries 3",
            "install": "install with --no-cache",
        }

        for keyword, alternative in alternatives.items():
            if keyword in action_lower:
                return alternative

        # Generic fallback: extract key terms and wrap with retry semantics
        words = content.split()[:10] if content else action.split()[:10]
        core_terms = [w for w in words if len(w) > 3][:3]
        if core_terms:
            return f"retry with modified parameters: {', '.join(core_terms)}"

        return None

    # ── Novel Connection Discovery ──────────────────────────────

    def _discover_novel_connections(self) -> int:
        """Find cross-domain entity pairs with path > 3 hops, link via shared attributes.

        Strategy:
          1. Pick top entities from each domain
          2. For cross-domain pairs, check if shortest path > 3 hops
          3. If yes, check for shared domain overlap
          4. Create weak relationship (strength=0.15)
        """
        domains = self.db.fetchall(
            "SELECT DISTINCT domain FROM entities "
            "WHERE domain IS NOT NULL AND domain != '' "
            "LIMIT ?",
            (NOVEL_DOMAIN_LIMIT,),
        )

        domain_names = [d["domain"] for d in domains]
        if len(domain_names) < 2:
            return 0

        created = 0
        for i in range(len(domain_names)):
            for j in range(i + 1, len(domain_names)):
                d1 = domain_names[i]
                d2 = domain_names[j]

                # Get top entities from each domain
                d1_entities = self.db.fetchall(
                    "SELECT id, name FROM entities WHERE domain = ? "
                    "ORDER BY mention_count DESC LIMIT 5",
                    (d1,),
                )
                d2_entities = self.db.fetchall(
                    "SELECT id, name FROM entities WHERE domain = ? "
                    "ORDER BY mention_count DESC LIMIT 5",
                    (d2,),
                )

                for e1 in d1_entities:
                    for e2 in d2_entities:
                        if e1["id"] == e2["id"]:
                            continue

                        # Check if any direct or short relationship exists
                        existing = self.db.fetchone(
                            "SELECT id, strength FROM relationships "
                            "WHERE ((source_entity_id = ? AND target_entity_id = ?) "
                            "   OR (source_entity_id = ? AND target_entity_id = ?))",
                            (e1["id"], e2["id"], e2["id"], e1["id"]),
                        )
                        if existing:
                            continue  # Already connected

                        # Check path length
                        path_len = self._shortest_path_length(e1["id"], e2["id"])
                        if path_len is not None and path_len <= NOVEL_HOPS_THRESHOLD:
                            continue  # Already closely connected

                        # Check for shared attributes (overlapping entity types, domain patterns)
                        shared_attrs = self._find_shared_attributes(e1["id"], e2["id"])
                        if shared_attrs:
                            self.graph.add_relationship(
                                source_entity_id=e1["id"],
                                target_entity_id=e2["id"],
                                relation_type=RelationType.RELATED_TO.value,
                                strength=NOVEL_CONNECTION_STRENGTH,
                                domain=f"{d1}-{d2}",
                            )
                            created += 1

                            # Enforce cap per cycle
                            if created >= 20:
                                return created

        return created

    def _shortest_path_length(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> int | None:
        """BFS to find shortest path length between two entities."""
        if source_id == target_id:
            return 0

        visited: set[str] = {source_id}
        queue: deque[tuple[str, int]] = deque([(source_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Get all neighbors (both directions)
            neighbors = self.graph.get_neighbors(
                entity_id=current_id,
                direction="both",
                min_strength=0.0,
                limit=50,
            )

            for neighbor in neighbors:
                nid = neighbor["id"]
                if nid == target_id:
                    return depth + 1
                if nid not in visited:
                    visited.add(nid)
                    queue.append((nid, depth + 1))

        return None  # Not connected within max_depth

    def _find_shared_attributes(self, e1_id: str, e2_id: str) -> list[str]:
        """Find shared attributes between two entities.

        Checks for: matching entity type (tool/service/concept),
        overlapping domains, shared tags, co-mention in episodes.
        """
        shared: list[str] = []
        e1 = self.entities.get_entity(e1_id)
        e2 = self.entities.get_entity(e2_id)

        if not e1 or not e2:
            return shared

        # Same entity type
        if e1.get("entity_type") == e2.get("entity_type"):
            shared.append(f"same_type:{e1['entity_type']}")

        # Check if they co-occur in any episode
        co_mention = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM entity_mentions m1 "
            "JOIN entity_mentions m2 ON m1.source_id = m2.source_id "
            "AND m1.source_type = m2.source_type "
            "WHERE m1.entity_id = ? AND m2.entity_id = ?",
            (e1_id, e2_id),
        )
        if co_mention and co_mention["cnt"] > 0:
            shared.append(f"co_mention_count:{co_mention['cnt']}")

        return shared

    # ── TAG Scoring and Importance-Weighted Replay Selection ───

    def _tag_scoring_and_sample(self) -> tuple[int, int]:
        """Compute TAG scores and sample episodes for REM processing.

        TAG(e) = 0.4 * |delta_TD| + 0.35 * R_e + 0.25 * N_e

        Where:
        - delta_TD = td_error (prediction error magnitude, normalized 0-1)
        - R_e = consequence (importance_score, already 0-1)
        - N_e = novelty (1 - replay_count / max_replay), captures freshness

        Returns (tag_scores_computed, episodes_selected).
        """
        candidates = self.db.fetchall(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 200"
        )
        if not candidates:
            return 0, 0

        # Find max replay count for normalization
        max_replay_row = self.db.fetchone(
            "SELECT MAX(replay_count) as max_rp FROM episodes"
        )
        max_replay = float(max_replay_row["max_rp"]) if max_replay_row else 1.0
        if max_replay <= 0:
            max_replay = 1.0

        tag_scores: list[tuple[str, float, dict[str, Any]]] = []

        for ep in candidates:
            ep_id = ep["id"]

            # TD-error component
            td_error_raw = ep.get("td_error")
            td_error = abs(float(td_error_raw)) if td_error_raw is not None else 0.0
            td_error = max(0.0, min(1.0, td_error))  # Clamp to [0, 1]

            # Consequence (R_e): importance_score
            r_e = float(ep.get("importance_score", 0.5))
            r_e = max(0.0, min(1.0, r_e))

            # Novelty (N_e): 1 - replay_count/max_replay
            replay_count = float(ep.get("replay_count", 0))
            n_e = 1.0 - (replay_count / max_replay)
            n_e = max(0.0, min(1.0, n_e))

            # TAG score
            tag = (
                TAG_TD_WEIGHT * td_error
                + TAG_R_WEIGHT * r_e
                + TAG_N_WEIGHT * n_e
            )
            tag = max(0.0, min(1.0, tag))

            # Persist TAG score to episode
            self.db.update(
                "episodes",
                ep_id,
                {"priority_score": tag},
            )

            tag_scores.append((ep_id, tag, ep))

        # Sort by TAG score descending and pick top
        tag_scores.sort(key=lambda x: x[1], reverse=True)
        selected = tag_scores[:REM_SAMPLE_LIMIT]

        # Mark selected episodes for REM processing (increment replay_count)
        for ep_id, tag, ep in selected:
            current_count = int(ep.get("replay_count", 0))
            self.db.update(
                "episodes",
                ep_id,
                {"replay_count": current_count + 1},
            )

        return len(tag_scores), len(selected)
