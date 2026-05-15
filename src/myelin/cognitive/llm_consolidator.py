"""LLM Consolidator: pattern-aware semantic summarization of episode clusters.

Instead of placeholder summaries, performs actual informative extraction:
- Groups unconsolidated episodes by domain + content similarity
- Extracts entities, common actions, success/failure patterns from each cluster
- Generates contextual abstractive summaries with meaningful signal
- Stores as semantic nodes with confidence calibrated to cluster quality

No external LLM API calls — purely algorithmic pattern extraction
from structured episode data. Produces genuinely informative summaries.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import NodeType, ProcessName, SemanticNode, SourceType
from ..memory.episodic import EpisodicMemory
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess

# ── Constants ──────────────────────────────────────────────────────

JACCARD_SIMILARITY_THRESHOLD = 0.15
MIN_CLUSTER_SIZE = 2
CONSOLIDATION_BATCH = 50
MAX_ACTIONS_IN_SUMMARY = 5
MAX_ENTITIES_IN_SUMMARY = 5
CONFIDENCE_MIN = 0.4
CONFIDENCE_MAX = 0.95


def _new_id() -> str:
    return uuid4().hex[:16]


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _extract_action_type(action: str) -> str:
    """Extract the verb-like action base from an action string.

    Uses a pre-built mapping for the -ing forms we recognize.
    Falls back to heuristic for unknown actions.
    """
    _ING_TO_BASE = {
        "running": "run", "executing": "execute", "calling": "call",
        "deploying": "deploy", "building": "build",
        "testing": "test", "configuring": "configure",
        "installing": "install", "updating": "update", "creating": "create",
        "removing": "remove", "checking": "check", "verifying": "verify",
        "analyzing": "analyze", "processing": "process",
        "using": "use", "applying": "apply",
        "starting": "start", "stopping": "stop", "restarting": "restart",
        "fetching": "fetch", "downloading": "download", "uploading": "upload",
        "pushing": "push", "pulling": "pull",
        "logging": "log", "monitoring": "monitor", "scanning": "scan",
        "backing up": "backup", "restoring": "restore",
        "migrating": "migrate", "connecting": "connect",
        "disconnecting": "disconnect", "investigating": "investigate",
        "debugging": "debug", "fixing": "fix", "patching": "patch",
        "rolling back": "rollback",
    }
    action_lower = action.lower().strip()
    for prefix, base in _ING_TO_BASE.items():
        if action_lower.startswith(prefix):
            return base
    # Fallback: take first meaningful word
    words = action_lower.split()[:3]
    for w in words:
        if w.endswith("ing") and len(w) > 5:
            # Only strip "ing" if the root is at least 3 chars
            # e.g. "debugging" -> "debugg" (fallback), "thing" -> "thing" (no strip)
            root = w[:-3]
            if len(root) >= 3:
                return root
        if w.endswith("ed") and len(w) > 4:
            return w[:-2] if len(w) > 3 else w
    return words[0] if words else "performed"


def _extract_entities_from_text(text: str) -> list[str]:
    """Extract candidate entity names from text content using heuristics."""
    words = text.split()
    entities: list[str] = []
    for w in words:
        clean = w.strip('"\'(),.;:!?[]{}')
        if not clean or len(clean) < 3:
            continue
        # Capitalized words (proper nouns)
        if clean[0].isupper() and not clean.isupper():
            entities.append(clean.lower())
        # snake_case or kebab-case terms
        if "_" in clean or "-" in clean:
            entities.append(clean.lower())
        # Known tool/service keywords
        if any(kw in clean.lower() for kw in [
            "git", "docker", "pip", "npm", "yarn", "aws", "gcp", "azure",
            "kubernetes", "k8s", "terraform", "ansible", "helm", "istio",
            "postgres", "mysql", "redis", "mongodb", "elasticsearch",
            "python", "node", "go", "rust", "typescript", "javascript",
            "react", "vue", "angular", "django", "flask", "fastapi",
            "pytest", "jest", "mocha", "eslint", "prettier",
            "linux", "ubuntu", "debian", "alpine", "nginx", "apache",
            "vscode", "neovim", "vim", "intellij",
        ]):
            entities.append(clean.lower())
    return entities


def _action_sequence_summary(actions: list[str]) -> str:
    """Summarize a list of action types into a readable pattern."""
    if not actions:
        return "no specific pattern"

    action_counts = Counter(actions)
    most_common = action_counts.most_common(MAX_ACTIONS_IN_SUMMARY)

    if len(most_common) == 1:
        action, count = most_common[0]
        return f"predominantly {action} ({count}x)"
    elif len(most_common) <= 3:
        parts = [f"{a} ({c}x)" for a, c in most_common]
        return "followed by ".join([parts[0], parts[1]]) if len(parts) == 2 \
            else ", then ".join([", ".join(parts[:-1]), parts[-1]])
    else:
        return f"varied: {', '.join(a for a, _ in most_common[:3])} plus {len(most_common) - 3} others"


def _success_analysis(successes: int, total: int, action: str) -> str:
    """Generate a success/failure analysis for a cluster."""
    rate = successes / total if total > 0 else 0.0
    failures = total - successes

    if rate >= 1.0:
        return f"All {total} attempts succeeded — {action} is reliable in this context"
    elif rate >= 0.8:
        return f"{successes}/{total} succeeded ({rate:.0%}) — high reliability, minor issues present"
    elif rate >= 0.5:
        return f"{successes}/{total} succeeded ({rate:.0%}) — moderate reliability, {failures} failure{'s' if failures > 1 else ''} suggest edge cases"
    elif rate > 0.0:
        return f"Only {successes}/{total} succeeded ({rate:.0%}) — low reliability, {failures} failure{'s' if failures > 1 else ''} indicate significant issues"
    else:
        return f"All {total} attempts failed — {action} is unreliable in this context"


def _compute_cluster_confidence(
    total: int, success_rate: float, entity_count: int, unique_actions: int
) -> float:
    """Compute confidence based on cluster quality signals."""
    size_factor = min(total / 10.0, 1.0)
    success_factor = 0.5 + 0.5 * (1.0 - abs(success_rate - 0.75) * 2)
    success_factor = max(0.0, min(1.0, success_factor))
    entity_factor = min(entity_count / 5.0, 1.0)
    action_factor = min(unique_actions / 3.0, 1.0)

    confidence = (
        0.40 * size_factor
        + 0.30 * success_factor
        + 0.15 * entity_factor
        + 0.15 * action_factor
    )
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, confidence))


# ── LLM Consolidator ──────────────────────────────────────────────


class LLMConsolidator(CognitiveProcess):
    """Pattern-aware episode consolidator that generates informative summaries.

    Groups unconsolidated episodes by:
    1. Exact domain match (primary grouping)
    2. Content similarity using Jaccard distance within each domain (secondary grouping)

    For each cluster, extracts:
    - Common entities (tools, services, concepts)
    - Common action types with frequency counts
    - Success/failure patterns with contextual analysis
    - Action sequence patterns

    Stores results as semantic nodes with calibrated confidence scores.
    """

    name = ProcessName.CONSOLIDATOR

    def __init__(self, db: Database, episodic: EpisodicMemory, semantic: SemanticMemory):
        super().__init__(db)
        self.episodic = episodic
        self.semantic = semantic

    def should_run(self) -> bool:
        count = self.episodic.db.fetchone(
            "SELECT COUNT(*) as cnt FROM episodes WHERE consolidated = 0"
        )
        return (count["cnt"] if count else 0) >= CONSOLIDATION_BATCH

    async def execute(self) -> dict[str, Any]:
        episodes = self.episodic.get_unconsolidated(limit=CONSOLIDATION_BATCH)
        if not episodes:
            return {"processed": 0, "created": 0}

        # Phase 1: Cluster by domain
        domain_groups = self._cluster_by_domain(episodes)
        created = 0
        processed = 0

        for domain, domain_eps in domain_groups.items():
            # Phase 2: Within each domain, cluster by content similarity
            clusters = self._cluster_by_content(domain_eps)

            for cluster in clusters:
                if len(cluster) < MIN_CLUSTER_SIZE:
                    continue

                processed += len(cluster)
                cluster_id = _new_id()
                episode_ids = [ep["id"] for ep in cluster]

                # Build informative summary
                summary, metadata = self._build_summary(cluster, domain)

                confidence = _compute_cluster_confidence(
                    total=metadata["total"],
                    success_rate=metadata["success_rate"],
                    entity_count=len(metadata["entities"]),
                    unique_actions=metadata["unique_actions"],
                )

                node = SemanticNode(
                    node_type=NodeType.FACT,
                    content=summary,
                    source_type=SourceType.OBSERVATION,
                    source_ids=episode_ids,
                    domain=domain,
                    confidence=round(confidence, 3),
                    tags=metadata.get("tags", []),
                )
                self.semantic.store(node)
                self.episodic.mark_consolidated(episode_ids, cluster_id)
                created += 1

        return {"processed": processed, "created": created}

    def _cluster_by_domain(
        self, episodes: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group episodes by domain."""
        groups: dict[str, list] = defaultdict(list)
        for ep in episodes:
            domain = ep.get("domain") or "general"
            groups[domain].append(ep)
        return dict(groups)

    def _cluster_by_content(
        self, episodes: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Cluster episodes within a domain using Jaccard similarity on content_text.

        Always checks content similarity — even small groups must pass threshold.
        """
        if not episodes:
            return []

        # For very small groups, still check similarity
        if len(episodes) <= MIN_CLUSTER_SIZE:
            if len(episodes) == 1:
                return [[episodes[0]]]
            # Check pairwise similarity for small groups
            sims = []
            for i in range(len(episodes)):
                for j in range(i + 1, len(episodes)):
                    s = _jaccard_similarity(
                        episodes[i].get("content_text", "") or "",
                        episodes[j].get("content_text", "") or "",
                    )
                    sims.append(s)
            if any(s >= JACCARD_SIMILARITY_THRESHOLD for s in sims):
                return [episodes]
            # No pair meets threshold — keep separate
            return [[ep] for ep in episodes]

        n = len(episodes)
        # Build similarity matrix
        sim_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                text_a = episodes[i].get("content_text", "") or ""
                text_b = episodes[j].get("content_text", "") or ""
                s = _jaccard_similarity(text_a, text_b)
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s

        # Greedy clustering: each ep joins a cluster if it's similar
        # to ANY member already in the cluster
        assigned: set[int] = set()
        clusters: list[list[dict[str, Any]]] = []

        for i in range(n):
            if i in assigned:
                continue
            cluster = [episodes[i]]
            assigned.add(i)

            for j in range(i + 1, n):
                if j in assigned:
                    continue
                # Join if similar to any existing cluster member
                cluster_indices = [
                    episodes.index(m) for m in cluster
                ]
                if any(sim_matrix[k][j] >= JACCARD_SIMILARITY_THRESHOLD for k in cluster_indices):
                    cluster.append(episodes[j])
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def _build_summary(
        self, cluster: list[dict[str, Any]], domain: str
    ) -> tuple[str, dict[str, Any]]:
        """Generate an informative textual summary from a cluster of episodes.

        Returns:
            Tuple of (summary_text, metadata_dict).
        """
        # ── Extract raw signals ──────────────────────────────────
        actions = [ep.get("action", "") for ep in cluster]
        action_types = [ep.get("action_type", "") for ep in cluster]
        successes = sum(1 for ep in cluster if ep.get("success"))
        total = len(cluster)
        success_rate = successes / total if total > 0 else 0.0

        # ── Extract entities from content_text ───────────────────
        entity_counter: Counter[str] = Counter()
        for ep in cluster:
            content = ep.get("content_text", "") or ""
            action = ep.get("action", "") or ""
            combined = f"{action} {content}"
            for ent in _extract_entities_from_text(combined):
                entity_counter[ent] += 1

        top_entities = [e for e, _ in entity_counter.most_common(MAX_ENTITIES_IN_SUMMARY)]

        # ── Extract action type frequency ────────────────────────
        type_counter: Counter[str] = Counter()
        for at in action_types:
            if at:
                type_counter[at] += 1

        # ── Extract normalized action verbs ──────────────────────
        action_verbs = [_extract_action_type(a) for a in actions if a]
        verb_counter = Counter(action_verbs)
        most_common_action = verb_counter.most_common(1)[0][0] if verb_counter else "performed"

        # ── Identify failed episodes for context ─────────────────
        failed = [ep for ep in cluster if not ep.get("success")]
        failure_contexts = []
        for ep in failed:
            action = ep.get("action", "") or ""
            content_text = ep.get("content_text", "") or ""
            snippet = content_text[:120] if content_text else action[:120]
            failure_contexts.append(snippet)

        # ── Build tags ───────────────────────────────────────────
        tags = [domain]
        if success_rate >= 0.8:
            tags.append("reliable")
        elif success_rate <= 0.3:
            tags.append("unreliable")
        if len(cluster) >= 5:
            tags.append("high_volume")
        if len(top_entities) >= 3:
            tags.append("multi_entity")
        if len(failure_contexts) > 0 and success_rate < 0.5:
            tags.append("problematic")

        # ── Action sequence summary ──────────────────────────────
        seq_summary = _action_sequence_summary(action_verbs)

        # ── Success analysis ─────────────────────────────────────
        success_text = _success_analysis(successes, total, most_common_action)

        # ── Entity context ───────────────────────────────────────
        entity_text = ""
        if top_entities:
            entity_text = f"Primarily involved {', '.join(top_entities)}."
        else:
            entity_text = f"Occurred in {domain} context."

        # ── Action type context ──────────────────────────────────
        action_type_text = ""
        type_parts = [f"{t} ({c}x)"
                       for t, c in type_counter.most_common(3)] if type_counter else []
        if type_parts:
            action_type_text = f"Action types: {', '.join(type_parts)}."

        # ── Failure context (if any) ─────────────────────────────
        failure_text = ""
        if failure_contexts and success_rate < 0.8:
            # Pick the most informative failure context
            best_fail = max(failure_contexts, key=len)
            failure_text = f"Failure observed: \"{best_fail[:150]}\""

        # ── Assemble summary ─────────────────────────────────────
        parts = [
            f"{total} observations in {domain}",
            f"Common pattern: {seq_summary}",
            success_text,
            entity_text,
        ]
        if action_type_text:
            parts.append(action_type_text)
        if failure_text:
            parts.append(failure_text)

        summary = ". ".join(parts)

        metadata = {
            "total": total,
            "successes": successes,
            "success_rate": success_rate,
            "entities": top_entities,
            "verbs": list(verb_counter.keys()),
            "unique_actions": len(set(action_verbs)),
            "failure_count": total - successes,
            "tags": tags,
            "most_common_action": most_common_action,
        }

        return summary, metadata
