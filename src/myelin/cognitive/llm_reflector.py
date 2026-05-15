"""LLM Reflector: pattern-aware multi-level reflection engine.

Replaces placeholder reflections with actual multi-level insight generation:

Level 1 (Observation): Direct patterns from episode clusters
  - Domain-specific action frequency, success rate trends, entity density
  - Identifies what happened and with what reliability

Level 2 (Reflection): Cross-domain synthesis
  - Links actions across domains (e.g., testing before deploying)
  - Identifies procedural chains and dependency patterns

Level 3 (Meta-reflection): Agent behavior insights over time
  - Assesses mastery vs exploration vs learning_needed
  - Tracks confidence evolution across sessions

No external LLM API calls — purely algorithmic extraction from
recent semantic nodes and episode clusters using structured data.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from ..core.database import Database
from ..core.models import NodeType, ProcessName, SemanticNode, SourceType
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess

# ── Constants ──────────────────────────────────────────────────────

REFLECTION_HOURS = 24
MIN_FACTS_FOR_REFLECTION = 2
MIN_FACTS_FOR_LEVEL2 = 3
MIN_DOMAINS_FOR_LEVEL3 = 2
LEVEL2_CONFIDENCE_BOOST = 0.1
LEVEL3_CONFIDENCE_BOOST = 0.15
MAX_REFLECTIONS_PER_PASS = 20
TREND_LOOKBACK_DAYS = 7


def _parse_json_field(val: Any) -> Any:
    """Safely parse a JSON field from the DB."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def _extract_action_from_content(content: str) -> str:
    """Extract the most salient action verb from content.

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
        "debugging": "debug", "fixing": "fix", "patching": "patch",
        "migrating": "migrate", "connecting": "connect",
        "investigating": "investigate",
    }
    content_lower = content.lower()
    for prefix, base in _ING_TO_BASE.items():
        if content_lower.startswith(prefix):
            return base
    words = content_lower.split()[:3]
    for w in words:
        if w.endswith("ing") and len(w) > 5:
            root = w[:-3]
            if len(root) >= 3:
                return root
        if w.endswith("ed") and len(w) > 4:
            return w[:-2]
    return words[0] if words else "performed"


def _extract_entities_from_content(content: str) -> list[str]:
    """Extract potential entity names from content text."""
    words = content.split()
    entities: list[str] = []
    for w in words:
        clean = w.strip('"\'(),.;:!?[]{}')
        if not clean or len(clean) < 3:
            continue
        if clean[0].isupper() and not clean.isupper():
            entities.append(clean.lower())
        if "_" in clean or "-" in clean:
            entities.append(clean.lower())
        if any(kw in clean.lower() for kw in [
            "git", "docker", "pip", "npm", "yarn", "aws", "gcp",
            "kubernetes", "terraform", "postgres", "redis", "nginx",
            "python", "node", "go", "react", "pytest", "jest",
        ]):
            entities.append(clean.lower())
    return entities


def _compute_trend(
    interval_content: str, older_content: str
) -> str:
    """Estimate trend: improving, declining, or stable."""
    # Count success-related terms
    recent_success = sum(1 for w in interval_content.lower().split()
                          if w in ("success", "succeeded", "pass", "passed", "complete"))
    older_success = sum(1 for w in older_content.lower().split()
                         if w in ("success", "succeeded", "pass", "passed", "complete"))
    recent_fail = sum(1 for w in interval_content.lower().split()
                       if w in ("fail", "failed", "failure", "error", "broken"))
    older_fail = sum(1 for w in older_content.lower().split()
                      if w in ("fail", "failed", "failure", "error", "broken"))

    recent_ratio = (recent_success / max(recent_success + recent_fail, 1))
    older_ratio = (older_success / max(older_success + older_fail, 1))

    delta = recent_ratio - older_ratio
    if delta > 0.15:
        return "improving"
    elif delta < -0.15:
        return "declining"
    return "stable"


def _insight_for_domain(
    action: str,
    frequency: int,
    success_rate: float,
    n_total: int,
    entity_density: float,
) -> str:
    """Generate a contextual insight for a domain group."""
    if success_rate > 0.8:
        if n_total >= 5:
            return (
                f"Behavior: {action} appears {frequency}x with {success_rate:.0%} success "
                f"across {n_total} observations. The agent demonstrates mastery in this area — "
                f"the action is reliable and well-practiced."
            )
        else:
            return (
                f"Behavior: {action} appears {frequency}x with {success_rate:.0%} success "
                f"across {n_total} observations. Early positive signal, but limited data — "
                f"continue monitoring for consistency."
            )
    elif n_total < 5:
        return (
            f"Behavior: {action} appears {frequency}x with {success_rate:.0%} success "
            f"across {n_total} observations. The agent is still exploring — limited samples "
            f"prevent reliable assessment."
        )
    elif 0.5 <= success_rate <= 0.8:
        return (
            f"Behavior: {action} appears {frequency}x with {success_rate:.0%} success "
            f"across {n_total} observations. Mixed results suggest the agent has partial "
            f"competence but encounters edge cases or contextual challenges. Learning needed: "
            f"identify the conditions where this action succeeds vs fails."
        )
    else:
        return (
            f"Behavior: {action} appears {frequency}x with {success_rate:.0%} success "
            f"across {n_total} observations. Low success rate indicates this domain requires "
            f"more structured learning. The agent may be attempting tasks beyond current "
            f"capability or lacking necessary context."
        )


def _compute_event_density(entities: list[str]) -> float:
    """Compute entity density as ratio of unique entities to total mentions."""
    if not entities:
        return 0.0
    counter = Counter(entities)
    unique = len(counter)
    total = sum(counter.values())
    return unique / max(total, 1)


# ── Reflector ──────────────────────────────────────────────────────


class LLMReflector(CognitiveProcess):
    """Pattern-aware multi-level reflection engine.

    Generates insights at three levels:
    L1: Direct domain-level pattern observations
    L2: Cross-domain synthesis (actions in domain A → domain B)
    L3: Meta-reflection on agent behavior patterns over time
    """

    name = ProcessName.REFLECTOR

    def __init__(self, db: Database, semantic: SemanticMemory):
        super().__init__(db)
        self.semantic = semantic

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        recent_facts = self.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE node_type = ? "
            "AND created_at > datetime('now', ? || ' hours') "
            "ORDER BY created_at DESC LIMIT 50",
            (NodeType.FACT.value, f"-{REFLECTION_HOURS}"),
        )

        if not recent_facts:
            return {"processed": 0, "created": 0}

        # Parse JSON fields
        for fact in recent_facts:
            fact["source_ids"] = _parse_json_field(fact.get("source_ids", "[]"))
            fact["tags"] = _parse_json_field(fact.get("tags", "[]"))

        created = 0
        level_counts: dict[str, int] = defaultdict(int)

        # ── LEVEL 1: Domain-level observation reflections ────────
        domain_groups = self._group_by_domain(recent_facts)
        for domain, facts in domain_groups.items():
            if len(facts) < MIN_FACTS_FOR_REFLECTION:
                continue

            reflection = self._build_level1_reflection(domain, facts)
            if reflection:
                source_ids = [f["id"] for f in facts]
                node = SemanticNode(
                    node_type=NodeType.REFLECTION,
                    content=reflection,
                    source_type=SourceType.REFLECTION,
                    source_ids=source_ids,
                    domain=domain,
                    confidence=0.55,
                )
                self.semantic.store(node)
                created += 1
                level_counts["level1"] += 1

        # ── LEVEL 2: Cross-domain synthesis ──────────────────────
        if len(domain_groups) >= 2:
            cross_reflections = self._build_level2_reflections(domain_groups)
            for domain, reflection_text in cross_reflections:
                source_ids = [f["id"] for group in domain_groups.values() for f in group]
                node = SemanticNode(
                    node_type=NodeType.REFLECTION,
                    content=reflection_text,
                    source_type=SourceType.REFLECTION,
                    source_ids=source_ids,
                    domain=domain,
                    confidence=0.5 + LEVEL2_CONFIDENCE_BOOST,
                )
                self.semantic.store(node)
                created += 1
                level_counts["level2"] += 1

        # ── LEVEL 3: Meta-reflection ─────────────────────────────
        if len(domain_groups) >= MIN_DOMAINS_FOR_LEVEL3:
            meta = self._build_level3_reflection(domain_groups)
            if meta:
                source_ids = [f["id"] for group in domain_groups.values() for f in group]
                node = SemanticNode(
                    node_type=NodeType.META_REFLECTION,
                    content=meta,
                    source_type=SourceType.REFLECTION,
                    source_ids=source_ids,
                    domain=None,
                    confidence=0.45 + LEVEL3_CONFIDENCE_BOOST,
                )
                self.semantic.store(node)
                created += 1
                level_counts["level3"] = 1

        return {
            "processed": len(recent_facts),
            "created": created,
            "level1": level_counts.get("level1", 0),
            "level2": level_counts.get("level2", 0),
            "level3": level_counts.get("level3", 0),
        }

    def _group_by_domain(
        self, facts: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group semantic nodes by domain."""
        groups: dict[str, list] = defaultdict(list)
        for fact in facts:
            d = fact.get("domain") or "general"
            groups[d].append(fact)
        return dict(groups)

    def _build_level1_reflection(
        self, domain: str, facts: list[dict[str, Any]]
    ) -> str | None:
        """Level 1: Direct domain-level pattern observation.

        Extracts: most common action, success rate trend,
        entity density, and generates insight.
        """
        all_actions: list[str] = []
        all_entities: list[str] = []
        success_count = 0
        total = len(facts)

        for fact in facts:
            content = fact.get("content", "") or ""
            all_actions.append(_extract_action_from_content(content))
            all_entities.extend(_extract_entities_from_content(content))
            if "success" in content.lower() and "fail" not in content.lower()[:20]:
                success_count += 1

        action_counter = Counter(all_actions)
        most_common_action = action_counter.most_common(1)[0][0] if action_counter else "performed"
        top_actions = action_counter.most_common(5)

        # Success rate from confidence-weighted estimation
        confidences = [float(f.get("confidence", 0.5)) for f in facts]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Entity density
        entity_density = _compute_event_density(all_entities)
        top_entities = [e for e, _ in Counter(all_entities).most_common(5)]

        # Trend estimation (compare to older semantic nodes)
        older_facts = self.db.fetchall(
            "SELECT * FROM semantic_nodes WHERE node_type = ? AND domain = ? "
            "AND created_at > datetime('now', ? || ' days') "
            "AND created_at <= datetime('now', ? || ' hours') "
            "ORDER BY created_at ASC LIMIT 20",
            (NodeType.FACT.value, domain, f"-{TREND_LOOKBACK_DAYS}", f"-{REFLECTION_HOURS}"),
        )
        older_content = " ".join(f.get("content", "") for f in older_facts) if older_facts else ""
        interval_content = " ".join(f.get("content", "") for f in facts)
        trend = _compute_trend(interval_content, older_content) if older_facts else "stable"

        # Generate insight
        insight = _insight_for_domain(
            action=most_common_action,
            frequency=top_actions[0][1] if top_actions else 1,
            success_rate=success_count / max(total, 1),
            n_total=total,
            entity_density=entity_density,
        )

        # Entity summary
        entity_text = ""
        if top_entities:
            entity_text = f"Key entities: {', '.join(top_entities[:4])}."
        else:
            entity_text = "No prominent entities identified."

        return (
            f"Reflection on {domain} ({total} observations over {REFLECTION_HOURS}h): "
            f"{trend.capitalize()} trend. "
            f"{success_count} successful, {total - success_count} failed. "
            f"{entity_text} "
            f"Average confidence: {avg_confidence:.2f}. "
            f"{insight}"
        )

    def _build_level2_reflections(
        self, domain_groups: dict[str, list[dict[str, Any]]]
    ) -> list[tuple[str, str]]:
        """Level 2: Cross-domain synthesis.

        Looks for procedural chains: if the most common action in domain A
        typically precedes the most common action in domain B, create a
        linkage reflection.
        """
        domains = list(domain_groups.keys())
        reflections: list[tuple[str, str]] = []

        # Get the primary action for each domain
        domain_actions: dict[str, str] = {}
        for domain, facts in domain_groups.items():
            all_actions = [
                _extract_action_from_content(f.get("content", ""))
                for f in facts
            ]
            if all_actions:
                domain_actions[domain] = Counter(all_actions).most_common(1)[0][0]
            else:
                domain_actions[domain] = "performed"

        # Check sequential dependencies: if domain A episodes were created
        # before domain B episodes, suggest a pipeline
        for i, d1 in enumerate(domains):
            for d2 in domains[i + 1:]:
                facts1 = domain_groups[d1]
                facts2 = domain_groups[d2]

                # Get average creation time
                timestamps1 = [
                    f.get("created_at", "") for f in facts1 if f.get("created_at")
                ]
                timestamps2 = [
                    f.get("created_at", "") for f in facts2 if f.get("created_at")
                ]

                # Simple heuristic: if domain A facts were mostly created
                # before domain B facts, suggest d1 -> d2 pipeline
                if timestamps1 and timestamps2:
                    avg_ts1 = sorted(timestamps1)[len(timestamps1) // 2]
                    avg_ts2 = sorted(timestamps2)[len(timestamps2) // 2]

                    if avg_ts1 < avg_ts2:
                        reflection_text = (
                            f"Cross-domain observation: Activities in {d1} "
                            f"({domain_actions.get(d1, 'performed')}) temporally precede "
                            f"activities in {d2} ({domain_actions.get(d2, 'performed')}). "
                            f"This suggests a procedural pipeline: "
                            f"{domain_actions.get(d1, 'work')} in {d1} enables or triggers "
                            f"{domain_actions.get(d2, 'work')} in {d2}. "
                            f"The agent may benefit from formalizing this as a composite procedure."
                        )
                        reflections.append((d2, reflection_text))
                    elif avg_ts2 < avg_ts1:
                        reflection_text = (
                            f"Cross-domain observation: Activities in {d2} "
                            f"({domain_actions.get(d2, 'performed')}) temporally precede "
                            f"activities in {d1} ({domain_actions.get(d1, 'performed')}). "
                            f"This suggests a feedback loop: "
                            f"outcomes in {d2} inform decisions in {d1}. "
                            f"The agent appears to be iterating between these domains."
                        )
                        reflections.append((d1, reflection_text))

        return reflections

    def _build_level3_reflection(
        self, domain_groups: dict[str, list[dict[str, Any]]]
    ) -> str | None:
        """Level 3: Meta-reflection on agent behavior patterns.

        Synthesizes across all domains to produce a higher-order insight
        about the agent's learning trajectory and behavioral style.
        """
        total_observations = sum(len(facts) for facts in domain_groups.values())
        n_domains = len(domain_groups)

        if total_observations < MIN_FACTS_FOR_REFLECTION:
            return None

        # Calculate domain diversity
        domain_sizes = {d: len(f) for d, f in domain_groups.items()}
        largest_domain = max(domain_sizes, key=domain_sizes.get)
        smallest_domain = min(domain_sizes, key=domain_sizes.get)

        # Get overall success rate
        total_success = 0
        all_actions: list[str] = []
        for facts in domain_groups.values():
            for fact in facts:
                content = fact.get("content", "") or ""
                if "success" in content.lower() and "fail" not in content.lower()[:20]:
                    total_success += 1
                all_actions.append(_extract_action_from_content(content))

        overall_success_rate = total_success / max(total_observations, 1)
        action_counter = Counter(all_actions)
        top3_actions = [a for a, _ in action_counter.most_common(3)]

        # Behavioral characterization
        if n_domains >= 3 and overall_success_rate > 0.7:
            behavioral_profile = (
                f"The agent is operating across {n_domains} distinct domains "
                f"with {overall_success_rate:.0%} overall success. This indicates "
                f"versatility and effective multi-domain task management. "
                f"The strongest domain is {largest_domain} ({domain_sizes[largest_domain]} observations), "
                f"suggesting a concentration of effort there."
            )
        elif overall_success_rate < 0.4 and total_observations > 10:
            behavioral_profile = (
                f"The agent has {overall_success_rate:.0%} overall success across "
                f"{n_domains} domains ({total_observations} total observations). "
                f"This low success rate despite significant activity suggests the agent "
                f"is operating at the edge of its capabilities. The most active domain "
                f"({largest_domain}) may need procedural scaffolding or additional context."
            )
        else:
            behavioral_profile = (
                f"The agent is active in {n_domains} domains with "
                f"{overall_success_rate:.0%} overall success. Most activity is in "
                f"{largest_domain} ({domain_sizes[largest_domain]} observations), "
                f"while {smallest_domain} ({domain_sizes[smallest_domain]} observations) "
                f"receives less attention. Primary actions: "
                f"{', '.join(top3_actions)}."
            )

        # Cross-domain learning pattern
        if n_domains >= 2:
            # Check if one domain has much higher success than others
            domain_rates: dict[str, float] = {}
            for domain, facts in domain_groups.items():
                s = sum(1 for f in facts if "success" in (f.get("content", "") or "").lower()
                        and "fail" not in (f.get("content", "") or "").lower()[:20])
                domain_rates[domain] = s / max(len(facts), 1)

            max_rate = max(domain_rates.values()) if domain_rates else 0
            min_rate = min(domain_rates.values()) if domain_rates else 0
            spread = max_rate - min_rate

            if spread > 0.5:
                learning_insight = (
                    f"Success rate varies significantly across domains "
                    f"(range: {min_rate:.0%}–{max_rate:.0%}), suggesting domain-specific "
                    f"competence gaps. The agent may benefit from cross-domain transfer: "
                    f"applying successful patterns from high-performing domains to "
                    f"low-performing ones."
                )
            else:
                learning_insight = (
                    f"Success rates are relatively consistent across domains "
                    f"(range: {min_rate:.0%}–{max_rate:.0%}), indicating balanced capability."
                )
        else:
            learning_insight = (
                f"Single-domain focus on {largest_domain}. The agent may benefit from "
                f"exploring related domains to develop transferable skills."
            )

        return (
            f"Meta-reflection based on {total_observations} observations across "
            f"{n_domains} domains in the last {REFLECTION_HOURS}h. "
            f"{behavioral_profile} "
            f"{learning_insight} "
            f"Overall trajectory: {overall_success_rate:.0%} success rate across "
            f"{total_observations} observations suggests "
            f"{'strong consolidation' if overall_success_rate > 0.75 else 'active learning in progress' if overall_success_rate > 0.4 else 'need for structured intervention'}."
        )
