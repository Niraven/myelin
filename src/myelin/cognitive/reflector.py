"""Reflector: generate higher-order insights from episode clusters.

Inspired by Stanford Generative Agents (Park et al., 2023).
Trigger: session end.

Observations -> Reflections -> Higher-order Reflections
"Nino ran npm test 5x" -> "Nino always tests before deploying"
  -> "Nino is cautious about breaking production"
"""

from __future__ import annotations

from typing import Any

from ..core.database import Database
from ..core.models import NodeType, ProcessName, SemanticNode, SourceType
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess


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
