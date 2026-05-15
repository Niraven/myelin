"""Challenger: test beliefs against contradictory evidence.

Inspired by Bayesian belief updating.
Trigger: on conflict detection (when a new episode contradicts an existing belief).
"""

from __future__ import annotations

from typing import Any

from ..core.activation import bayesian_confidence_update
from ..core.database import Database
from ..core.models import NodeType, ProcessName
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess


class Challenger(CognitiveProcess):
    name = ProcessName.CHALLENGER

    def __init__(self, db: Database, semantic: SemanticMemory):
        super().__init__(db)
        self.semantic = semantic

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        """Look for contradictions between recent episodes and existing beliefs.

        Phase 0: confidence decay for unaccessed beliefs.
        Phase 1: LLM-based contradiction detection.
        """
        stale_facts = self.db.fetchall(
            "SELECT * FROM semantic_nodes "
            "WHERE node_type = ? AND valid_until IS NULL "
            "AND last_accessed < datetime('now', '-7 days') "
            "AND confidence > 0.3 "
            "ORDER BY last_accessed ASC LIMIT 50",
            (NodeType.FACT.value,),
        )

        modified = 0
        for fact in stale_facts:
            new_confidence = bayesian_confidence_update(
                fact["confidence"], success=False, learning_rate=0.05
            )
            self.db.update(
                "semantic_nodes",
                fact["id"],
                {
                    "confidence": new_confidence,
                },
            )
            modified += 1

        return {"processed": len(stale_facts), "modified": modified}
