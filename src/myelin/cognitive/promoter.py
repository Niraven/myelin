"""Promoter: detect patterns and promote episode clusters to procedures.

Uses ACT-R activation math instead of simple counting.
Trigger: session end.
"""

from __future__ import annotations

from typing import Any

from ..core.activation import base_level_activation, should_promote
from ..core.database import Database
from ..core.models import (
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    ProcessName,
    StepType,
)
from ..memory.episodic import EpisodicMemory
from ..memory.procedural import ProceduralMemory
from .base import CognitiveProcess

PROMOTION_THRESHOLD = 1.0
MIN_EPISODES = 2


class Promoter(CognitiveProcess):
    name = ProcessName.PROMOTER

    def __init__(
        self,
        db: Database,
        episodic: EpisodicMemory,
        procedural: ProceduralMemory,
    ):
        super().__init__(db)
        self.episodic = episodic
        self.procedural = procedural

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        """Find episode clusters above activation threshold and promote them."""
        clusters = self.episodic.get_activation_scores(min_activation=PROMOTION_THRESHOLD)

        created = 0
        for cluster in clusters:
            if cluster["episode_count"] < MIN_EPISODES:
                continue

            existing = self.db.fetchone(
                "SELECT id FROM procedures WHERE source_episodes LIKE ?",
                (f'%{cluster["cluster_id"]}%',),
            )
            if existing:
                continue

            procedure = self._extract_procedure(cluster)
            if procedure:
                self.procedural.store(procedure)
                created += 1

        return {"processed": len(clusters), "created": created}

    def _extract_procedure(self, cluster: dict[str, Any]) -> Procedure | None:
        """Extract a procedure from an episode cluster.

        Phase 0: basic action sequence extraction.
        Phase 1: ClustalW-inspired progressive multiple alignment.
        """
        episodes = cluster["episodes"]
        if not episodes:
            return None

        actions = [ep["action"] for ep in episodes]
        domain = episodes[0].get("domain")
        agent_id = episodes[0].get("agent_id", "unknown")

        action_counts: dict[str, int] = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        total = len(episodes)
        steps = []
        for i, (action, count) in enumerate(
            sorted(action_counts.items(), key=lambda x: -x[1])
        ):
            ratio = count / total
            if ratio > 0.8:
                step_type = StepType.CORE
            elif ratio > 0.4:
                step_type = StepType.OPTIONAL
            else:
                continue

            steps.append(ProcedureStep(
                order=i,
                description=action,
                step_type=step_type,
            ))

        if not steps:
            return None

        name = f"auto_{domain or 'general'}_{cluster['cluster_id'][:8]}"
        trigger = f"When performing {domain or 'general'} tasks similar to: {actions[0]}"

        return Procedure(
            name=name,
            description=f"Auto-promoted from {len(episodes)} episodes (activation: {cluster['activation']:.2f})",
            trigger_pattern=trigger,
            steps=steps,
            confidence=0.5,
            activation_score=cluster["activation"],
            access_times=[],
            source_agent=agent_id,
            source_episodes=[ep["id"] for ep in episodes],
            domain=domain,
            status=ProcedureStatus.DRAFT,
        )
