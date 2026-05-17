"""Promoter: detect patterns and promote episode clusters to procedures.

Phase 1 upgrade: uses real clustering + ClustalW progressive multiple alignment.
Trigger: session end.

Pipeline:
1. Gather unconsolidated episodes
2. Cluster using hierarchical agglomerative clustering (multi-signal similarity)
3. For each cluster, check ACT-R activation score
4. If above threshold, extract action sequences per session
5. Run progressive multiple alignment (ClustalW-inspired)
6. Extract consensus: CORE/OPTIONAL/VARIANT steps
7. Create procedure with branching support
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ..core.activation import base_level_activation, initial_procedure_confidence
from ..core.database import Database
from ..core.models import (
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    ProcessName,
    StepType,
)
from ..memory.alignment import AlignedStep, extract_consensus, progressive_align
from ..memory.clustering import EpisodeClusterer
from ..memory.episodic import EpisodicMemory
from ..memory.procedural import ProceduralMemory
from .base import CognitiveProcess

PROMOTION_THRESHOLD = 0.5
MIN_SESSIONS = 2
MIN_STEPS = 2
RECENT_FALLBACK_LIMIT = 2000
RECENT_FALLBACK_SESSIONS = 10


class Promoter(CognitiveProcess):
    name = ProcessName.PROMOTER

    def __init__(
        self,
        db: Database,
        episodic: EpisodicMemory,
        procedural: ProceduralMemory,
        similarity_threshold: float = 0.5,
    ):
        super().__init__(db)
        self.episodic = episodic
        self.procedural = procedural
        self.clusterer = EpisodeClusterer(
            similarity_threshold=similarity_threshold,
            min_cluster_size=MIN_SESSIONS,
        )

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        """Full Phase 1 promotion pipeline."""
        if not self._has_new_episodes_since_last_run():
            return {"processed": 0, "created": 0, "reason": "no new episodes"}

        # 1. Get unconsolidated episodes first; after session-end consolidation,
        # fall back to recent history so procedure promotion still has evidence.
        all_episodes = self.episodic.get_unconsolidated(limit=500)
        if len(all_episodes) < MIN_SESSIONS:
            all_episodes = self._limit_to_recent_sessions(
                list(reversed(self.episodic.get_recent(limit=RECENT_FALLBACK_LIMIT))),
                max_sessions=RECENT_FALLBACK_SESSIONS,
            )
        if len(all_episodes) < MIN_SESSIONS:
            return {"processed": 0, "created": 0, "reason": "not enough episodes"}

        # 2. Cluster by session sequences (find sessions with similar workflows)
        session_clusters = self.clusterer.cluster_by_session_sequences(all_episodes)

        created = 0
        processed = 0

        for cluster_episodes in session_clusters:
            processed += 1

            # 3. Check activation score (soft signal — skip only if really low AND small cluster)
            all_times: list[float] = []
            for ep in cluster_episodes:
                times = ep.get("access_times", [])
                if isinstance(times, str):
                    times = json.loads(times)
                all_times.extend(times)

            # Fall back to created_at if access_times is empty (fresh episodes)
            if not all_times:
                for ep in cluster_episodes:
                    created_at = ep.get("created_at", "")
                    if created_at:
                        try:
                            import datetime

                            dt = datetime.datetime.fromisoformat(created_at)
                            all_times.append(dt.timestamp())
                        except (ValueError, TypeError):
                            pass

            activation = base_level_activation(all_times) if all_times else 0.0
            # Allow promotion if we have enough sessions regardless of activation
            if activation < PROMOTION_THRESHOLD and len(cluster_episodes) < 5:
                continue

            # 4. Extract action sequences per session
            sessions = self._group_by_session(cluster_episodes)
            if len(sessions) < MIN_SESSIONS:
                continue

            sequences = [
                [ep.get("action", "") for ep in session_eps] for session_eps in sessions.values()
            ]

            # Skip if already have a procedure from similar episodes
            episode_ids = [ep["id"] for ep in cluster_episodes]
            if self._has_existing_procedure(episode_ids):
                continue

            # 5. Run progressive multiple alignment
            alignment = progressive_align(sequences)
            if not alignment:
                continue

            # 6. Extract consensus steps
            consensus = extract_consensus(alignment, min_frequency=0.3)
            if len(consensus) < MIN_STEPS:
                continue

            # 7. Create procedure with branching
            procedure = self._build_procedure(
                consensus=consensus,
                episodes=cluster_episodes,
                activation=activation,
                sessions=sessions,
            )
            if procedure:
                self.procedural.store(procedure)
                created += 1

                # Mark episodes as consolidated
                cluster_id = procedure.id[:16]
                self.episodic.mark_consolidated(episode_ids, cluster_id)

        return {"processed": processed, "created": created}

    def _has_new_episodes_since_last_run(self) -> bool:
        """Check if there are new episodes since the last promoter run.

        Uses unconsolidated count first (fast path), but also falls back
        to checking if *any* new episodes exist since last run. This
        handles the case where the Consolidator already marked all
        episodes as consolidated (because it runs first in on_session_end).
        """
        # Fast path: still have unconsolidated episodes
        row = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM episodes WHERE consolidated = 0",
        )
        if row and row["cnt"] > 0:
            return True

        # Fallback: check if there are any episodes created since last run
        last_run = self.db.fetchone(
            "SELECT completed_at FROM process_runs "
            "WHERE process_name = ? AND status = 'completed' "
            "ORDER BY completed_at DESC LIMIT 1",
            (self.name.value,),
        )
        if not last_run or not last_run.get("completed_at"):
            # No prior promoter run: check if there are any episodes at all
            total = self.db.fetchone("SELECT COUNT(*) as cnt FROM episodes")
            return bool(total and total["cnt"] > 0)

        row2 = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM episodes WHERE created_at > ?",
            (last_run["completed_at"],),
        )
        return bool(row2 and row2["cnt"] > 0)

    def _limit_to_recent_sessions(
        self,
        episodes: list[dict[str, Any]],
        max_sessions: int,
    ) -> list[dict[str, Any]]:
        """Return all episodes from the most recent `max_sessions` distinct sessions.

        Episodes are assumed to be in chronological order (oldest first).
        We iterate from the end to find the most recent sessions, then
        return all episodes belonging to those sessions in original order.
        """
        # Collect session IDs in reverse order (most recent first)
        seen_sessions: set[str] = set()
        recent_session_ids: list[str] = []
        for ep in reversed(episodes):
            sid = ep.get("session_id", "unknown")
            if sid not in seen_sessions:
                seen_sessions.add(sid)
                recent_session_ids.append(sid)
                if len(recent_session_ids) >= max_sessions:
                    break

        session_set = set(recent_session_ids)
        return [ep for ep in episodes if ep.get("session_id", "unknown") in session_set]

    def _group_by_session(self, episodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        sessions: dict[str, list[dict]] = defaultdict(list)
        for ep in episodes:
            sessions[ep.get("session_id", "unknown")].append(ep)
        for sid in sessions:
            sessions[sid].sort(key=lambda e: e.get("timestamp", ""))
        return dict(sessions)

    def _has_existing_procedure(self, episode_ids: list[str]) -> bool:
        """Check if we already have a procedure from overlapping episodes."""
        for eid in episode_ids[:5]:
            existing = self.db.fetchone(
                "SELECT id FROM procedures WHERE source_episodes LIKE ?",
                (f"%{eid}%",),
            )
            if existing:
                return True
        return False

    def _build_procedure(
        self,
        consensus: list[AlignedStep],
        episodes: list[dict[str, Any]],
        activation: float,
        sessions: dict[str, list[dict[str, Any]]],
    ) -> Procedure | None:
        """Build a Procedure from aligned consensus steps."""
        domain = episodes[0].get("domain")
        agent_id = episodes[0].get("agent_id", "unknown")
        n_sessions = len(sessions)

        steps = []
        for aligned_step in consensus:
            step_type_str = aligned_step.step_type
            if step_type_str == "core":
                step_type = StepType.CORE
            elif step_type_str == "optional":
                step_type = StepType.OPTIONAL
            else:
                step_type = StepType.VARIANT

            step = ProcedureStep(
                order=aligned_step.position,
                description=aligned_step.primary_action,
                step_type=step_type,
                variants=aligned_step.variants[:5],
            )
            steps.append(step)

        if not steps:
            return None

        core_steps = [s for s in steps if s.step_type == StepType.CORE]
        variant_steps = [s for s in steps if s.step_type == StepType.VARIANT]
        name_parts = [s.description[:30] for s in core_steps[:3]]
        name = (
            f"auto_{'_'.join(w.split()[0].lower() for w in name_parts)}"
            if name_parts
            else f"auto_{domain or 'workflow'}"
        )

        workflow_label = f"{domain} workflow" if domain else "tasks"
        trigger = (
            f"When performing {workflow_label} involving: "
            f"{', '.join(s.description[:50] for s in core_steps[:3])}"
        )

        return Procedure(
            name=name,
            description=(
                f"Auto-promoted from {len(episodes)} episodes across {n_sessions} sessions "
                f"(activation: {activation:.2f}). "
                f"{len(core_steps)} core steps, "
                f"{len([s for s in steps if s.step_type == StepType.OPTIONAL])} optional, "
                f"{len([s for s in steps if s.step_type == StepType.VARIANT])} variant."
            ),
            trigger_pattern=trigger,
            steps=steps,
            confidence=initial_procedure_confidence(
                session_count=n_sessions,
                core_step_count=len(core_steps),
                variant_step_count=len(variant_steps),
            ),
            activation_score=activation,
            access_times=[],
            source_agent=agent_id,
            source_episodes=[ep["id"] for ep in episodes],
            domain=domain,
            status=ProcedureStatus.ACTIVE if n_sessions >= 3 else ProcedureStatus.DRAFT,
        )
