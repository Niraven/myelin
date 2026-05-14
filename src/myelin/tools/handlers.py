"""MCP tool handlers for all 7 Myelin tools."""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.models import (
    ActionType,
    Episode,
    GoalStatus,
    LearningGoal,
    NodeType,
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    SemanticNode,
    SourceType,
    StepType,
)
from ..memory.embedding import EmbeddingProvider
from ..memory.episodic import EpisodicMemory
from ..memory.procedural import ProceduralMemory
from ..memory.semantic import SemanticMemory


class ToolHandlers:
    """Implements all 7 MCP tool operations."""

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        embedder: EmbeddingProvider,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.embedder = embedder

    # ── myelin_observe ─────────────────────────────────────────

    async def observe(
        self,
        agent_id: str,
        session_id: str,
        action: str,
        action_type: str,
        content_text: str,
        input_context: dict | None = None,
        output_result: dict | None = None,
        success: bool = True,
        domain: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record an agent action as an episodic memory."""
        embedding = self.embedder.embed(content_text) or None

        episode = Episode(
            agent_id=agent_id,
            session_id=session_id,
            action=action,
            action_type=ActionType(action_type),
            content_text=content_text,
            input_context=input_context,
            output_result=output_result,
            success=success,
            embedding=embedding,
            domain=domain,
            tags=tags or [],
        )

        episode_id = self.episodic.record(episode)

        self._check_learning_goals(domain, agent_id)

        return {
            "episode_id": episode_id,
            "status": "recorded",
            "total_episodes": self.episodic.count(agent_id),
        }

    # ── myelin_recall ──────────────────────────────────────────

    async def recall(
        self,
        query: str,
        limit: int = 10,
        memory_types: list[str] | None = None,
        domain: str | None = None,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        """Search across all memory types for relevant knowledge."""
        types = memory_types or ["episodic", "semantic", "procedural"]
        query_vec = self.embedder.embed(query) or None
        results: dict[str, list] = {}

        if "episodic" in types:
            episodes = self.episodic.search_hybrid(query, query_vec, limit=limit)
            for ep in episodes:
                self.episodic.access(ep["id"])
            results["episodes"] = episodes

        if "semantic" in types:
            nodes = self.semantic.search_hybrid(query, query_vec, limit=limit)
            for node in nodes:
                self.semantic.access(node["id"])
            results["semantic"] = [
                n for n in nodes if n.get("confidence", 0) >= min_confidence
            ]

        if "procedural" in types:
            procedures = self.procedural.find_matching(
                query, query_vec, limit=limit, min_confidence=min_confidence
            )
            results["procedures"] = procedures

        return {
            "query": query,
            "results": results,
            "total_results": sum(len(v) for v in results.values()),
        }

    # ── myelin_execute_procedure ───────────────────────────────

    async def execute_procedure(
        self,
        query: str,
        agent_id: str,
        context: dict | None = None,
    ) -> dict[str, Any]:
        """Find and return the best matching procedure for a trigger."""
        query_vec = self.embedder.embed(query) or None
        matches = self.procedural.find_matching(query, query_vec, limit=3)

        if not matches:
            return {
                "found": False,
                "message": "No matching procedure found.",
                "suggestion": "Try myelin_recall to search broader memory.",
            }

        best = matches[0]
        steps = best["steps"] if isinstance(best["steps"], list) else json.loads(best["steps"])

        return {
            "found": True,
            "procedure_id": best["id"],
            "name": best["name"],
            "confidence": best["confidence"],
            "calibration_offset": best.get("calibration_offset", 0.0),
            "steps": steps,
            "preconditions": best.get("preconditions", []),
            "alternatives": [
                {"id": m["id"], "name": m["name"], "confidence": m["confidence"]}
                for m in matches[1:]
            ],
        }

    # ── myelin_procedure_feedback ──────────────────────────────

    async def procedure_feedback(
        self,
        procedure_id: str,
        success: bool,
        modifications: list[dict] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Report execution outcome for a procedure."""
        new_confidence = self.procedural.record_execution(procedure_id, success)

        if modifications:
            self.procedural.record_modification(procedure_id, modifications)

        proc = self.procedural.get(procedure_id)
        return {
            "procedure_id": procedure_id,
            "new_confidence": new_confidence,
            "success_count": proc["success_count"] if proc else 0,
            "failure_count": proc["failure_count"] if proc else 0,
            "status": proc["status"] if proc else "unknown",
        }

    # ── myelin_confidence ──────────────────────────────────────

    async def confidence(
        self,
        domain: str | None = None,
        procedure_id: str | None = None,
    ) -> dict[str, Any]:
        """Query confidence levels across domains or for a specific procedure."""
        result: dict[str, Any] = {}

        if procedure_id:
            proc = self.procedural.get(procedure_id)
            if proc:
                result["procedure"] = {
                    "id": proc["id"],
                    "name": proc["name"],
                    "confidence": proc["confidence"],
                    "actual_success_rate": proc.get("actual_success_rate"),
                    "calibration_offset": proc.get("calibration_offset", 0.0),
                    "executions": proc["success_count"] + proc["failure_count"],
                }

        if domain:
            from ..core.database import Database
            db = self.episodic.db
            row = db.fetchone(
                "SELECT * FROM confidence_map WHERE domain = ?", (domain,)
            )
            if row:
                result["domain"] = dict(row)

            procedures = self.procedural.get_by_domain(domain)
            result["domain_procedures"] = len(procedures)

        if not domain and not procedure_id:
            db = self.episodic.db
            domains = db.fetchall(
                "SELECT * FROM confidence_map ORDER BY confidence DESC LIMIT 20"
            )
            result["all_domains"] = [dict(d) for d in domains]
            result["total_procedures"] = self.procedural.count()
            result["active_procedures"] = self.procedural.count(ProcedureStatus.ACTIVE)

        return result

    # ── myelin_teach ───────────────────────────────────────────

    async def teach(
        self,
        name: str,
        trigger_pattern: str,
        steps: list[dict],
        agent_id: str,
        description: str | None = None,
        preconditions: list[str] | None = None,
        postconditions: list[str] | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Manually teach a procedure."""
        trigger_embedding = self.embedder.embed(trigger_pattern) or None

        procedure = Procedure(
            name=name,
            description=description,
            trigger_pattern=trigger_pattern,
            trigger_embedding=trigger_embedding,
            steps=[
                ProcedureStep(
                    order=i,
                    description=s.get("description", s.get("step", "")),
                    step_type=StepType(s.get("type", "core")),
                    variants=s.get("variants", []),
                    condition=s.get("condition"),
                )
                for i, s in enumerate(steps)
            ],
            preconditions=preconditions or [],
            postconditions=postconditions or [],
            confidence=0.7,
            source_agent=agent_id,
            promotion_method="taught",
            status=ProcedureStatus.ACTIVE,
            domain=domain,
        )

        proc_id = self.procedural.store(procedure)

        return {
            "procedure_id": proc_id,
            "name": name,
            "status": "taught",
            "confidence": 0.7,
            "steps_count": len(steps),
        }

    # ── myelin_status ──────────────────────────────────────────

    async def status(self, agent_id: str | None = None) -> dict[str, Any]:
        """Get overall Myelin system status."""
        db = self.episodic.db

        total_episodes = self.episodic.count(agent_id)
        total_semantic = self.semantic.count()
        total_procedures = self.procedural.count()
        active_procedures = self.procedural.count(ProcedureStatus.ACTIVE)

        goals = db.fetchall(
            "SELECT * FROM learning_goals WHERE status = ?",
            (GoalStatus.ACTIVE.value,),
        )

        last_run = db.fetchone(
            "SELECT * FROM process_runs ORDER BY started_at DESC LIMIT 1"
        )

        return {
            "episodes": total_episodes,
            "semantic_nodes": total_semantic,
            "procedures": {
                "total": total_procedures,
                "active": active_procedures,
                "draft": self.procedural.count(ProcedureStatus.DRAFT),
                "archived": self.procedural.count(ProcedureStatus.ARCHIVED),
            },
            "learning_goals": len(goals),
            "last_cognitive_process": dict(last_run) if last_run else None,
        }

    # ── Internal helpers ───────────────────────────────────────

    def _check_learning_goals(self, domain: str | None, agent_id: str) -> None:
        """Check if this episode contributes to any active learning goal."""
        if not domain:
            return
        db = self.episodic.db
        goals = db.fetchall(
            "SELECT * FROM learning_goals WHERE domain = ? AND status = ?",
            (domain, GoalStatus.ACTIVE.value),
        )
        for goal in goals:
            new_count = goal["episodes_collected"] + 1
            updates: dict[str, Any] = {"episodes_collected": new_count}
            if new_count >= goal["episodes_needed"]:
                updates["status"] = GoalStatus.ACHIEVED.value
                updates["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            db.update("learning_goals", goal["id"], updates)
