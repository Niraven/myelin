"""Procedural memory: learned workflows, composition, and transfer."""

from __future__ import annotations

import json
import time
from typing import Any

from ..core.activation import (
    base_level_activation,
    bayesian_confidence_update,
)
from ..core.database import Database
from ..core.models import Procedure, ProcedureStatus, PromotionMethod


class ProceduralMemory:
    def __init__(self, db: Database):
        self.db = db

    def store(self, procedure: Procedure) -> str:
        data = procedure.model_dump()
        data["steps"] = json.dumps([s.model_dump() for s in procedure.steps])
        if data.get("trigger_embedding"):
            from ..core.database import _serialize_f32

            data["trigger_embedding"] = _serialize_f32(data["trigger_embedding"])
        self.db.insert("procedures", data)
        return procedure.id

    def get(self, procedure_id: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM procedures WHERE id = ?", (procedure_id,))
        if row:
            row = dict(row)
            for field in (
                "steps",
                "preconditions",
                "postconditions",
                "source_episodes",
                "component_procedures",
                "parent_procedures",
                "transferred_to",
                "access_times",
                "tags",
            ):
                if isinstance(row.get(field), str):
                    row[field] = json.loads(row[field])
        return row

    def find_matching(
        self,
        text_query: str,
        query_vec: list[float] | None = None,
        limit: int = 5,
        min_confidence: float = 0.3,
        agent_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Find procedures matching a trigger pattern."""
        where = None
        where_params: tuple[Any, ...] = ()
        if agent_ids and agent_ids != ["*"]:
            placeholders = ",".join("?" for _ in agent_ids)
            where = f"source_agent IN ({placeholders})"
            where_params = tuple(agent_ids)
        results = self.db.hybrid_search(
            "procedures", "procedures_fts", text_query, query_vec, limit=limit * 2,
            embedding_col="trigger_embedding", where=where, where_params=where_params,
        )
        filtered = [
            r
            for r in results
            if r.get("confidence", 0) >= min_confidence
            and r.get("status") in (ProcedureStatus.ACTIVE.value, ProcedureStatus.REFLEXIVE.value)
        ]
        for r in filtered:
            for field in ("steps", "preconditions", "postconditions"):
                if isinstance(r.get(field), str):
                    r[field] = json.loads(r[field])
        return filtered[:limit]

    def record_execution(self, procedure_id: str, success: bool) -> float:
        """Record a procedure execution outcome. Returns updated confidence."""
        proc = self.get(procedure_id)
        if not proc:
            return 0.0

        new_confidence = bayesian_confidence_update(proc["confidence"], success)

        access_times = proc["access_times"]
        access_times.append(time.time())
        new_activation = base_level_activation(access_times)

        success_count = proc["success_count"] + (1 if success else 0)
        failure_count = proc["failure_count"] + (0 if success else 1)
        total = success_count + failure_count
        actual_rate = success_count / total if total > 0 else None

        self.db.update(
            "procedures",
            procedure_id,
            {
                "confidence": new_confidence,
                "success_count": success_count,
                "failure_count": failure_count,
                "activation_score": new_activation,
                "access_times": access_times,
                "actual_success_rate": actual_rate,
                "predicted_success_rate": new_confidence,
                "calibration_offset": (new_confidence - actual_rate)
                if actual_rate is not None
                else 0.0,
                "last_executed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        if new_confidence >= 0.8 and proc["status"] == ProcedureStatus.DRAFT.value:
            self.db.update(
                "procedures",
                procedure_id,
                {
                    "status": ProcedureStatus.ACTIVE.value,
                },
            )

        return new_confidence

    def record_modification(self, procedure_id: str, new_steps: list[dict]) -> None:
        """Record that a user modified a procedure's steps."""
        proc = self.get(procedure_id)
        if not proc:
            return
        self.db.update(
            "procedures",
            procedure_id,
            {
                "steps": new_steps,
                "modify_count": proc["modify_count"] + 1,
                "version": proc["version"] + 1,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def get_composable_pairs(self) -> list[tuple[dict, dict]]:
        """Find procedure pairs where A's postconditions match B's preconditions."""
        active = self.db.fetchall(
            "SELECT * FROM procedures WHERE status IN (?, ?) AND is_composite = 0",
            (ProcedureStatus.ACTIVE.value, ProcedureStatus.REFLEXIVE.value),
        )

        pairs = []
        for a in active:
            post_a = (
                json.loads(a["postconditions"])
                if isinstance(a["postconditions"], str)
                else a["postconditions"]
            )
            if not post_a:
                continue
            for b in active:
                if a["id"] == b["id"]:
                    continue
                pre_b = (
                    json.loads(b["preconditions"])
                    if isinstance(b["preconditions"], str)
                    else b["preconditions"]
                )
                if not pre_b:
                    continue
                overlap = set(post_a) & set(pre_b)
                if overlap:
                    pairs.append((dict(a), dict(b)))
        return pairs

    def create_composite(
        self,
        name: str,
        components: list[str],
        trigger_pattern: str,
        source_agent: str,
    ) -> str:
        """Create a composite (meta) procedure from component procedure IDs."""
        all_steps: list[dict[str, Any]] = []
        min_confidence = 1.0
        all_preconditions: list[str] = []
        all_postconditions: list[str] = []

        for i, comp_id in enumerate(components):
            comp = self.get(comp_id)
            if not comp:
                continue
            steps = comp["steps"] if isinstance(comp["steps"], list) else json.loads(comp["steps"])
            all_steps.extend(steps)
            min_confidence = min(min_confidence, comp["confidence"])
            if i == 0:
                pre = (
                    comp["preconditions"]
                    if isinstance(comp["preconditions"], list)
                    else json.loads(comp["preconditions"])
                )
                all_preconditions = pre
            if i == len(components) - 1:
                post = (
                    comp["postconditions"]
                    if isinstance(comp["postconditions"], list)
                    else json.loads(comp["postconditions"])
                )
                all_postconditions = post

        from ..core.models import ProcedureStep

        procedure = Procedure(
            name=name,
            trigger_pattern=trigger_pattern,
            steps=[ProcedureStep(**s) if isinstance(s, dict) else s for s in all_steps],
            preconditions=all_preconditions,
            postconditions=all_postconditions,
            confidence=min_confidence,
            source_agent=source_agent,
            promotion_method=PromotionMethod.COMPOSED,
            is_composite=True,
            component_procedures=components,
            status=ProcedureStatus.DRAFT,
        )
        self.store(procedure)

        for comp_id in components:
            comp = self.get(comp_id)
            if comp:
                parents = comp.get("parent_procedures", [])
                parents.append(procedure.id)
                self.db.update("procedures", comp_id, {"parent_procedures": parents})

        return procedure.id

    def get_by_domain(self, domain: str, min_confidence: float = 0.0) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM procedures WHERE domain = ? AND confidence >= ? ORDER BY confidence DESC",
            (domain, min_confidence),
        )

    def get_active(self) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM procedures WHERE status IN (?, ?) ORDER BY activation_score DESC",
            (ProcedureStatus.ACTIVE.value, ProcedureStatus.REFLEXIVE.value),
        )

    def archive(self, procedure_id: str) -> None:
        self.db.update(
            "procedures",
            procedure_id,
            {
                "status": ProcedureStatus.ARCHIVED.value,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def count(self, status: ProcedureStatus | None = None) -> int:
        if status:
            row = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM procedures WHERE status = ?",
                (status.value,),
            )
        else:
            row = self.db.fetchone("SELECT COUNT(*) as cnt FROM procedures")
        return row["cnt"] if row else 0
