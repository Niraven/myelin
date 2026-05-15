"""Cross-agent knowledge transfer protocol.

Handles the full transfer lifecycle: packaging a procedure from the source
agent, adapting it for the target agent's capabilities, and importing it
with discounted confidence based on adaptation complexity.

Differentiator: no other memory system does real cross-agent transfer.
mem0 is single-agent. hermes-lcm is single-agent. Myelin treats
procedures as portable, adaptable units of knowledge.
"""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from ..core.activation import transfer_confidence
from ..core.database import Database
from ..core.models import (
    Procedure,
    ProcedureStatus,
    ProcedureStep,
    PromotionMethod,
    StepType,
)
from ..memory.procedural import ProceduralMemory
from .adaptation import StepAdaptationEngine
from .profiling import AgentProfiler


class TransferProtocol:
    """Cross-agent knowledge sharing with capability-aware adaptation."""

    def __init__(self, db: Database, procedural: ProceduralMemory):
        self.db = db
        self.procedural = procedural
        self.profiler = AgentProfiler(db)
        self.adaptation_engine = StepAdaptationEngine()

    def export_procedure(
        self,
        procedure_id: str,
        source_agent: str,
        target_agent: str,
    ) -> dict[str, Any]:
        """Package a procedure for transfer to another agent.

        Returns a self-contained transfer package with the procedure,
        source agent profile, capability analysis, and computed transfer confidence.
        """
        proc = self.procedural.get(procedure_id)
        if not proc:
            return {"success": False, "error": "Procedure not found"}

        source_profile = self.profiler.get(source_agent)
        target_profile = self.profiler.get(target_agent)

        similarity = self.profiler.compute_similarity(source_agent, target_agent)
        t_confidence = transfer_confidence(proc["confidence"], similarity)

        steps = proc["steps"]
        if isinstance(steps, str):
            steps = json.loads(steps)

        # Normalize steps
        normalized_steps = [_normalize_step(step) for step in steps]

        # Capability analysis
        required_tools = self.adaptation_engine.analyze_requirements(
            {"steps": normalized_steps}
        )

        if not target_profile:
            adapted_steps = normalized_steps
            adaptation_notes = ["No target profile available, using original steps"]
            adaptation_results = []
        else:
            target_capabilities = self.profiler.get_toolset(target_agent, min_usage=1)
            target_tool_names = [c.tool_name for c in target_capabilities]

            if not target_tool_names:
                adapted_steps = normalized_steps
                adaptation_notes = ["Target agent tools unknown, using original steps"]
                adaptation_results = []
            else:
                adapted_steps, adaptation_results, adaptation_notes = (
                    self.adaptation_engine.adapt_procedure(
                        normalized_steps, target_tool_names
                    )
                )

        # Compute adaptation summary
        changed_count = sum(1 for r in adaptation_results if r.changed)
        flagged_count = sum(1 for r in adaptation_results if r.flag)
        unchanged_count = len(adaptation_results) - changed_count - flagged_count

        package = {
            "transfer_id": uuid4().hex[:16],
            "procedure_id": procedure_id,
            "procedure_name": proc["name"],
            "description": proc.get("description", ""),
            "trigger_pattern": proc["trigger_pattern"],
            "original_steps": normalized_steps,
            "adapted_steps": adapted_steps,
            "preconditions": _parse_json(proc.get("preconditions", "[]")),
            "postconditions": _parse_json(proc.get("postconditions", "[]")),
            "source_agent": source_agent,
            "target_agent": target_agent,
            "source_confidence": proc["confidence"],
            "transfer_confidence": t_confidence,
            "agent_similarity": similarity,
            "adaptation_notes": adaptation_notes,
            "capability_analysis": {
                "required_tools": required_tools,
                "target_tools_available": [
                    c.tool_name for c in self.profiler.get_toolset(target_agent)
                ]
                if target_profile
                else [],
                "steps_changed": changed_count,
                "steps_flagged": flagged_count,
                "steps_unchanged": unchanged_count,
            },
            "domain": proc.get("domain"),
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "success": True,
        }

        return package

    def import_procedure(
        self,
        package: dict[str, Any],
        agent_id: str,
    ) -> dict[str, Any]:
        """Import a transferred procedure, adapting and storing it.

        Creates a new procedure owned by the target agent with discounted
        confidence based on adaptation complexity and tracks the transfer
        in the transfer log.
        """
        steps = package.get("adapted_steps") or package.get("original_steps", [])

        # Rebuild adaptation results from package for confidence calibration
        analysis = package.get("capability_analysis", {})
        changed = analysis.get("steps_changed", 0)
        flagged = analysis.get("steps_flagged", 0)
        total = changed + flagged + analysis.get("steps_unchanged", len(steps))

        base_confidence = package.get("transfer_confidence", 0.3)

        # Apply confidence discount
        if flagged > 0:
            discount = 0.4
            status = ProcedureStatus.DRAFT
        elif changed == 0:
            discount = 1.0
            status = ProcedureStatus.ACTIVE
        elif changed <= 2:
            discount = 0.8
            status = ProcedureStatus.DRAFT
        else:
            discount = 0.6
            status = ProcedureStatus.DRAFT

        final_confidence = base_confidence * discount

        procedure = Procedure(
            name=package["procedure_name"],
            description=package.get("description"),
            trigger_pattern=package["trigger_pattern"],
            steps=[
                ProcedureStep(
                    order=i,
                    description=s.get("description", s) if isinstance(s, dict) else str(s),
                    step_type=StepType(s.get("type", "core"))
                    if isinstance(s, dict)
                    else StepType.CORE,
                    variants=s.get("variants", []) if isinstance(s, dict) else [],
                    condition=s.get("condition") if isinstance(s, dict) else None,
                )
                for i, s in enumerate(steps)
            ],
            preconditions=package.get("preconditions", []),
            postconditions=package.get("postconditions", []),
            confidence=final_confidence,
            source_agent=agent_id,
            promotion_method=PromotionMethod.TRANSFERRED,
            status=status,
            domain=package.get("domain"),
            source_episodes=[],
        )

        proc_id = self.procedural.store(procedure)

        self.profiler.record_transfer(
            procedure_id=package["procedure_id"],
            source_agent=package["source_agent"],
            target_agent=agent_id,
            source_confidence=package.get("source_confidence", 0.5),
            adapted=changed > 0 or flagged > 0,
            adaptation_details={
                "steps_changed": changed,
                "steps_flagged": flagged,
                "discount": discount,
                "final_confidence": final_confidence,
            },
        )

        return {
            "success": True,
            "new_procedure_id": proc_id,
            "name": package["procedure_name"],
            "transfer_confidence": final_confidence,
            "base_confidence": base_confidence,
            "discount": discount,
            "status": status.value,
            "adaptation_notes": package.get("adaptation_notes", []),
            "review_needed": flagged > 0,
        }

    def get_transferable_procedures(
        self,
        source_agent: str,
        target_agent: str,
        min_confidence: float = 0.6,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find procedures from source_agent that could be useful for target_agent.

        Returns procedures above min_confidence that the target hasn't received yet.
        """
        source_procs = self.db.fetchall(
            "SELECT * FROM procedures WHERE source_agent = ? AND confidence >= ? "
            "AND status IN ('active', 'reflexive') ORDER BY confidence DESC LIMIT ?",
            (source_agent, min_confidence, limit * 2),
        )

        already_transferred = set()
        transfers = self.db.fetchall(
            "SELECT procedure_id FROM transfer_log WHERE source_agent = ? AND target_agent = ?",
            (source_agent, target_agent),
        )
        for t in transfers:
            already_transferred.add(t["procedure_id"])

        similarity = self.profiler.compute_similarity(source_agent, target_agent)

        results = []
        for proc in source_procs:
            if proc["id"] in already_transferred:
                continue

            t_conf = transfer_confidence(proc["confidence"], similarity)
            results.append(
                {
                    "procedure_id": proc["id"],
                    "name": proc["name"],
                    "domain": proc.get("domain"),
                    "source_confidence": proc["confidence"],
                    "transfer_confidence": t_conf,
                    "agent_similarity": similarity,
                }
            )
            if len(results) >= limit:
                break

        return results

    def get_transfer_history(self, agent_id: str, direction: str = "both") -> list[dict[str, Any]]:
        """Get transfer history for an agent.

        direction: 'sent', 'received', or 'both'
        """
        results = []

        if direction in ("sent", "both"):
            sent = self.db.fetchall(
                "SELECT * FROM transfer_log WHERE source_agent = ? "
                "ORDER BY timestamp DESC LIMIT 50",
                (agent_id,),
            )
            for s in sent:
                results.append({**dict(s), "direction": "sent"})

        if direction in ("received", "both"):
            received = self.db.fetchall(
                "SELECT * FROM transfer_log WHERE target_agent = ? "
                "ORDER BY timestamp DESC LIMIT 50",
                (agent_id,),
            )
            for r in received:
                results.append({**dict(r), "direction": "received"})

        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results


def _normalize_step(step: dict | str) -> dict[str, Any]:
    if isinstance(step, str):
        return {"description": step, "type": "core"}
    return dict(step)


def _parse_json(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return list(parsed) if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []
