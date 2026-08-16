"""Decayer: reduce salience of unused memories using Ebbinghaus forgetting curve.

Trigger: hourly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..core.activation import ebbinghaus_decay
from ..core.database import Database
from ..core.models import ProcedureStatus, ProcessName
from .base import CognitiveProcess

ARCHIVE_THRESHOLD = 0.1
DECAY_STABILITY_BASE = 24.0
GRACE_HOURS = 72.0  # recently executed → awaiting feedback; skip decay entirely
UNVALIDATED_FLOOR = 0.2  # executed but never feedbacked → decay but never archive below this


class Decayer(CognitiveProcess):
    name = ProcessName.DECAYER

    def __init__(self, db: Database):
        super().__init__(db)

    def should_run(self) -> bool:
        last_run = self.db.fetchone(
            "SELECT completed_at FROM process_runs "
            "WHERE process_name = ? AND status = 'completed' "
            "ORDER BY completed_at DESC LIMIT 1",
            (ProcessName.DECAYER.value,),
        )
        if not last_run or not last_run["completed_at"]:
            return True
        last = datetime.fromisoformat(last_run["completed_at"])
        hours_since = (datetime.utcnow() - last).total_seconds() / 3600
        return hours_since >= 1.0

    async def execute(self) -> dict[str, Any]:
        modified = 0
        archived = 0

        procedures = self.db.fetchall(
            "SELECT id, confidence, last_executed, success_count, failure_count FROM procedures "
            "WHERE status IN (?, ?)",
            (ProcedureStatus.ACTIVE.value, ProcedureStatus.DRAFT.value),
        )

        for proc in procedures:
            last_exec = proc.get("last_executed")
            if not last_exec:
                continue

            try:
                last_dt = datetime.fromisoformat(last_exec)
                hours = (datetime.utcnow() - last_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                continue

            # Grace period: recently executed procedures are awaiting feedback.
            # Decaying them before feedback arrives cratered confidence —
            # 0.6 → ~0.03 in 3 days at stability 24h, then archived at <0.1.
            if hours < GRACE_HOURS:
                continue

            successes = proc.get("success_count") or 0
            # A procedure is only validated once it has accumulated enough
            # evidence (>= 3 successes — the TRUSTED threshold). One recorded
            # execution is NOT validation: it is a candidate awaiting feedback.
            is_validated = successes >= 3
            stability = DECAY_STABILITY_BASE * max(1, successes)
            new_confidence = ebbinghaus_decay(proc["confidence"], hours, stability)

            # Unvalidated candidates: decay salience but never archive — they
            # are awaiting validation, not dead procedures. Archiving them
            # removed them from the read path entirely and re-created the
            # empty-context deadlock (all 331 procedures were seed/candidate).
            if not is_validated:
                new_confidence = max(new_confidence, UNVALIDATED_FLOOR)

            if abs(new_confidence - proc["confidence"]) > 0.001:
                updates: dict[str, Any] = {"confidence": new_confidence}
                if new_confidence < ARCHIVE_THRESHOLD and is_validated:
                    updates["status"] = ProcedureStatus.ARCHIVED.value
                    archived += 1
                self.db.update("procedures", proc["id"], updates)
                modified += 1

        return {"processed": len(procedures), "modified": modified, "archived": archived}
