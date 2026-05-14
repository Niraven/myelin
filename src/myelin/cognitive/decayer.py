"""Decayer: reduce salience of unused memories using Ebbinghaus forgetting curve.

Trigger: hourly.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..core.activation import ebbinghaus_decay
from ..core.database import Database
from ..core.models import ProcessName, ProcedureStatus
from .base import CognitiveProcess

ARCHIVE_THRESHOLD = 0.1
DECAY_STABILITY_BASE = 24.0


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
            "SELECT id, confidence, last_executed, success_count FROM procedures "
            "WHERE status IN (?, ?)",
            (ProcedureStatus.ACTIVE.value, ProcedureStatus.DRAFT.value),
        )

        now = time.time()
        for proc in procedures:
            last_exec = proc.get("last_executed")
            if not last_exec:
                continue

            try:
                last_dt = datetime.fromisoformat(last_exec)
                hours = (datetime.utcnow() - last_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                continue

            stability = DECAY_STABILITY_BASE * max(1, proc.get("success_count", 0))
            new_confidence = ebbinghaus_decay(proc["confidence"], hours, stability)

            if abs(new_confidence - proc["confidence"]) > 0.001:
                updates: dict[str, Any] = {"confidence": new_confidence}
                if new_confidence < ARCHIVE_THRESHOLD:
                    updates["status"] = ProcedureStatus.ARCHIVED.value
                    archived += 1
                self.db.update("procedures", proc["id"], updates)
                modified += 1

        return {"processed": len(procedures), "modified": modified, "archived": archived}
