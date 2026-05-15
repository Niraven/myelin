"""Composer: chain compatible procedures into meta-procedures.

Inspired by SOAR's chunking mechanism.
Trigger: when a new procedure is created.
"""

from __future__ import annotations

from typing import Any

from ..core.database import Database
from ..core.models import ProcessName
from ..memory.procedural import ProceduralMemory
from .base import CognitiveProcess


class Composer(CognitiveProcess):
    name = ProcessName.COMPOSER

    def __init__(self, db: Database, procedural: ProceduralMemory):
        super().__init__(db)
        self.procedural = procedural

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        """Find composable procedure pairs and create meta-procedures."""
        pairs = self.procedural.get_composable_pairs()
        created = 0

        for proc_a, proc_b in pairs:
            existing = self.db.fetchone(
                "SELECT id FROM procedures WHERE is_composite = 1 "
                "AND component_procedures LIKE ? AND component_procedures LIKE ?",
                (f"%{proc_a['id']}%", f"%{proc_b['id']}%"),
            )
            if existing:
                continue

            name = f"{proc_a['name']}_then_{proc_b['name']}"
            trigger = f"{proc_a.get('trigger_pattern', '')} followed by {proc_b.get('trigger_pattern', '')}"

            self.procedural.create_composite(
                name=name,
                components=[proc_a["id"], proc_b["id"]],
                trigger_pattern=trigger,
                source_agent=proc_a.get("source_agent", "system"),
            )
            created += 1

        return {"processed": len(pairs), "created": created}
