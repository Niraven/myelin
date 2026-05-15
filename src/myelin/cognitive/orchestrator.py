"""Cognitive loop orchestrator. Schedules and runs all 6 background processes.

Trigger rules:
- Consolidator: every 50 writes or session end
- Reflector: session end
- Promoter: session end
- Composer: when a new procedure is created
- Decayer: hourly
- Challenger: on conflict detection (session end for now)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.database import Database
from ..memory.episodic import EpisodicMemory
from ..memory.procedural import ProceduralMemory
from ..memory.semantic import SemanticMemory
from .challenger import Challenger
from .composer import Composer
from .consolidator import Consolidator
from .decayer import Decayer
from .promoter import Promoter
from .reflector import Reflector
from .sleep import SleepCycle

log = logging.getLogger("myelin.orchestrator")


class CognitiveOrchestrator:
    """Manages the lifecycle and triggering of all cognitive processes."""

    def __init__(
        self,
        db: Database,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
    ):
        self.db = db
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

        self.consolidator = Consolidator(db, episodic, semantic)
        self.reflector = Reflector(db, semantic)
        self.promoter = Promoter(db, episodic, procedural)
        self.composer = Composer(db, procedural)
        self.decayer = Decayer(db)
        self.challenger = Challenger(db, semantic)
        self.sleep = SleepCycle(db)

        self._write_count = 0
        self._last_decay = 0.0

    def on_write(self) -> None:
        """Called after every episode write. Triggers consolidation if needed."""
        self._write_count += 1

    async def check_triggers(self) -> list[dict[str, Any]]:
        """Check all trigger conditions and run processes that should fire.

        Returns list of process run results.
        """
        results = []

        if self._write_count >= 50 and self.consolidator.should_run():
            log.info("Triggering consolidator (50+ writes)")
            result = await self.consolidator.run()
            results.append({"process": "consolidator", **result})
            self._write_count = 0

        now = time.time()
        if now - self._last_decay >= 3600 and self.decayer.should_run():
            log.info("Triggering decayer (hourly)")
            result = await self.decayer.run()
            results.append({"process": "decayer", **result})
            self._last_decay = now

        return results

    async def on_session_end(self) -> list[dict[str, Any]]:
        """Run all session-end processes in sequence.

        Order matters:
        1. Consolidator (merge episodes into semantic nodes)
        2. Reflector (generate insights from semantic nodes)
        3. Promoter (promote clusters to procedures)
        4. Composer (chain compatible procedures)
        5. Challenger (test stale beliefs)
        """
        results = []

        processes = [
            ("consolidator", self.consolidator),
            ("reflector", self.reflector),
            ("promoter", self.promoter),
            ("composer", self.composer),
            ("challenger", self.challenger),
            ("sleep", self.sleep),
        ]

        for name, process in processes:
            try:
                log.info(f"Running {name} at session end")
                result = await process.run()
                results.append({"process": name, **result})
                log.info(f"{name} complete: {result}")
            except Exception as e:
                log.error(f"{name} failed: {e}")
                results.append({"process": name, "error": str(e)})

        return results

    async def on_procedure_created(self) -> dict[str, Any] | None:
        """Run composer after a new procedure is created."""
        try:
            result = await self.composer.run()
            return {"process": "composer", **result}
        except Exception as e:
            log.error(f"Composer failed: {e}")
            return {"process": "composer", "error": str(e)}

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        last_runs = {}
        for process_name in [
            "consolidator",
            "reflector",
            "promoter",
            "composer",
            "decayer",
            "challenger",
            "sleep",
        ]:
            row = self.db.fetchone(
                "SELECT * FROM process_runs WHERE process_name = ? ORDER BY started_at DESC LIMIT 1",
                (process_name,),
            )
            if row:
                last_runs[process_name] = {
                    "last_run": row["started_at"],
                    "status": row["status"],
                    "items_processed": row["items_processed"],
                    "items_created": row["items_created"],
                }

        return {
            "write_count_since_consolidation": self._write_count,
            "processes": last_runs,
        }
