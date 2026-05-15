"""Cognitive loop orchestrator. Schedules and runs all cognitive processes.

Trigger rules:
- ReconsolidationEngine: session end + every 20 writes
- LLM Consolidator: every 50 writes or session end
- SchemaLearner: session end (after Consolidator)
- LLM Reflector: session end (after SchemaLearner)
- NREMSleep: every 50 writes OR session end (replaces old sleep)
- REMSleep: every 3rd NREM run
- PrioritizedReplay: during NREM sleep as sub-phase
- PredictionLearner: on procedure feedback
- Promoter: session end
- Composer: when a new procedure is created
- Decayer: hourly
- Challenger: on conflict detection (session end)
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
from .consolidator import Consolidator as OldConsolidator
from .decayer import Decayer
from .llm_consolidator import LLMConsolidator
from .llm_reflector import LLMReflector
from .nrem_sleep import NREMPhase
from .prediction_learner import PredictionLearner
from .prioritized_replay import PrioritizedReplay
from .promoter import Promoter
from .reconsolidator import ReconsolidationEngine
from .reflector import Reflector as OldReflector
from .rem_sleep import REMPhase
from .schema_learner import SchemaLearner

log = logging.getLogger("myelin.orchestrator")

NREM_EVERY_N_WRITES = 50
RECONSOLIDATION_CHECK_WRITES = 20
REM_EVERY_N_NREM = 3


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

        # New learning OS processes
        self.reconsolidation = ReconsolidationEngine(db, episodic, semantic, procedural)
        self.llm_consolidator = LLMConsolidator(db, episodic, semantic)
        self.llm_reflector = LLMReflector(db, semantic)
        self.schema_learner = SchemaLearner(db)
        self.nrem_sleep = NREMPhase(db)
        self.rem_sleep = REMPhase(db)
        self.prioritized_replay = PrioritizedReplay(db)
        self.prediction_learner = PredictionLearner(db, procedural)

        # Existing processes (kept)
        self.promoter = Promoter(db, episodic, procedural)
        self.composer = Composer(db, procedural)
        self.decayer = Decayer(db)
        self.challenger = Challenger(db, semantic)

        # Write tracking
        self._write_count = 0
        self._last_decay = 0.0
        self._nrem_run_count = 0

    def on_write(self) -> None:
        """Called after every episode write."""
        self._write_count += 1

    async def check_triggers(self) -> list[dict[str, Any]]:
        """Check all trigger conditions and run processes that should fire.

        Returns list of process run results.
        """
        results = []

        # Reconsolidation check (every 20 writes)
        if self._write_count > 0 and self._write_count % RECONSOLIDATION_CHECK_WRITES == 0:
            if self.reconsolidation.should_run():
                log.info("Triggering reconsolidation (periodic check)")
                result = await self.reconsolidation.run()
                results.append({"process": "reconsolidation", **result})

        # LLM Consolidator (every 50 writes)
        if self._write_count >= NREM_EVERY_N_WRITES and self.llm_consolidator.should_run():
            log.info("Triggering LLM consolidator (50+ writes)")
            result = await self.llm_consolidator.run()
            results.append({"process": "consolidator", **result})
            self._write_count = 0

        # NREM Sleep (every 50 writes OR when consolidator ran)
        if self._write_count >= NREM_EVERY_N_WRITES and self.nrem_sleep.should_run():
            log.info("Triggering NREM sleep (50+ writes)")
            # Run PrioritizedReplay as sub-phase of NREM
            log.info("Running PrioritizedReplay as NREM sub-phase")
            replay_result = await self.prioritized_replay.run()
            results.append({"process": "prioritized_replay", **replay_result})

            nrem_result = await self.nrem_sleep.run()
            results.append({"process": "nrem_sleep", **nrem_result})
            self._nrem_run_count += 1

            # REM Sleep (every 3rd NREM)
            if self._nrem_run_count % REM_EVERY_N_NREM == 0 and self.rem_sleep.should_run():
                log.info("Triggering REM sleep (3rd NREM cycle)")
                rem_result = await self.rem_sleep.run()
                results.append({"process": "rem_sleep", **rem_result})

            self._write_count = 0

        # Decayer (hourly)
        now = time.time()
        if now - self._last_decay >= 3600 and self.decayer.should_run():
            log.info("Triggering decayer (hourly)")
            result = await self.decayer.run()
            results.append({"process": "decayer", **result})
            self._last_decay = now

        return results

    async def on_session_end(self) -> list[dict[str, Any]]:
        """Run all session-end processes in the prescribed order.

        Order:
        1. ReconsolidationEngine (check labile windows + new evidence)
        2. LLM Consolidator (pattern-aware consolidation — replaces old consolidator)
        3. SchemaLearner (induce schemas from consolidated facts)
        4. LLM Reflector (multi-level reflection — replaces old reflector)
        5. NREMSleep (with PrioritizedReplay as sub-phase)
        6. REMSleep (every 3rd cycle)
        7. Promoter (existing)
        8. Composer (existing)
        9. Decayer (existing)
        10. Challenger (existing)
        """
        results = []

        processes = [
            ("reconsolidation", self.reconsolidation),
            ("consolidator", self.llm_consolidator),
            ("schema_learner", self.schema_learner),
            ("reflector", self.llm_reflector),
            ("nrem_sleep", self.nrem_sleep),
            ("promoter", self.promoter),
            ("composer", self.composer),
            ("decayer", self.decayer),
            ("challenger", self.challenger),
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

        # Handle REM sleep (every 3rd cycle) after NREM
        if self._nrem_run_count > 0 and self._nrem_run_count % REM_EVERY_N_NREM == 0:
            try:
                log.info("Running REM sleep at session end (3rd NREM cycle)")
                result = await self.rem_sleep.run()
                results.append({"process": "rem_sleep", **result})
                log.info(f"REM sleep complete: {result}")
            except Exception as e:
                log.error(f"REM sleep failed: {e}")
                results.append({"process": "rem_sleep", "error": str(e)})

        return results

    async def on_procedure_feedback(
        self, procedure_id: str, success: bool
    ) -> dict[str, Any] | None:
        """Handle procedure execution feedback — triggers PredictionLearner.

        Records the outcome against the most recent prediction for this
        procedure, or runs a full learning cycle if no prediction exists.
        """
        try:
            # Get the last pending prediction for this procedure
            last_pred = self.db.fetchone(
                "SELECT id FROM prediction_log "
                "WHERE procedure_id = ? AND actual_outcome IS NULL "
                "ORDER BY timestamp DESC LIMIT 1",
                (procedure_id,),
            )
            if last_pred:
                result = self.prediction_learner.record_outcome(
                    prediction_id=last_pred["id"],
                    actual_success=success,
                )
            else:
                # No pending prediction — do a learning update on the procedure directly
                proc = self.procedural.get(procedure_id)
                if proc:
                    from ..core.activation import bayesian_confidence_update

                    new_conf = bayesian_confidence_update(
                        proc["confidence"], success, learning_rate=0.15
                    )
                    self.db.update(
                        "procedures",
                        procedure_id,
                        {
                            "confidence": new_conf,
                            "success_count": proc["success_count"] + (1 if success else 0),
                            "failure_count": proc["failure_count"] + (0 if success else 1),
                            "last_executed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        },
                    )
                    result = {
                        "procedure_id": procedure_id,
                        "success": success,
                        "new_confidence": new_conf,
                        "note": "no pending prediction found, updated confidence directly",
                    }
                else:
                    result = {"error": f"Procedure {procedure_id} not found"}

            return {"process": "prediction_learner", **result}
        except Exception as e:
            log.error(f"PredictionLearner failed: {e}")
            return {"process": "prediction_learner", "error": str(e)}

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
            "reconsolidation",
            "consolidator",
            "schema_learner",
            "reflector",
            "nrem_sleep",
            "rem_sleep",
            "prioritized_replay",
            "prediction_learner",
            "promoter",
            "composer",
            "decayer",
            "challenger",
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
            "nrem_run_count": self._nrem_run_count,
            "processes": last_runs,
        }
