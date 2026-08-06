"""Prediction error learning: forward models, TD-error, surprise, and priority updates.

Implements Sutton & Barto TD learning adapted for procedure-level predictions:
  - Forward model predicts outcomes from procedure confidence
  - TD-error measures prediction-accuracy gap
  - Surprise signals high-confidence errors
  - TD-modulated learning rates for Bayesian confidence updates
  - Episode priority scored from TD-error, surprise, and importance
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from ..core.activation import bayesian_confidence_update
from ..core.database import Database
from ..memory.procedural import ProceduralMemory


def _new_id() -> str:
    return uuid4().hex[:16]


def compute_td_error(predicted_success: int, actual_outcome: int) -> float:
    """Temporal Difference error: δ = actual - predicted.

    Args:
        predicted_success: 1 if success predicted, 0 otherwise.
        actual_outcome: 1 if actual success, 0 otherwise.

    Returns:
        δ in {-1.0, 0.0, 1.0}.
    """
    return float(actual_outcome - predicted_success)


def compute_surprise(td_error: float, predicted_confidence: float) -> float:
    """Surprise = |δ| * (1 - predicted_confidence), normalized to [0.0, 1.0].

    High surprise means the model was confident AND wrong.
    Zero surprise when prediction matches outcome (δ = 0).
    """
    abs_delta = abs(td_error)
    if abs_delta == 0.0:
        return 0.0
    raw = abs_delta * (1.0 - predicted_confidence)
    return max(0.0, min(1.0, raw))


def td_modulated_learning_rate(base_lr: float = 0.15, abs_td_error: float = 0.0) -> float:
    """Learning rate modulated by TD-error magnitude.

    Bigger errors → bigger updates.
    lr = base_lr * (1 + |δ|)
    """
    return base_lr * (1.0 + abs_td_error)


def compute_priority_score(
    td_error: float,
    surprise: float,
    importance_score: float,
) -> float:
    """Composite priority score for prioritized replay.

    priority = 0.35 * |td_error| + 0.30 * surprise + 0.35 * importance_score
    All inputs in [0.0, 1.0] range. Result clamped to [0.0, 1.0].
    """
    score = 0.35 * min(abs(td_error), 1.0) + 0.30 * surprise + 0.35 * importance_score
    return max(0.0, min(1.0, score))


class PredictionLearner:
    """Forward model + TD-error learning for procedural memory.

    Predicts procedure outcomes, computes prediction errors on feedback,
    updates Bayesian confidence with TD-modulated learning rates, and
    maintains episode priority scores for replay.
    """

    def __init__(
        self,
        db: Database,
        procedural: ProceduralMemory,
        base_learning_rate: float = 0.15,
    ):
        self.db = db
        self.procedural = procedural
        self.base_lr = base_learning_rate

    # ── Forward Model ─────────────────────────────────────────────

    def predict_outcome(
        self,
        procedure_id: str,
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
        domain: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Make a prediction about a procedure execution.

        Forward model logic:
        - predicted_success = 1 if confidence > 0.5 else 0
        - predicted_confidence = procedure.confidence

        Stores the prediction in prediction_log with td_error=0 (pending).
        Returns the prediction record including id for later outcome reporting.
        """
        proc = self.procedural.get(procedure_id)
        if not proc:
            return {
                "error": f"Procedure {procedure_id} not found",
                "prediction_id": None,
            }

        predicted_success = 1 if proc["confidence"] > 0.5 else 0
        predicted_confidence = proc["confidence"]

        pred_id = _new_id()

        self.db.insert(
            "prediction_log",
            {
                "id": pred_id,
                "procedure_id": procedure_id,
                "episode_id": episode_id,
                "predicted_success": predicted_success,
                "predicted_confidence": predicted_confidence,
                "actual_outcome": None,
                "td_error": 0.0,
                "surprise_score": None,
                "domain": domain or proc.get("domain"),
                "agent_id": agent_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        return {
            "prediction_id": pred_id,
            "procedure_id": procedure_id,
            "procedure_name": proc.get("name"),
            "predicted_success": bool(predicted_success),
            "predicted_confidence": predicted_confidence,
            "calibration_offset": proc.get("calibration_offset", 0.0),
        }

    # ── TD-Error & Surprise ───────────────────────────────────────

    def record_outcome(
        self,
        prediction_id: str,
        actual_success: bool,
        episode_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Record actual outcome against a prediction. Computes TD-error,
        surprise, updates procedure confidence, and updates episode priority.

        Full learning update chain:
        1. Fetch prediction record
        2. Compute TD-error: δ = actual - predicted
        3. Compute surprise: |δ| * (1 - predicted_confidence)
        4. Update prediction_log with results
        5. Update procedure confidence with TD-modulated learning rate
        6. Update source episode priority if episode_id provided
        """
        pred = self.db.fetchone(
            "SELECT * FROM prediction_log WHERE id = ?",
            (prediction_id,),
        )
        if not pred:
            return {"error": f"Prediction {prediction_id} not found"}

        # Skip if already recorded
        if pred.get("actual_outcome") is not None:
            return self._build_outcome_response(pred, already_recorded=True)

        predicted_success = pred["predicted_success"]
        predicted_confidence = pred["predicted_confidence"]
        procedure_id = pred["procedure_id"]
        actual_val = 1 if actual_success else 0

        # Step 2: TD-error
        td_error = compute_td_error(predicted_success, actual_val)

        # Step 3: Surprise
        surprise = compute_surprise(td_error, predicted_confidence)

        # Update prediction log
        self.db.update(
            "prediction_log",
            prediction_id,
            {
                "actual_outcome": actual_val,
                "td_error": td_error,
                "surprise_score": surprise,
            },
        )

        # Step 4: Update procedure confidence with TD-modulated learning rate
        proc = self.procedural.get(procedure_id)
        old_confidence = proc["confidence"] if proc else 0.0
        self._update_procedure_confidence(
            procedure_id=procedure_id,
            td_error=td_error,
            actual_success=actual_success,
            predicted_confidence=predicted_confidence,
        )

        # Record one procedure_evidence row linked to this prediction (verified).
        proc_after = self.procedural.get(procedure_id)
        new_confidence = proc_after["confidence"] if proc_after else old_confidence
        self.procedural.record_evidence(
            procedure_id=procedure_id,
            source="feedback",
            outcome="success" if actual_success else "failure",
            confidence_delta=new_confidence - old_confidence,
            prediction_id=prediction_id,
        )

        # Step 5: Update episode priority if episode_id available
        resolved_episode_id = episode_id or pred.get("episode_id")
        if resolved_episode_id:
            self._update_episode_priority(
                episode_id=resolved_episode_id,
                td_error=td_error,
                surprise=surprise,
            )

        return self._build_outcome_response(pred, td_error, surprise, actual_success)

    def _update_procedure_confidence(
        self,
        procedure_id: str,
        td_error: float,
        actual_success: bool,
        predicted_confidence: float,
    ) -> float:
        """Update procedure confidence with TD-modulated learning rate.

        - learning_rate = base_lr * (1 + |δ|)
        - Uses existing bayesian_confidence_update()
        - Updates calibration_offset = predicted_confidence - actual_rate
        - Updates prediction_error and surprise_score on procedure row
        """
        proc = self.procedural.get(procedure_id)
        if not proc:
            return 0.0

        abs_td = abs(td_error)
        lr = td_modulated_learning_rate(self.base_lr, abs_td)

        new_confidence = bayesian_confidence_update(
            proc["confidence"],
            actual_success,
            learning_rate=lr,
        )

        # Running PE tracking
        total_pe_sum = (proc.get("total_pe_sum", 0) or 0) + abs_td
        pe_count = (proc.get("pe_count", 0) or 0) + 1

        success_count = proc["success_count"] + (1 if actual_success else 0)
        failure_count = proc["failure_count"] + (0 if actual_success else 1)
        execution_count = proc["execution_count"] + 1
        total = success_count + failure_count
        actual_rate = success_count / total if total > 0 else None

        self.db.update(
            "procedures",
            procedure_id,
            {
                "confidence": new_confidence,
                "success_count": success_count,
                "failure_count": failure_count,
                "execution_count": execution_count,
                "prediction_error": abs_td,
                "surprise_score": abs(td_error) * (1.0 - predicted_confidence)
                if td_error != 0
                else 0.0,
                "total_pe_sum": total_pe_sum,
                "pe_count": pe_count,
                "actual_success_rate": actual_rate,
                "calibration_offset": (new_confidence - actual_rate)
                if actual_rate is not None
                else 0.0,
                "last_executed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        # Auto-promote to active if threshold met (mirrors procedural.record_execution)
        if new_confidence >= 0.8 and proc.get("status") in ("draft",):
            self.db.update(
                "procedures",
                procedure_id,
                {
                    "status": "active",
                },
            )

        return new_confidence

    def _update_episode_priority(
        self,
        episode_id: str,
        td_error: float,
        surprise: float,
    ) -> None:
        """Update an episode's td_error, surprise_score, and priority_score.

        priority = 0.35 * |td_error| + 0.30 * surprise + 0.35 * importance_score
        """
        episode = self.db.fetchone(
            "SELECT * FROM episodes WHERE id = ?",
            (episode_id,),
        )
        if not episode:
            return

        importance = episode.get("importance_score", 0.5) or 0.5
        priority = compute_priority_score(td_error, surprise, importance)

        self.db.update(
            "episodes",
            episode_id,
            {
                "td_error": abs(td_error),
                "surprise_score": surprise,
                "priority_score": priority,
            },
        )

    def _build_outcome_response(
        self,
        pred: dict[str, Any],
        td_error: float | None = None,
        surprise: float | None = None,
        actual_success: bool | None = None,
        already_recorded: bool = False,
    ) -> dict[str, Any]:
        """Build the response dict for record_outcome."""
        proc = self.procedural.get(pred["procedure_id"])
        result: dict[str, Any] = {
            "prediction_id": pred["id"],
            "procedure_id": pred["procedure_id"],
            "procedure_name": proc["name"] if proc else "unknown",
        }

        if already_recorded:
            result["status"] = "already_recorded"
            result["td_error"] = pred["td_error"]
            result["surprise_score"] = pred["surprise_score"]
            if proc:
                result["new_confidence"] = proc["confidence"]
                result["calibration_offset"] = proc.get("calibration_offset", 0.0)
            return result

        result.update(
            {
                "status": "recorded",
                "actual_success": actual_success,
                "td_error": td_error,
                "surprise_score": surprise,
            }
        )
        if proc:
            result["new_confidence"] = proc["confidence"]
            result["calibration_offset"] = proc.get("calibration_offset", 0.0)
        return result

    # ── MCP Tool Handlers ─────────────────────────────────────────

    async def handle_predict_outcome(
        self,
        procedure_id: str,
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
        domain: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """MCP tool: myelin_predict_outcome.

        Makes a prediction about a procedure's next outcome.
        """
        return self.predict_outcome(
            procedure_id=procedure_id,
            context=context,
            agent_id=agent_id,
            domain=domain,
            episode_id=episode_id,
        )

    async def handle_record_outcome(
        self,
        prediction_id: str,
        actual_success: bool,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """MCP tool: myelin_record_outcome.

        Records actual outcome, computes TD-error, updates confidence.
        """
        return self.record_outcome(
            prediction_id=prediction_id,
            actual_success=actual_success,
            episode_id=episode_id,
        )
