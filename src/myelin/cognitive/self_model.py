"""SelfModel — introspective confidence and competence assessment.

A learning OS needs to know what it knows, what it doesn't know,
and where it might be wrong. This provides:
1. Domain confidence calibration (how good are we in each domain?)
2. Bias detection (confirmation bias, recency bias, overconfidence)
3. Competence assessment (which domains are we improving/declining in?)
4. Self-evaluation reports (actionable insights for the agent)
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from ..core.database import Database
from ..core.models import ProcessName
from ..memory.episodic import EpisodicMemory
from ..memory.procedural import ProceduralMemory
from ..memory.semantic import SemanticMemory
from .base import CognitiveProcess

import logging

log = logging.getLogger("myelin.self_model")

# ── Constants ────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLDS = {
    "mastered": 0.85,
    "competent": 0.65,
    "learning": 0.40,
    "novel": 0.0,
}

BIAS_LOOKBACK_DAYS = 30
OVERCONFIDENCE_THRESHOLD = 0.15
RECENCY_WINDOW_HOURS = 24
IMPROVEMENT_THRESHOLD = 0.05
DECLINE_THRESHOLD = -0.05


# ── Pure Functions (testable) ────────────────────────────────────────


def assess_domain_competence(
    episode_count: int,
    procedure_count: int,
    success_rate: float,
    avg_confidence: float,
    calibration_offset: float,
    trend_delta: float,
) -> dict[str, Any]:
    """Assess competence level for a single domain.

    Returns dict with level, score, and recommendations.
    """
    coverage = min(1.0, (episode_count / 50) * 0.5 + (procedure_count / 10) * 0.5)
    # Quality score: success rate weighted by episode count confidence
    ep_confidence = min(1.0, episode_count / 10.0)  # need 10+ episodes to trust sr
    quality = success_rate * 0.4 * ep_confidence + (1.0 - abs(calibration_offset)) * 0.6
    competence = coverage * 0.4 + quality * 0.6

    if competence >= CONFIDENCE_THRESHOLDS["mastered"]:
        level = "mastered"
    elif competence >= CONFIDENCE_THRESHOLDS["competent"]:
        level = "competent"
    elif competence >= CONFIDENCE_THRESHOLDS["learning"]:
        level = "learning"
    else:
        level = "novel"

    if trend_delta > IMPROVEMENT_THRESHOLD:
        trend = "improving"
    elif trend_delta < DECLINE_THRESHOLD:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "level": level,
        "competence_score": round(competence, 3),
        "trend": trend,
        "trend_delta": round(trend_delta, 3),
        "episode_count": episode_count,
        "procedure_count": procedure_count,
        "success_rate": round(success_rate, 3),
        "avg_confidence": round(avg_confidence, 3),
        "calibration_offset": round(calibration_offset, 3),
    }


def detect_confirmation_bias(
    recent_outcomes: list[bool],
    recent_confidences: list[float],
) -> float:
    """Detect confirmation bias: tendency to ignore disconfirming evidence.
    
    Returns bias score 0-1 where:
    - 0 = no bias (evidence is honestly evaluated)
    - 1 = strong bias (failures explained away, successes over-weighted)
    
    Proxy: if confidence stays high despite repeated failures, bias is present.
    """
    if len(recent_outcomes) < 3:
        return 0.0

    failures = sum(1 for o in recent_outcomes if not o)
    if failures == 0:
        return 0.0

    # Confidence after failures should decrease
    fail_confidences = [
        recent_confidences[i] for i, o in enumerate(recent_outcomes) if not o
    ]
    if not fail_confidences:
        return 0.0

    avg_fail_conf = sum(fail_confidences) / len(fail_confidences)
    # If average confidence after failures > 0.6, likely confirmation bias
    return max(0.0, min(1.0, (avg_fail_conf - 0.3) / 0.7))


def detect_recency_bias(
    recent_success_rate: float,
    overall_success_rate: float,
) -> float:
    """Detect recency bias: recent events unduly influence confidence.
    
    Returns bias score 0-1 where:
    - large gap between recent and overall = recency bias
    """
    gap = abs(recent_success_rate - overall_success_rate)
    return min(1.0, gap * 2.0)


def detect_overconfidence(
    calibration_offset: float,
) -> float:
    """Detect overconfidence: predicted > actual success rate.
    
    Returns bias score 0-1.
    """
    if calibration_offset <= 0:
        return 0.0
    return min(1.0, calibration_offset / 0.5)


def compute_bias_report(
    recent_outcomes: list[bool],
    recent_confidences: list[float],
    recent_success_rate: float,
    overall_success_rate: float,
    calibration_offset: float,
) -> dict[str, Any]:
    """Compute full bias detection report."""
    conf_bias = detect_confirmation_bias(recent_outcomes, recent_confidences)
    rec_bias = detect_recency_bias(recent_success_rate, overall_success_rate)
    overconf = detect_overconfidence(calibration_offset)

    biases = []
    if conf_bias > 0.4:
        biases.append({"bias": "confirmation", "score": round(conf_bias, 3)})
    if rec_bias > 0.4:
        biases.append({"bias": "recency", "score": round(rec_bias, 3)})
    if overconf > 0.4:
        biases.append({"bias": "overconfidence", "score": round(overconf, 3)})

    return {
        "biases": biases,
        "confirmation_bias": round(conf_bias, 3),
        "recency_bias": round(rec_bias, 3),
        "overconfidence": round(overconf, 3),
        "overall_calibration_offset": round(calibration_offset, 3),
    }


def generate_insights(competence_map: dict[str, Any]) -> list[str]:
    """Generate actionable insights from competence assessment."""
    insights = []

    for domain, info in competence_map.items():
        level = info.get("level", "novel")
        trend = info.get("trend", "stable")

        if level == "novel":
            insights.append(
                f"Domain '{domain}' is novel — only {info['episode_count']} episodes. "
                "Prioritize collecting more observations."
            )
        elif level == "learning":
            if trend == "declining":
                insights.append(
                    f"Domain '{domain}' is declining from learning level. "
                    "Recent performance suggests regression — investigate and practice."
                )
            elif info["calibration_offset"] > OVERCONFIDENCE_THRESHOLD:
                insights.append(
                    f"Domain '{domain}' shows overconfidence "
                    f"(offset={info['calibration_offset']:.2f}). "
                    "Confidence exceeds actual performance — review failures carefully."
                )
        elif level == "competent":
            if trend == "improving":
                insights.append(
                    f"Domain '{domain}' approaching mastery "
                    f"(competence={info['competence_score']:.2f}). "
                    "Continue current practice pattern."
                )
        elif level == "mastered":
            insights.append(
                f"Domain '{domain}' is mastered "
                f"(competence={info['competence_score']:.2f}). "
                "Can reliably produce correct procedures. "
                "Focus maintenance effort elsewhere."
            )

    return insights


# ── SelfModel Cognitive Process ──────────────────────────────────────


class SelfModel(CognitiveProcess):
    """Self-assessment process that monitors calibration and competence.

    Runs on session end. Produces a bias report and competence map
    that the agent can use to adjust its behavior.
    """

    name = ProcessName.SELF_MODEL

    def __init__(
        self,
        db: Database,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
    ):
        super().__init__(db)
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

    def should_run(self) -> bool:
        return True

    async def execute(self) -> dict[str, Any]:
        results = {
            "domains_assessed": 0,
            "biases_detected": 0,
            "insights_generated": 0,
        }

        # 1. Assess competence per domain
        domains = self._get_domains()
        competence_map: dict[str, Any] = {}

        for domain in domains:
            info = self._assess_domain(domain)
            if info:
                competence_map[domain] = info
                results["domains_assessed"] += 1

        # 2. Detect biases
        recent_outcomes, recent_confidences = self._get_recent_calibration_data()
        calibration_offsets = [
            c.get("calibration_offset", 0.0)
            for c in competence_map.values()
        ]
        avg_offset = (
            sum(calibration_offsets) / len(calibration_offsets)
            if calibration_offsets
            else 0.0
        )

        overall_sr = self._get_overall_success_rate()
        recent_sr = self._get_recent_success_rate()

        bias_report = compute_bias_report(
            recent_outcomes,
            recent_confidences,
            recent_sr,
            overall_sr,
            avg_offset,
        )
        results["biases_detected"] = len(bias_report["biases"])

        # 3. Generate insights
        insights = generate_insights(competence_map)
        results["insights_generated"] = len(insights)

        # 4. Store self-evaluation in DB
        self._store_evaluation(competence_map, bias_report, insights)

        results["competence_map"] = {
            d: {"level": v["level"], "trend": v["trend"], "score": v["competence_score"]}
            for d, v in competence_map.items()
        }
        results["bias_report"] = {b["bias"]: b["score"] for b in bias_report["biases"]}

        return results

    def _get_domains(self) -> list[str]:
        rows = self.db.fetchall(
            "SELECT DISTINCT domain FROM episodes WHERE domain IS NOT NULL"
        )
        return [r["domain"] for r in rows if r["domain"]]

    def _assess_domain(self, domain: str) -> dict[str, Any] | None:
        """Assess competence for a single domain."""
        ep_count = self.episodic.count()
        procs = self.procedural.get_by_domain(domain)
        proc_count = len(procs)

        if ep_count == 0:
            return None

        # Get domain episodes
        eps = self.episodic.get_by_domain(domain, limit=200)
        if not eps:
            return None

        success_rate = sum(1 for e in eps if e.get("success", 1)) / len(eps)
        avg_confidence = (
            sum(p.get("confidence", 0.5) for p in procs) / len(procs)
            if procs
            else 0.0
        )

        # Calibration offset: predicted vs actual
        calibration_offset = 0.0
        if procs:
            offsets = [
                p.get("calibration_offset", 0.0) for p in procs if p.get("calibration_offset")
            ]
            calibration_offset = sum(offsets) / len(offsets) if offsets else 0.0

        # Trend: compare recent success rate to overall
        recent_eps = [e for e in eps if self._is_recent(e.get("created_at", ""), 7)]
        recent_sr = (
            sum(1 for e in recent_eps if e.get("success", 1)) / len(recent_eps)
            if recent_eps
            else success_rate
        )
        trend_delta = recent_sr - success_rate

        return assess_domain_competence(
            ep_count, proc_count, success_rate, avg_confidence,
            calibration_offset, trend_delta,
        )

    def _get_recent_calibration_data(self) -> tuple[list[bool], list[float]]:
        """Get recent outcomes and confidences for bias detection."""
        rows = self.db.fetchall(
            "SELECT actual_outcome, predicted_confidence FROM calibration_log "
            "ORDER BY timestamp DESC LIMIT 50"
        )
        outcomes = [bool(r["actual_outcome"]) for r in rows]
        confidences = [float(r["predicted_confidence"]) for r in rows]
        return outcomes, confidences

    def _get_overall_success_rate(self) -> float:
        row = self.db.fetchone(
            "SELECT AVG(CAST(success AS REAL)) as sr FROM episodes"
        )
        return row["sr"] if row and row["sr"] is not None else 0.5

    def _get_recent_success_rate(self) -> float:
        cutoff = (datetime.utcnow() - timedelta(hours=RECENCY_WINDOW_HOURS)).isoformat()
        row = self.db.fetchone(
            "SELECT AVG(CAST(success AS REAL)) as sr FROM episodes WHERE created_at > ?",
            (cutoff,),
        )
        return row["sr"] if row and row["sr"] is not None else 0.5

    def _is_recent(self, timestamp_str: str, days: int) -> bool:
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return (datetime.utcnow() - dt.replace(tzinfo=None)).days <= days
        except (ValueError, TypeError):
            return False

    def _store_evaluation(
        self,
        competence_map: dict,
        bias_report: dict,
        insights: list[str],
    ) -> None:
        """Store self-evaluation in the self_evaluations table."""
        try:
            top = sorted(
                competence_map.items(),
                key=lambda x: x[1].get("competence_score", 0),
                reverse=True,
            )[:5]
            weak = sorted(
                competence_map.items(),
                key=lambda x: x[1].get("competence_score", 0),
            )[:5]
            improving = [
                d for d, v in competence_map.items()
                if v.get("trend") == "improving"
            ]
            declining = [
                d for d, v in competence_map.items()
                if v.get("trend") == "declining"
            ]

            self.db.insert("self_evaluations", {
                "id": __import__("uuid").uuid4().hex[:16],
                "timestamp": datetime.utcnow().isoformat(),
                "top_domains": json.dumps([{"domain": d, **v} for d, v in top]),
                "weak_domains": json.dumps([{"domain": d, **v} for d, v in weak]),
                "improving": json.dumps(improving),
                "declining": json.dumps(declining),
                "insights": json.dumps(insights),
            })
            self.db.commit()
        except Exception as e:
            log.warning(f"Failed to store self-evaluation: {e}")
