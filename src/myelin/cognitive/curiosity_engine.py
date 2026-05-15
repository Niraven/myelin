"""Curiosity-driven active learning engine for Myelin.

Detects knowledge gaps across entities, domains, procedures, and relationships;
computes composite curiosity scores; generates learning goals; and manages
exploration-vs-exploitation decisions with epsilon-greedy scheduling.

References:
    - IAC/R-IAC (Oudeyer 2007, Baranes 2009)
    - ICM (Pathak 2017), Bayesian Curiosity (Stadie 2019)
    - FreshPER (2026) staleness-aware prioritization
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..core.database import Database
from ..core.models import (
    CuriousGoalModel,
    CuriosityTopic,
    GoalStatus,
    LearningGoal,
    ProcessName,
)
from .base import CognitiveProcess


def _new_id() -> str:
    return uuid4().hex[:16]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ── Constants (from spec §11) ─────────────────────────────────────

N_NOVELTY_MAX = 10       # entity mentions for novelty=0
P_NOVELTY_MAX = 5        # domain procedures for novelty=0
E_NOVELTY_MAX = 5        # procedure executions for novelty=0
R_NOVELTY_MAX = 5        # relationship evidence count for novelty=0

# Default weights (spec §3.4)
W_NOVELTY = 0.30
W_UNCERTAINTY = 0.35
W_INFOGAIN = 0.35

# Gap-type-specific weight overrides (spec §3.4, §11.6)
WEIGHT_OVERRIDES: dict[str, tuple[float, float, float]] = {
    "entity_undermentions":            (0.45, 0.35, 0.20),
    "domain_low_procedures":           (0.40, 0.40, 0.20),
    "domain_high_uncertainty":         (0.15, 0.50, 0.35),
    "procedure_needs_testing":         (0.15, 0.35, 0.50),
    "relationship_needs_verification": (0.30, 0.35, 0.35),
}

# Epsilon schedule (spec §5.1)
EPSILON_START = 0.30
EPSILON_END = 0.05
EPSILON_TAU = 100  # half-life in cycles

# Exploration budget (spec §5.4)
MAX_EXPLORATIONS_PER_HOUR = 3

# Staleness / fatigue (spec §3.5, §6.4)
STALENESS_HALFLIFE_DAYS = 14  # τ_curiosity
FATIGUE_HALFLIFE_ATTEMPTS = 5  # τ_fatigue

# Goal thresholds (spec §4)
MIN_CURIOSITY_THRESHOLD = 0.10
TOP_GAPS_KEPT = 20
GOAL_ABANDONMENT_DAYS = 30

# Cold-start (spec §10.1)
COLD_START_EPISODE_LIMIT = 5

# Intrinsic reward parameters (spec §6)
ALPHA_IRL = 0.10  # intrinsic reward learning rate

# Metadata keys
ENTITY_MENTION_COUNT = "mention_count"
DOMAIN_AVG_MENTIONS = "domain_avg_mentions"
DOMAIN_PROCEDURE_COUNT = "procedure_count"
DOMAIN_EPISODE_COUNT = "episode_count"
DOMAIN_CONFIDENCE = "domain_confidence"
DOMAIN_TREND = "domain_trend"
PROCEDURE_CONFIDENCE = "confidence"
PROCEDURE_EXECUTION_COUNT = "execution_count"
PROCEDURE_ACTUAL_SUCCESS = "actual_success_rate"
PROCEDURE_PREDICTED_SUCCESS = "predicted_success_rate"
RELATION_STRENGTH = "strength"
RELATION_EVIDENCE_COUNT = "evidence_count"
RELATION_SOURCE = "source_entity"
RELATION_TARGET = "target_entity"
RELATION_TYPE = "relation_type"

# Gap type constants
GAP_ENTITY = "entity_undermentions"
GAP_DOMAIN_LOW = "domain_low_procedures"
GAP_DOMAIN_UNCERT = "domain_high_uncertainty"
GAP_PROCEDURE = "procedure_needs_testing"
GAP_RELATIONSHIP = "relationship_needs_verification"
GAP_COLD_START = "explore_environment"


# ═══════════════════════════════════════════════════════════════════
#  Pure scoring functions
# ═══════════════════════════════════════════════════════════════════


def compute_novelty_entity(mention_count: int) -> float:
    """novelty(e) = 1 - min(mention_count / N_novelty_max, 1.0)"""
    return 1.0 - min(mention_count / N_NOVELTY_MAX, 1.0)


def compute_novelty_domain(procedure_count: int, episode_count: int = 0) -> float:
    """novelty(d) = 1 - min(procedure_count / P_novelty_max, 1.0)

    Special case: 0 procedures but >= 10 episodes → novelty = 0.8.
    """
    if procedure_count == 0 and episode_count >= 10:
        return 0.8
    return 1.0 - min(procedure_count / P_NOVELTY_MAX, 1.0)


def compute_novelty_procedure(execution_count: int) -> float:
    """novelty(p) = 1 - min(execution_count / E_novelty_max, 1.0)"""
    return 1.0 - min(execution_count / E_NOVELTY_MAX, 1.0)


def compute_novelty_relationship(evidence_count: int) -> float:
    """novelty(r) = 1 - min(evidence_count / R_novelty_max, 1.0)"""
    return 1.0 - min(evidence_count / R_NOVELTY_MAX, 1.0)


def compute_uncertainty_entity(max_confidence_involving: float = 0.0) -> float:
    """uncertainty(e) = 1 - max_confidence_involving(e), fallback 0.5"""
    if max_confidence_involving <= 0.0:
        return 0.5
    return 1.0 - max_confidence_involving


def compute_uncertainty_domain(domain_confidence: float = 0.0) -> float:
    """uncertainty(d) = 1 - confidence_map.confidence(d), fallback 0.7"""
    if domain_confidence <= 0.0:
        return 0.7
    return _clamp(1.0 - domain_confidence)


def compute_uncertainty_procedure(confidence: float) -> float:
    """uncertainty(p) = 1 - confidence(p)"""
    return _clamp(1.0 - confidence)


def compute_uncertainty_relationship(strength: float) -> float:
    """uncertainty(r) = 1 - strength(r)"""
    return _clamp(1.0 - strength)


def compute_infogain_entity(
    prediction_error_variance: float = 0.5,
) -> float:
    """information_gain_potential(e) = PE_variance(domain_of(e)), fallback 0.5"""
    return prediction_error_variance if prediction_error_variance > 0 else 0.5


def compute_infogain_domain(prediction_error_variance: float) -> float:
    """information_gain_potential(d) = PE_variance(d)"""
    return prediction_error_variance


def compute_infogain_procedure(
    predicted_success_rate: float | None,
    actual_success_rate: float | None,
    execution_count: int,
) -> float:
    """information_gain_potential(p) =
        |predicted - actual| + (1 - min(exec_count / 10, 1.0))
    """
    pred = predicted_success_rate if predicted_success_rate is not None else 0.5
    actual = actual_success_rate if actual_success_rate is not None else 0.5
    calibration_gap = abs(pred - actual)
    exec_novelty = 1.0 - min(execution_count / 10, 1.0)
    return _clamp(calibration_gap + exec_novelty)


def compute_infogain_relationship(evidence_count: int, strength: float) -> float:
    """information_gain_potential(r) =
        0.5 / (evidence_count + 1) * (1 - strength)
    """
    return 0.5 / (evidence_count + 1) * (1.0 - strength)


def compute_curiosity_score(
    novelty: float,
    uncertainty: float,
    infogain: float,
    gap_type: str = "",
) -> float:
    """Composite weighted score.

    Uses gap-type-specific weight overrides when available.
    """
    w_n, w_u, w_i = WEIGHT_OVERRIDES.get(gap_type, (W_NOVELTY, W_UNCERTAINTY, W_INFOGAIN))
    return _clamp(w_n * _clamp(novelty) + w_u * _clamp(uncertainty) + w_i * _clamp(infogain))


def apply_staleness_decay(
    curiosity_score: float,
    age_days: float,
    half_life_days: float = STALENESS_HALFLIFE_DAYS,
) -> float:
    """FreshPER-inspired decay: score *= exp(-age / τ)"""
    return curiosity_score * math.exp(-age_days / half_life_days)


def apply_fatigue(
    curiosity_score: float,
    attempts: int,
    half_life: int = FATIGUE_HALFLIFE_ATTEMPTS,
) -> float:
    """Curiosity fatigue: score *= exp(-attempts / τ_fatigue)"""
    return curiosity_score * math.exp(-attempts / half_life)


def compute_epsilon(sleep_cycles_completed: int) -> float:
    """ε(t) = ε_end + (ε_start - ε_end) × exp(-t / τ_epsilon)"""
    return EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
        -sleep_cycles_completed / EPSILON_TAU
    )


def compute_prediction_error_variance(
    predicted: list[float],
    actual: list[float],
) -> float:
    """Compute variance of δ = predicted - actual across procedures."""
    if len(predicted) < 2 or len(actual) < 2:
        return 0.0
    deltas = [p - a for p, a in zip(predicted, actual)]
    n = len(deltas)
    mean = sum(deltas) / n
    variance = sum((d - mean) ** 2 for d in deltas) / n
    return _clamp(variance, 0.0, 1.0)


def softmax_sample(
    items: list[Any],
    scores: list[float],
    temperature: float = 0.5,
) -> Any | None:
    """Sample an item from a list using softmax over scores."""
    if not items or not scores:
        return None
    scaled = [s / temperature for s in scores]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]  # numeric stability
    total = sum(exps)
    if total == 0:
        return None
    probs = [e / total for e in exps]
    r = random.random()
    cumulative = 0.0
    for i, prob in enumerate(probs):
        cumulative += prob
        if r <= cumulative:
            return items[i]
    return items[-1]


# ═══════════════════════════════════════════════════════════════════
#  Goal Template Helpers
# ═══════════════════════════════════════════════════════════════════


def _goal_text_and_needed(topic: CuriosityTopic) -> tuple[str, int]:
    """Generate a human-readable goal description and episodes_needed."""
    meta = topic.metadata
    if topic.gap_type == GAP_ENTITY:
        mention_count = meta.get(ENTITY_MENTION_COUNT, 0)
        needed = max(3, 10 - mention_count)
        return (
            f"Learn more about entity {topic.target_name} — "
            f"only {mention_count} mentions, needs {needed} more observations",
            needed,
        )
    elif topic.gap_type == GAP_DOMAIN_LOW:
        proc_count = meta.get(DOMAIN_PROCEDURE_COUNT, 0)
        needed = max(3, 5 - proc_count)
        return (
            f"Improve procedural knowledge in {topic.target_name} — "
            f"{proc_count} procedures, needs {needed} more",
            needed,
        )
    elif topic.gap_type == GAP_DOMAIN_UNCERT:
        return (
            f"Reduce prediction uncertainty in {topic.target_name} — "
            f"variance is {topic.raw_score:.3f}, collect calibration data",
            5,
        )
    elif topic.gap_type == GAP_PROCEDURE:
        confidence = meta.get(PROCEDURE_CONFIDENCE, 0.0)
        exec_count = meta.get(PROCEDURE_EXECUTION_COUNT, 0)
        return (
            f"Test procedure {topic.target_name} — "
            f"confidence is {confidence:.2f}, "
            f"executed {exec_count} times, untested recently",
            3,
        )
    elif topic.gap_type == GAP_RELATIONSHIP:
        source = meta.get(RELATION_SOURCE, "?")
        target = meta.get(RELATION_TARGET, "?")
        rtype = meta.get(RELATION_TYPE, "?")
        return (
            f"Verify relationship: {source} {rtype} {target} — "
            f"strength {meta.get(RELATION_STRENGTH, 0):.2f}, "
            f"evidence {meta.get(RELATION_EVIDENCE_COUNT, 0)}",
            3,
        )
    else:
        return (f"Explore: {topic.target_name}", 3)


def _format_strategy(topic: CuriosityTopic) -> str:
    """Return the exploration strategy for a gap type."""
    strategies = {
        GAP_ENTITY: "collect_entity_observations",
        GAP_DOMAIN_LOW: "gather_domain_evidence",
        GAP_DOMAIN_UNCERT: "collect_calibration_data",
        GAP_PROCEDURE: "execute_procedure",
        GAP_RELATIONSHIP: "gather_relationship_evidence",
    }
    return strategies.get(topic.gap_type, "explore")


# ═══════════════════════════════════════════════════════════════════
#  CuriosityGapDetector — Inner class
# ═══════════════════════════════════════════════════════════════════


class CuriosityGapDetector:
    """Five signal-source gap detectors that scan Myelin stores."""

    def __init__(self, db: Database):
        self.db = db

    def detect_all(self) -> list[CuriosityTopic]:
        """Run all five detectors and return deduplicated topics."""
        topics: dict[str, CuriosityTopic] = {}

        for detector in [
            self._detect_entity_gaps,
            self._detect_domain_low_procedure_gaps,
            self._detect_high_pe_variance_gaps,
            self._detect_untested_procedure_gaps,
            self._detect_unverified_relationship_gaps,
        ]:
            try:
                for topic in detector():
                    key = f"{topic.gap_type}:{topic.target_id}"
                    if key in topics:
                        if topic.curiosity_score > topics[key].curiosity_score:
                            topics[key] = topic
                    else:
                        topics[key] = topic
            except Exception:
                continue

        return list(topics.values())

    # ── 2.1 Entity Gap Detector ─────────────────────────────────

    def _detect_entity_gaps(self) -> list[CuriosityTopic]:
        """Low-mention entities: mention_count < 3 OR < 30% of domain avg."""
        results = []
        rows = self.db.fetchall(
            "SELECT e.id, e.name, e.entity_type, e.mention_count, e.domain, "
            "       e.created_at "
            "FROM entities e"
        )

        # Compute domain averages
        domain_counts: dict[str, list[int]] = {}
        for r in rows:
            domain = r["domain"]
            if domain:
                domain_counts.setdefault(domain, []).append(r["mention_count"])
        domain_avgs: dict[str, float] = {
            d: sum(c) / len(c) for d, c in domain_counts.items()
        }

        for r in rows:
            mention_count = r["mention_count"] or 0
            domain = r["domain"]
            avg_mentions = domain_avgs.get(domain, 999) if domain else 999

            # Thresholds from spec §2.1
            if mention_count >= 3 and (not domain or mention_count >= 0.3 * avg_mentions):
                continue

            # Age filter: mention_count=0 and age > 30 days → archive, skip
            if mention_count == 0:
                created = r.get("created_at", "")
                if created:
                    try:
                        from datetime import datetime
                        created_dt = datetime.strptime(created.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                        age_days = (datetime.utcnow() - created_dt).days
                        if age_days > 30:
                            continue
                    except (ValueError, TypeError):
                        pass

            # Max confidence involving this entity
            max_conf = self._max_confidence_for_entity(r["id"])

            novelty = compute_novelty_entity(mention_count)
            uncertainty = compute_uncertainty_entity(max_conf)
            infogain = compute_infogain_entity(
                self._pe_variance_for_domain(domain) if domain else 0.5,
            )
            score = compute_curiosity_score(novelty, uncertainty, infogain, GAP_ENTITY)

            results.append(CuriosityTopic(
                gap_type=GAP_ENTITY,
                target_id=r["id"],
                target_name=r["name"],
                domain=domain,
                raw_score=3 - mention_count,
                novelty_score=novelty,
                uncertainty_score=uncertainty,
                infogain_potential=infogain,
                curiosity_score=score,
                metadata={
                    ENTITY_MENTION_COUNT: mention_count,
                    DOMAIN_AVG_MENTIONS: avg_mentions,
                    "entity_type": r["entity_type"],
                },
            ))
        return results

    # ── 2.2 Low-Procedure Domain Detector ───────────────────────

    def _detect_domain_low_procedure_gaps(self) -> list[CuriosityTopic]:
        """Domains with < 2 procedures or 10+ episodes with zero procedures."""
        results = []
        rows = self.db.fetchall(
            "SELECT cm.domain, cm.procedure_count, cm.episode_count, "
            "       cm.confidence, cm.trend "
            "FROM confidence_map cm"
        )

        for r in rows:
            domain = r["domain"]
            proc_count = r["procedure_count"] or 0
            ep_count = r["episode_count"] or 0
            trend = r["trend"] or "stable"
            domain_conf = r["confidence"] or 0.0

            # Skip inherently non-procedural domains: stable + 0 proc + 20+ episodes
            if (
                proc_count == 0
                and ep_count >= 20
                and trend == "stable"
            ):
                continue

            # Thresholds from spec §2.2
            if proc_count >= 2 and not (ep_count > 10 and proc_count == 0):
                continue

            novelty = compute_novelty_domain(proc_count, ep_count)
            uncertainty = compute_uncertainty_domain(domain_conf)
            pe_var = self._pe_variance_for_domain(domain)
            infogain = compute_infogain_domain(pe_var)
            score = compute_curiosity_score(novelty, uncertainty, infogain, GAP_DOMAIN_LOW)

            results.append(CuriosityTopic(
                gap_type=GAP_DOMAIN_LOW,
                target_id=f"domain:{domain}",
                target_name=domain,
                domain=domain,
                raw_score=2 - proc_count,
                novelty_score=novelty,
                uncertainty_score=uncertainty,
                infogain_potential=infogain,
                curiosity_score=score,
                metadata={
                    DOMAIN_PROCEDURE_COUNT: proc_count,
                    DOMAIN_EPISODE_COUNT: ep_count,
                    DOMAIN_CONFIDENCE: domain_conf,
                    DOMAIN_TREND: trend,
                },
            ))
        return results

    # ── 2.3 High PE Variance Domain Detector ────────────────────

    def _detect_high_pe_variance_gaps(self) -> list[CuriosityTopic]:
        """Domains where prediction error variance > 0.15."""
        results = []
        domains = self.db.fetchall(
            "SELECT DISTINCT domain FROM procedures WHERE domain IS NOT NULL"
        )
        if not domains:
            # Also check confidence_map for domains
            cm_domains = self.db.fetchall(
                "SELECT DISTINCT domain FROM confidence_map WHERE domain IS NOT NULL"
            )
            domains = cm_domains

        for r in domains:
            domain = r["domain"]
            pe_var = self._pe_variance_for_domain(domain)

            if pe_var <= 0.15:
                continue

            cm = self.db.fetchone(
                "SELECT confidence, procedure_count FROM confidence_map WHERE domain = ?",
                (domain,),
            )
            proc_count = cm["procedure_count"] if cm else 0
            domain_conf = cm["confidence"] if cm else 0.0
            ep_count = 0
            ep_row = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM episodes WHERE domain = ?",
                (domain,),
            )
            if ep_row:
                ep_count = ep_row["cnt"]

            novelty = compute_novelty_domain(proc_count, ep_count)
            uncertainty = pe_var  # Already in [0,1], used as uncertainty
            infogain = pe_var
            score = compute_curiosity_score(novelty, uncertainty, infogain, GAP_DOMAIN_UNCERT)

            results.append(CuriosityTopic(
                gap_type=GAP_DOMAIN_UNCERT,
                target_id=f"domain:{domain}",
                target_name=domain,
                domain=domain,
                raw_score=pe_var,
                novelty_score=novelty,
                uncertainty_score=uncertainty,
                infogain_potential=infogain,
                curiosity_score=score,
                metadata={
                    "variance": pe_var,
                    DOMAIN_PROCEDURE_COUNT: proc_count,
                    DOMAIN_CONFIDENCE: domain_conf,
                },
            ))
        return results

    # ── 2.4 Untested Low-Confidence Procedure Detector ──────────

    def _detect_untested_procedure_gaps(self) -> list[CuriosityTopic]:
        """Procedures with confidence < 0.6 and last_executed > 7 days ago."""
        results = []
        rows = self.db.fetchall(
            "SELECT p.id, p.name, p.domain, p.confidence, "
            "       p.predicted_success_rate, p.actual_success_rate, "
            "       p.last_executed, p.execution_count "
            "FROM procedures p "
            "WHERE p.status = 'active' "
            "  AND p.confidence < 0.6"
        )

        for r in rows:
            # Check last_executed: NULL or > 7 days ago
            last_exc = r.get("last_executed")
            if last_exc:
                try:
                    from datetime import datetime
                    last_dt = datetime.strptime(last_exc.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                    delta_days = (datetime.utcnow() - last_dt).days
                    if delta_days <= 7:
                        continue
                except (ValueError, TypeError):
                    pass
            # If last_executed is NULL, it means never executed — include it

            exec_count = r["execution_count"] or 0
            confidence = r["confidence"] or 0.5
            pred_rate = r["predicted_success_rate"]
            actual_rate = r["actual_success_rate"]

            novelty = compute_novelty_procedure(exec_count)
            uncertainty = compute_uncertainty_procedure(confidence)
            infogain = compute_infogain_procedure(pred_rate, actual_rate, exec_count)
            score = compute_curiosity_score(novelty, uncertainty, infogain, GAP_PROCEDURE)

            results.append(CuriosityTopic(
                gap_type=GAP_PROCEDURE,
                target_id=r["id"],
                target_name=r["name"],
                domain=r.get("domain"),
                raw_score=0.6 - confidence,
                novelty_score=novelty,
                uncertainty_score=uncertainty,
                infogain_potential=infogain,
                curiosity_score=score,
                metadata={
                    PROCEDURE_CONFIDENCE: confidence,
                    PROCEDURE_EXECUTION_COUNT: exec_count,
                    PROCEDURE_PREDICTED_SUCCESS: pred_rate,
                    PROCEDURE_ACTUAL_SUCCESS: actual_rate,
                },
            ))
        return results

    # ── 2.5 Unverified Relationship Detector ────────────────────

    def _detect_unverified_relationship_gaps(self) -> list[CuriosityTopic]:
        """Relationships with strength < 0.3 and evidence_count < 3, age < 30 days."""
        results = []
        rows = self.db.fetchall(
            "SELECT r.id, r.source_entity_id, r.target_entity_id, "
            "       r.relation_type, r.strength, r.evidence_count, "
            "       r.last_observed, r.domain, "
            "       se.name AS source_name, te.name AS target_name "
            "FROM relationships r "
            "JOIN entities se ON r.source_entity_id = se.id "
            "JOIN entities te ON r.target_entity_id = te.id "
            "WHERE r.strength < 0.3 "
            "  AND r.evidence_count < 3"
        )

        for r in rows:
            # Age filter: skip if older than 30 days
            last_obs = r.get("last_observed")
            if last_obs:
                try:
                    from datetime import datetime
                    obs_dt = datetime.strptime(last_obs.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                    age_days = (datetime.utcnow() - obs_dt).days
                    if age_days >= 30:
                        continue
                except (ValueError, TypeError):
                    pass

            evidence_count = r["evidence_count"] or 1
            strength = r["strength"] or 0.0

            novelty = compute_novelty_relationship(evidence_count)
            uncertainty = compute_uncertainty_relationship(strength)
            infogain = compute_infogain_relationship(evidence_count, strength)
            score = compute_curiosity_score(novelty, uncertainty, infogain, GAP_RELATIONSHIP)

            target_name = (
                f"{r['source_name']} → {r['target_name']} [{r['relation_type']}]"
            )

            results.append(CuriosityTopic(
                gap_type=GAP_RELATIONSHIP,
                target_id=r["id"],
                target_name=target_name,
                domain=r.get("domain"),
                raw_score=0.3 - strength,
                novelty_score=novelty,
                uncertainty_score=uncertainty,
                infogain_potential=infogain,
                curiosity_score=score,
                metadata={
                    RELATION_STRENGTH: strength,
                    RELATION_EVIDENCE_COUNT: evidence_count,
                    RELATION_SOURCE: r["source_name"],
                    RELATION_TARGET: r["target_name"],
                    RELATION_TYPE: r["relation_type"],
                },
            ))
        return results

    # ── Helpers ─────────────────────────────────────────────────

    def _max_confidence_for_entity(self, entity_id: str) -> float:
        """Maximum confidence of any procedure mentioning this entity."""
        row = self.db.fetchone(
            "SELECT MAX(p.confidence) as max_conf "
            "FROM procedures p "
            "JOIN entity_mentions em ON em.source_id = p.id AND em.source_type = 'procedure' "
            "WHERE em.entity_id = ?",
            (entity_id,),
        )
        if row and row["max_conf"] is not None:
            return row["max_conf"]
        return 0.0

    def _pe_variance_for_domain(self, domain: str) -> float:
        """Compute prediction error variance for a domain's procedures.

        Implements spec §2.3 algorithm:
        For each procedure: δ = predicted - actual success rate.
        variance(δ) for >= 2 procedures.
        """
        procs = self.db.fetchall(
            "SELECT predicted_success_rate, actual_success_rate "
            "FROM procedures WHERE domain = ? ",
            (domain,),
        )
        valid = [p for p in procs if p["predicted_success_rate"] is not None]
        if not valid:
            return 0.0

        if len(valid) >= 2:
            deltas = [
                (p["predicted_success_rate"] or 0.5) - (p["actual_success_rate"] or 0.5)
                for p in valid
            ]
            n = len(deltas)
            mean = sum(deltas) / n
            var = sum((d - mean) ** 2 for d in deltas) / n
            return _clamp(var)
        elif len(valid) == 1:
            # Single procedure: use calibration log
            proc_id = procs[0]["id"] if "id" in procs[0] else None
            if proc_id:
                cal = self.db.fetchall(
                    "SELECT predicted_confidence, actual_outcome "
                    "FROM prediction_log WHERE procedure_id = ? "
                    "ORDER BY timestamp DESC LIMIT 20",
                    (proc_id,),
                )
                if len(cal) >= 3:
                    deltas = [
                        (c["predicted_confidence"] or 0.5) - c["actual_outcome"]
                        for c in cal
                    ]
                    n = len(deltas)
                    mean = sum(deltas) / n
                    var = sum((d - mean) ** 2 for d in deltas) / n
                    return _clamp(var)
                else:
                    return 0.5  # default uncertainty
            return 0.5
        return 0.0


# ═══════════════════════════════════════════════════════════════════
#  CuriosityEngine — Main class
# ═══════════════════════════════════════════════════════════════════


class CuriosityEngine(CognitiveProcess):
    """Curiosity-driven active learning engine.

    Detects knowledge gaps, computes exploration scores, generates
    learning goals, and manages epsilon-greedy exploration decisions.

    Runs during sleep cycle as the `curious_explorer` process.
    """

    name = ProcessName.CURIOUS_EXPLORER

    CURIOSITY_SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS curiosity_scores (
            id TEXT PRIMARY KEY,
            gap_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            target_name TEXT NOT NULL,
            novelty_score REAL NOT NULL DEFAULT 0.0,
            uncertainty_score REAL NOT NULL DEFAULT 0.0,
            infogain_potential REAL NOT NULL DEFAULT 0.0,
            curiosity_score REAL NOT NULL DEFAULT 0.0,
            domain TEXT,
            raw_score REAL NOT NULL DEFAULT 0.0,
            metadata TEXT DEFAULT '{}',
            exploration_attempts INTEGER NOT NULL DEFAULT 0,
            last_explored_at TEXT,
            fatigue_factor REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_curiosity_scores_score ON curiosity_scores(curiosity_score DESC);
        CREATE INDEX IF NOT EXISTS idx_curiosity_scores_domain ON curiosity_scores(domain);
        CREATE INDEX IF NOT EXISTS idx_curiosity_scores_type ON curiosity_scores(gap_type);
        CREATE INDEX IF NOT EXISTS idx_curiosity_scores_target ON curiosity_scores(target_id);

        CREATE TABLE IF NOT EXISTS intrinsic_reward_log (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            reward_value REAL NOT NULL,
            metadata TEXT DEFAULT '{}',
            trigger_episode_id TEXT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_reward_source ON intrinsic_reward_log(source_type, source_id);
        CREATE INDEX IF NOT EXISTS idx_reward_timestamp ON intrinsic_reward_log(timestamp);

        CREATE TABLE IF NOT EXISTS exploration_arms (
            id TEXT PRIMARY KEY,
            arm_name TEXT NOT NULL UNIQUE,
            arm_type TEXT NOT NULL,
            exploration_count INTEGER NOT NULL DEFAULT 0,
            sum_curiosity REAL NOT NULL DEFAULT 0.0,
            avg_curiosity REAL NOT NULL DEFAULT 0.0,
            last_updated TEXT NOT NULL DEFAULT (datetime('now')),
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_arms_type ON exploration_arms(arm_type);
    """

    def __init__(self, db: Database):
        super().__init__(db)
        self.detector = CuriosityGapDetector(db)
        self._sleep_cycles_completed: int = 0
        self._exploration_log: list[dict[str, Any]] = []
        self._init_schema()

    def _init_schema(self) -> None:
        """Create curiosity-specific tables and add auxiliary columns."""
        self.db.conn.executescript(self.CURIOSITY_SCHEMA_SQL)
        # Add auxiliary columns to episodes (safe — catches if exists)
        aux_episodes = [
            ("episodes", "procedure_id", "TEXT REFERENCES procedures(id)"),
            ("episodes", "is_exploration", "INTEGER NOT NULL DEFAULT 0"),
            ("episodes", "intrinsic_reward", "REAL"),
        ]
        for table, col, col_type in aux_episodes:
            try:
                self.db.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # column already exists
        # Add execution_count to procedures if missing
        try:
            self.db.conn.execute(
                "ALTER TABLE procedures ADD COLUMN execution_count INTEGER NOT NULL DEFAULT 0"
            )
        except Exception:
            pass
        # Add gap_type and target_id to learning_goals for curiosity tracking
        for col, col_type in [("gap_type", "TEXT"), ("target_id", "TEXT")]:
            try:
                self.db.conn.execute(
                    f"ALTER TABLE learning_goals ADD COLUMN {col} {col_type}"
                )
            except Exception:
                pass
        self.db.commit()

    # ── Required overrides ──────────────────────────────────────

    async def execute(self) -> dict[str, Any]:
        """Full curiosity engine run during sleep cycle.

        Phases (spec §7.1):
          1. Detect knowledge gaps (all five types)
          2. Compute curiosity scores (composite weighted)
          3. Apply staleness decay
          4. Persist to curiosity_scores table
          5. Generate learning goals from top gaps
          6. Maintain existing learning goals (age-out, completion)
        """
        created = 0
        modified = 0

        # Phase 1-2: Detect gaps and compute scores
        topics = self.detector.detect_all()

        # Cold-start bootstrap (spec §10.1)
        ep_count = self._get_episode_count()
        if ep_count < COLD_START_EPISODE_LIMIT:
            topics = self._bootstrap_cold_start(topics)

        # Phase 3: Apply staleness decay
        topics = self._apply_staleness_to_topics(topics)

        # Phase 4: Persist curiosity scores
        self._persist_curiosity_scores(topics)
        modified += len(topics)

        # Phase 5: Generate learning goals from top gaps
        top = sorted(topics, key=lambda t: t.curiosity_score, reverse=True)
        if not topics:
            top = []

        for topic in top[:TOP_GAPS_KEPT]:
            if topic.curiosity_score < MIN_CURIOSITY_THRESHOLD:
                continue
            if self._learning_goal_exists(topic):
                existing = self._get_learning_goal_for_topic(topic)
                if existing:
                    old_priority = existing["priority"]
                    new_priority = self._compute_goal_priority(topic)
                    if abs(new_priority - old_priority) > 0.01:
                        self.db.update(
                            "learning_goals",
                            existing["id"],
                            {
                                "priority": new_priority,
                                "updated_at": _now_iso(),
                            },
                        )
                        modified += 1
            else:
                goal_text, episodes_needed = _goal_text_and_needed(topic)
                goal = CuriousGoalModel(
                    domain=topic.domain or "",
                    goal=goal_text,
                    strategy=_format_strategy(topic),
                    priority=self._compute_goal_priority(topic),
                    status=GoalStatus.ACTIVE,
                    episodes_needed=episodes_needed,
                    episodes_collected=0,
                    gap_type=topic.gap_type,
                    target_id=topic.target_id,
                )
                self.db.insert("learning_goals", goal.model_dump())
                created += 1

        # Phase 6: Maintain existing goals
        self._maintain_learning_goals()
        modified += 1

        # Advance sleep cycle counter
        self._sleep_cycles_completed += 1

        return {
            "processed": len(topics),
            "created": created,
            "modified": modified,
            "topics": len(topics),
            "sleep_cycles": self._sleep_cycles_completed,
        }

    async def execute_exploration_cycle(self) -> dict[str, Any]:
        """Run exploration-vs-exploitation decision (epsilon-greedy).

        This is the foreground mode — called during context assembly.

        Returns:
            dict with keys:
                explored: bool — whether exploration was triggered
                topic: str | None — name of explored topic
                curiosity_score: float — score of explored topic
        """
        # Budget check (spec §5.4)
        recent = self._count_recent_explorations(hours=1)
        if recent >= MAX_EXPLORATIONS_PER_HOUR:
            return {
                "explored": False,
                "topic": None,
                "curiosity_score": 0.0,
                "reason": "budget_exceeded",
            }

        # Epsilon-greedy decision (spec §5.1)
        epsilon = compute_epsilon(self._sleep_cycles_completed)
        if random.random() >= epsilon:
            return {
                "explored": False,
                "topic": None,
                "curiosity_score": 0.0,
                "reason": "exploit_mode",
            }

        # Select exploration target (softmax sampling, spec §5.1)
        topics = self._load_curiosity_topics(min_score=MIN_CURIOSITY_THRESHOLD)
        if not topics:
            return {
                "explored": False,
                "topic": None,
                "curiosity_score": 0.0,
                "reason": "no_gaps",
            }

        # Apply fatigue
        for t in topics:
            fatigue = math.exp(-t.exploration_attempts / FATIGUE_HALFLIFE_ATTEMPTS)
            t.curiosity_score *= fatigue

        # Filter out zero-score after fatigue
        topics = [t for t in topics if t.curiosity_score > MIN_CURIOSITY_THRESHOLD]
        if not topics:
            return {
                "explored": False,
                "topic": None,
                "curiosity_score": 0.0,
                "reason": "no_gaps_after_fatigue",
            }

        scores = [t.curiosity_score for t in topics]
        selected = softmax_sample(topics, scores)
        if not selected:
            return {
                "explored": False,
                "topic": None,
                "curiosity_score": 0.0,
                "reason": "softmax_failed",
            }

        # Record this exploration
        self._record_exploration(selected)
        self._increment_exploration_attempt(selected)

        return {
            "explored": True,
            "topic": selected.target_name,
            "curiosity_score": selected.curiosity_score,
            "gap_type": selected.gap_type,
            "target_id": selected.target_id,
            "domain": selected.domain,
        }

    def should_run(self) -> bool:
        """Run the curiosity engine every sleep cycle.

        Also runs if there are pending goals to maintain.
        """
        active_goals = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM learning_goals WHERE status = 'active'",
        )
        has_active_goals = active_goals and active_goals["cnt"] > 0 if active_goals else False

        # Check if it's been a while since last run
        last_run = self.db.fetchone(
            "SELECT completed_at FROM process_runs "
            "WHERE process_name = 'curious_explorer' "
            "AND status = 'completed' "
            "ORDER BY started_at DESC LIMIT 1",
        )
        if not last_run:
            return True  # never run

        return True  # always run during sleep cycle

    # ── Intrinsic motivation (spec §6) ──────────────────────────

    def compute_learning_signal(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Compute intrinsic learning signals from an observed episode.

        Called after myelin_observe saves a new episode.
        Returns dict of signal_type → signal_value.

        Spec §6.3.
        """
        signals: dict[str, Any] = {}

        # 1. Prediction error (if procedure-linked)
        procedure_id = episode.get("procedure_id")
        if procedure_id:
            proc = self.db.fetchone(
                "SELECT predicted_success_rate, actual_success_rate FROM procedures WHERE id = ?",
                (procedure_id,),
            )
            if proc and proc["predicted_success_rate"] is not None:
                actual = 1.0 if episode.get("success", True) else 0.0
                td_error = proc["predicted_success_rate"] - actual
                signals["prediction_error"] = abs(td_error)
                signals["td_error"] = td_error

        # 2. New entity discovered
        episode_id = episode.get("id", "")
        if episode_id:
            new_entities = self.db.fetchall(
                "SELECT id, name FROM entities WHERE id IN ("
                "SELECT entity_id FROM entity_mentions "
                "WHERE source_type = 'episode' AND source_id = ?"
                ") AND mention_count = 1",
                (episode_id,),
            )
            for ent in new_entities:
                signals["new_entity_discovered"] = ent["id"]
                signals["new_entity_name"] = ent["name"]

        # 3. Relationship strengthened past threshold
        if episode_id:
            rels = self.db.fetchall(
                "SELECT r.id, r.evidence_count, r.strength "
                "FROM relationships r "
                "JOIN entity_mentions em1 ON r.source_entity_id = em1.entity_id "
                "WHERE em1.source_id = ? AND em1.source_type = 'episode'",
                (episode_id,),
            )
            for rel in rels:
                if rel["evidence_count"] == 3:  # Just crossed verification threshold
                    signals["relationship_verified"] = rel["id"]

        # 4. Goal progress
        episode_domain = episode.get("domain")
        if episode_domain:
            try:
                active_goals = self.db.fetchall(
                    "SELECT id, gap_type, target_id FROM learning_goals "
                    "WHERE status = 'active' AND (domain = ? OR domain = '')",
                    (episode_domain,),
                )
                for goal in active_goals:
                    signals["goal_progress"] = goal["id"]
            except Exception:
                # Gap_type column may not exist in older schema
                active_goals = self.db.fetchall(
                    "SELECT id FROM learning_goals "
                    "WHERE status = 'active' AND (domain = ? OR domain = '')",
                    (episode_domain,),
                )
                for goal in active_goals:
                    signals["goal_progress"] = goal["id"]

        # 5. New domain discovered
        episode_domain = episode.get("domain")
        if episode_domain:
            existing = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM confidence_map WHERE domain = ?",
                (episode_domain,),
            )
            if not existing or existing["cnt"] == 0:
                signals["new_domain"] = episode_domain

        return signals

    def apply_intrinsic_reward(
        self,
        source_type: str,
        source_id: str,
        reward_value: float,
        trigger_episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply an intrinsic reward signal (spec §6.2).

        Returns dict with outcome details.
        """
        result: dict[str, Any] = {
            "source_type": source_type,
            "source_id": source_id,
            "reward_value": reward_value,
            "applied": False,
        }

        # 1. Record in intrinsic_reward_log
        reward_id = _new_id()
        self.db.insert("intrinsic_reward_log", {
            "id": reward_id,
            "source_type": source_type,
            "source_id": source_id,
            "reward_value": reward_value,
            "metadata": metadata or {},
            "trigger_episode_id": trigger_episode_id,
            "timestamp": _now_iso(),
        })
        result["log_id"] = reward_id

        # 2. Procedure-related: TD update (confidence += α * (reward - confidence))
        if source_type == "procedure":
            proc = self.db.fetchone("SELECT * FROM procedures WHERE id = ?", (source_id,))
            if proc:
                old_conf = proc["confidence"] or 0.5
                td_error = reward_value - old_conf
                new_conf = _clamp(old_conf + ALPHA_IRL * td_error)
                self.db.update("procedures", source_id, {
                    "confidence": new_conf,
                    "updated_at": _now_iso(),
                })
                result["old_confidence"] = old_conf
                result["new_confidence"] = new_conf
                result["td_error"] = td_error
                result["applied"] = True

        # 3. Entity reward: set curiosity floor
        elif source_type == "entity":
            # We track this in curiosity_scores metadata
            result["applied"] = True

        # 4. Learning goal reward: boost priority
        elif source_type == "learning_goal":
            goal = self.db.fetchone(
                "SELECT * FROM learning_goals WHERE id = ?",
                (source_id,),
            )
            if goal:
                new_priority = _clamp(goal["priority"] + 0.05)
                self.db.update("learning_goals", source_id, {
                    "priority": new_priority,
                })
                result["old_priority"] = goal["priority"]
                result["new_priority"] = new_priority
                result["applied"] = True

        # 5. Domain reward
        elif source_type == "domain":
            result["applied"] = True

        return result

    def handle_new_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Full handler called after an episode is observed (spec §7.3).

        Computes learning signals and applies intrinsic rewards.
        """
        signals = self.compute_learning_signal(episode)
        results: dict[str, Any] = {"signals": signals, "rewards": []}

        episode_id = episode.get("id", "")

        for signal_type, signal_value in signals.items():
            if signal_type == "prediction_error":
                procedure_id = episode.get("procedure_id")
                if procedure_id:
                    reward = self.apply_intrinsic_reward(
                        source_type="procedure",
                        source_id=procedure_id,
                        reward_value=0.3 * (1.0 - signal_value),
                        trigger_episode_id=episode_id,
                    )
                    results["rewards"].append(reward)

            elif signal_type == "new_entity_discovered":
                reward = self.apply_intrinsic_reward(
                    source_type="entity",
                    source_id=signal_value,
                    reward_value=0.3,
                    trigger_episode_id=episode_id,
                )
                results["rewards"].append(reward)

            elif signal_type == "relationship_verified":
                reward = self.apply_intrinsic_reward(
                    source_type="relationship",
                    source_id=signal_value,
                    reward_value=0.1,
                    trigger_episode_id=episode_id,
                )
                results["rewards"].append(reward)

            elif signal_type == "goal_progress":
                goal = self.db.fetchone(
                    "SELECT * FROM learning_goals WHERE id = ?",
                    (signal_value,),
                )
                if goal:
                    new_collected = (goal["episodes_collected"] or 0) + 1
                    updates: dict[str, Any] = {
                        "episodes_collected": new_collected,
                    }
                    if new_collected >= (goal["episodes_needed"] or 3):
                        updates["status"] = GoalStatus.ACHIEVED.value
                        updates["resolved_at"] = _now_iso()
                        # Goal completion reward
                        self.apply_intrinsic_reward(
                            source_type="learning_goal",
                            source_id=goal["id"],
                            reward_value=0.5,
                            trigger_episode_id=episode_id,
                        )
                    elif goal.get("status") != GoalStatus.ACHIEVED.value:
                        self.apply_intrinsic_reward(
                            source_type="learning_goal",
                            source_id=goal["id"],
                            reward_value=0.05,
                            trigger_episode_id=episode_id,
                        )
                    self.db.update("learning_goals", goal["id"], updates)
                    results["rewards"].append({
                        "source_type": "learning_goal",
                        "source_id": goal["id"],
                        "applied": True,
                        "updated_status": updates.get("status"),
                    })

            elif signal_type == "new_domain":
                reward = self.apply_intrinsic_reward(
                    source_type="domain",
                    source_id=signal_value,
                    reward_value=0.4,
                    trigger_episode_id=episode_id,
                )
                results["rewards"].append(reward)

        return results

    # ── Private helpers ─────────────────────────────────────────

    def _get_episode_count(self) -> int:
        row = self.db.fetchone("SELECT COUNT(*) as cnt FROM episodes")
        return row["cnt"] if row else 0

    def _bootstrap_cold_start(self, existing: list[CuriosityTopic]) -> list[CuriosityTopic]:
        """Add synthetic cold-start gap when total episodes < 5 (spec §10.1)."""
        if existing:
            return existing
        cold = CuriosityTopic(
            gap_type=GAP_COLD_START,
            target_id="environment",
            target_name="Environment",
            domain=None,
            raw_score=1.0,
            novelty_score=1.0,
            uncertainty_score=1.0,
            infogain_potential=0.5,
            curiosity_score=0.5,
            metadata={"episode_count": self._get_episode_count()},
        )
        return [cold]

    def _apply_staleness_to_topics(self, topics: list[CuriosityTopic]) -> list[CuriosityTopic]:
        """Apply FreshPER staleness decay based on gap age."""
        now = time.time()
        for topic in topics:
            # Check when this gap was last updated in curiosity_scores
            row = self.db.fetchone(
                "SELECT updated_at, exploration_attempts FROM curiosity_scores "
                "WHERE gap_type = ? AND target_id = ?",
                (topic.gap_type, topic.target_id),
            )
            if row:
                updated_str = row["updated_at"] or topic.created_at
                try:
                    from datetime import datetime
                    updated_dt = datetime.strptime(
                        updated_str.split(".")[0], "%Y-%m-%dT%H:%M:%S"
                    )
                    age_days = (datetime.utcnow() - updated_dt).days
                    topic.curiosity_score = apply_staleness_decay(
                        topic.curiosity_score, float(age_days)
                    )
                    topic.exploration_attempts = row["exploration_attempts"] or 0
                    fatigue = math.exp(
                        -topic.exploration_attempts / FATIGUE_HALFLIFE_ATTEMPTS
                    )
                    topic.curiosity_score *= fatigue
                except (ValueError, TypeError):
                    pass
        return topics

    def _persist_curiosity_scores(self, topics: list[CuriosityTopic]) -> None:
        """Upsert topics into curiosity_scores table."""
        for topic in topics:
            existing = self.db.fetchone(
                "SELECT id FROM curiosity_scores WHERE gap_type = ? AND target_id = ?",
                (topic.gap_type, topic.target_id),
            )
            now = _now_iso()
            data = {
                "gap_type": topic.gap_type,
                "target_id": topic.target_id,
                "target_name": topic.target_name,
                "novelty_score": topic.novelty_score,
                "uncertainty_score": topic.uncertainty_score,
                "infogain_potential": topic.infogain_potential,
                "curiosity_score": topic.curiosity_score,
                "domain": topic.domain,
                "raw_score": topic.raw_score,
                "metadata": topic.metadata,
                "exploration_attempts": topic.exploration_attempts,
                "fatigue_factor": topic.fatigue_factor,
                "updated_at": now,
            }
            if existing:
                self.db.update("curiosity_scores", existing["id"], data)
            else:
                data["id"] = _new_id()
                data["created_at"] = now
                self.db.insert("curiosity_scores", data)

    def _learning_goal_exists(self, topic: CuriosityTopic) -> bool:
        row = self.db.fetchone(
            "SELECT id FROM learning_goals "
            "WHERE gap_type = ? AND target_id = ? AND status = 'active'",
            (topic.gap_type, topic.target_id),
        )
        return row is not None

    def _get_learning_goal_for_topic(self, topic: CuriosityTopic) -> dict[str, Any] | None:
        return self.db.fetchone(
            "SELECT * FROM learning_goals "
            "WHERE gap_type = ? AND target_id = ? AND status = 'active' "
            "ORDER BY created_at DESC LIMIT 1",
            (topic.gap_type, topic.target_id),
        )

    def _compute_goal_priority(self, topic: CuriosityTopic) -> float:
        """Goal priority = curiosity_score + bonuses (spec §4.2)."""
        priority = topic.curiosity_score

        # Urgency bonus: +0.15 for declining trend
        if topic.domain:
            cm = self.db.fetchone(
                "SELECT trend FROM confidence_map WHERE domain = ?",
                (topic.domain,),
            )
            if cm and cm.get("trend") == "declining":
                priority += 0.15

        # Recency bonus: +0.10 if gap created within last 24h
        try:
            from datetime import datetime
            created_dt = datetime.strptime(
                topic.created_at.split(".")[0], "%Y-%m-%dT%H:%M:%S"
            )
            age_hours = (datetime.utcnow() - created_dt).total_seconds() / 3600
            if age_hours < 24:
                priority += 0.10
        except (ValueError, TypeError):
            pass

        return _clamp(priority)

    def _maintain_learning_goals(self) -> None:
        """Maintain active learning goals (spec §4.5).

        1. Update episodes_collected
        2. Check success conditions
        3. Age out abandoned goals
        4. Merge duplicates
        """
        from datetime import datetime

        now = datetime.utcnow()

        # Get active goals
        goals = self.db.fetchall(
            "SELECT * FROM learning_goals WHERE status = 'active'"
        )

        for goal in goals:
            goal_id = goal["id"]
            updates: dict[str, Any] = {}
            needs_update = False

            # 1. Update episodes_collected
            count_row = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM episodes WHERE domain = ?",
                (goal["domain"],),
            )
            if count_row and count_row["cnt"] > (goal["episodes_collected"] or 0):
                updates["episodes_collected"] = count_row["cnt"]
                needs_update = True

            # 2. Check success: episodes_collected >= episodes_needed
            collected = updates.get("episodes_collected", goal["episodes_collected"] or 0)
            if collected >= (goal["episodes_needed"] or 3):
                updates["status"] = GoalStatus.ACHIEVED.value
                updates["resolved_at"] = _now_iso()
                needs_update = True

            # 3. Age > 30 days: abandon
            created_str = goal.get("created_at", "")
            if created_str:
                try:
                    created_dt = datetime.strptime(
                        created_str.split(".")[0], "%Y-%m-%dT%H:%M:%S"
                    )
                    age_days = (now - created_dt).days
                    if age_days >= GOAL_ABANDONMENT_DAYS:
                        updates["status"] = GoalStatus.ABANDONED.value
                        updates["resolved_at"] = _now_iso()
                        needs_update = True
                except (ValueError, TypeError):
                    pass

            # 4. Priority < 0.1: abandon
            if (goal["priority"] or 0.0) < MIN_CURIOSITY_THRESHOLD:
                updates["status"] = GoalStatus.ABANDONED.value
                updates["resolved_at"] = _now_iso()
                needs_update = True

            if needs_update:
                self.db.update("learning_goals", goal_id, updates)

    def _count_recent_explorations(self, hours: int = 1) -> int:
        """Count exploration-triggered episodes in the last N hours."""
        from datetime import datetime, timedelta

        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        row = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM episodes "
            "WHERE is_exploration = 1 AND timestamp >= ?",
            (cutoff,),
        )
        return row["cnt"] if row else 0

    def _record_exploration(self, topic: CuriosityTopic) -> None:
        """Record that an exploration was performed."""
        self._exploration_log.append({
            "gap_type": topic.gap_type,
            "target_id": topic.target_id,
            "target_name": topic.target_name,
            "curiosity_score": topic.curiosity_score,
            "timestamp": _now_iso(),
        })

    def _increment_exploration_attempt(self, topic: CuriosityTopic) -> None:
        """Increment exploration_attempts for a topic in curiosity_scores."""
        existing = self.db.fetchone(
            "SELECT id, exploration_attempts FROM curiosity_scores "
            "WHERE gap_type = ? AND target_id = ?",
            (topic.gap_type, topic.target_id),
        )
        if existing:
            attempts = (existing["exploration_attempts"] or 0) + 1
            self.db.update("curiosity_scores", existing["id"], {
                "exploration_attempts": attempts,
                "last_explored_at": _now_iso(),
                "fatigue_factor": math.exp(-attempts / FATIGUE_HALFLIFE_ATTEMPTS),
                "updated_at": _now_iso(),
            })

    def _load_curiosity_topics(
        self,
        min_score: float = 0.0,
        limit: int = 50,
    ) -> list[CuriosityTopic]:
        """Load curiosity topics from the curiosity_scores table."""
        rows = self.db.fetchall(
            "SELECT * FROM curiosity_scores "
            "WHERE curiosity_score >= ? "
            "ORDER BY curiosity_score DESC "
            "LIMIT ?",
            (min_score, limit),
        )
        result = []
        for r in rows:
            meta = r.get("metadata", {})
            if isinstance(meta, str):
                import json
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            result.append(CuriosityTopic(
                gap_type=r["gap_type"],
                target_id=r["target_id"],
                target_name=r.get("target_name", ""),
                domain=r.get("domain"),
                raw_score=r.get("raw_score", 0.0),
                novelty_score=r.get("novelty_score", 0.0),
                uncertainty_score=r.get("uncertainty_score", 0.0),
                infogain_potential=r.get("infogain_potential", 0.0),
                curiosity_score=r.get("curiosity_score", 0.0),
                exploration_attempts=r.get("exploration_attempts", 0),
                fatigue_factor=r.get("fatigue_factor", 1.0),
                metadata=meta,
                created_at=r.get("created_at", _now_iso()),
            ))
        return result

    def get_curiosity_state(
        self,
        domain: str | None = None,
        gap_type: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Query the current curiosity state — top gaps and scores."""
        conditions = ["curiosity_score >= ?"]
        params: list[Any] = [MIN_CURIOSITY_THRESHOLD]
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if gap_type:
            conditions.append("gap_type = ?")
            params.append(gap_type)
        where = " AND ".join(conditions)
        rows = self.db.fetchall(
            f"SELECT * FROM curiosity_scores "
            f"WHERE {where} "
            f"ORDER BY curiosity_score DESC LIMIT ?",
            tuple(params + [limit]),
        )
        return {
            "top_gaps": [dict(r) for r in rows],
            "total_count": len(rows),
            "epsilon": compute_epsilon(self._sleep_cycles_completed),
            "sleep_cycles": self._sleep_cycles_completed,
        }
