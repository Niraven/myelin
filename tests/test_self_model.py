"""Tests for SelfModel — confidence calibration and bias detection."""

import json
from datetime import datetime
from uuid import uuid4

from myelin.core.database import Database
from myelin.core.models import ProcessName
from myelin.memory.episodic import EpisodicMemory
from myelin.memory.procedural import ProceduralMemory
from myelin.memory.semantic import SemanticMemory
from myelin.cognitive.self_model import (
    assess_domain_competence,
    compute_bias_report,
    detect_confirmation_bias,
    detect_overconfidence,
    detect_recency_bias,
    generate_insights,
    SelfModel,
)


# ── assess_domain_competence ───────────────────────────────────────


def test_assess_novel_domain():
    """Low coverage = novel level."""
    r = assess_domain_competence(1, 0, 0.5, 0.5, 0.0, 0.0)
    assert r["level"] == "novel"
    assert r["competence_score"] < 0.4


def test_assess_competent_domain():
    """Moderate coverage + good quality = competent."""
    r = assess_domain_competence(25, 5, 0.85, 0.7, 0.02, 0.03)
    assert r["level"] in ("competent", "mastered")
    assert 0.5 <= r["competence_score"] <= 1.0


def test_assess_mastered_domain():
    """High coverage + excellent quality = mastered."""
    r = assess_domain_competence(50, 10, 0.95, 0.9, 0.01, 0.1)
    assert r["level"] == "mastered"
    assert r["competence_score"] >= 0.85


def test_assess_trend_improving():
    """Positive trend delta = improving."""
    r = assess_domain_competence(20, 3, 0.8, 0.7, 0.0, 0.1)
    assert r["trend"] == "improving"


def test_assess_trend_declining():
    """Negative trend delta = declining."""
    r = assess_domain_competence(20, 3, 0.8, 0.7, 0.0, -0.1)
    assert r["trend"] == "declining"


def test_assess_trend_stable():
    """Small trend delta = stable."""
    r = assess_domain_competence(20, 3, 0.8, 0.7, 0.0, 0.01)
    assert r["trend"] == "stable"


def test_assess_calibration_included():
    """Calibration offset included in result."""
    r = assess_domain_competence(10, 2, 0.7, 0.6, 0.12, 0.0)
    assert "calibration_offset" in r
    assert r["calibration_offset"] == 0.12


# ── Bias Detection ─────────────────────────────────────────────────


def test_no_confirmation_bias():
    """No confirmation bias when there are no failures."""
    assert detect_confirmation_bias([True, True], [0.9, 0.8]) == 0.0


def test_high_confirmation_bias():
    """Confidence stays high after failures = confirmation bias."""
    bias = detect_confirmation_bias(
        [False, False, False, True], [0.7, 0.65, 0.6, 0.9]
    )
    assert bias > 0.3


def test_no_recency_bias():
    """Similar recent and overall rates = no bias."""
    assert detect_recency_bias(0.7, 0.7) < 0.1


def test_high_recency_bias():
    """Large gap between recent and overall = recency bias."""
    bias = detect_recency_bias(0.9, 0.3)
    assert bias > 0.5


def test_no_overconfidence():
    """Negative or zero calibration offset = no overconfidence."""
    assert detect_overconfidence(0.0) == 0.0
    assert detect_overconfidence(-0.1) == 0.0


def test_high_overconfidence():
    """Large positive calibration offset = overconfidence."""
    bias = detect_overconfidence(0.4)
    assert bias > 0.5


def test_compute_bias_report_no_biases():
    """Clean data produces no flagged biases."""
    report = compute_bias_report(
        [True, True], [0.8, 0.9], 0.85, 0.85, 0.02
    )
    assert len(report["biases"]) == 0


def test_compute_bias_report_with_biases():
    """Problematic data flags biases."""
    report = compute_bias_report(
        [False, False, False], [0.7, 0.65, 0.6], 1.0, 0.3, 0.4
    )
    assert len(report["biases"]) >= 1


# ── generate_insights ──────────────────────────────────────────────


def test_insights_novel_domain():
    """Novel domain generates explore recommendation."""
    insights = generate_insights({
        "testing": {"level": "novel", "competence_score": 0.2,
                     "episode_count": 2, "calibration_offset": 0.0, "trend": "stable"},
    })
    assert any("novel" in i for i in insights)


def test_insights_mastered_domain():
    """Mastered domain gets maintain recommendation."""
    insights = generate_insights({
        "deployment": {"level": "mastered", "competence_score": 0.9,
                        "episode_count": 60, "calibration_offset": 0.02, "trend": "improving"},
    })
    assert any("mastered" in i for i in insights)


# ── SelfModel Integration ──────────────────────────────────────────


def _new_id():
    return uuid4().hex[:16]


def _now_iso():
    return datetime.utcnow().isoformat()


def test_self_model_execute_empty_db(tmp_path):
    """SelfModel execute handles empty DB gracefully."""
    db = Database(path=tmp_path / "test.db", enable_vec=False)
    _ = db.conn
    model = SelfModel(db, EpisodicMemory(db), SemanticMemory(db), ProceduralMemory(db))
    import asyncio
    result = asyncio.run(model.execute())
    assert result["domains_assessed"] == 0
    assert result["biases_detected"] == 0


def test_self_model_with_data(tmp_path):
    """SelfModel assess domain from episodes."""
    db = Database(path=tmp_path / "test.db", enable_vec=False)
    _ = db.conn
    episodic = EpisodicMemory(db)
    semantic = SemanticMemory(db)
    procedural = ProceduralMemory(db)

    # Insert some episodes
    from myelin.core.models import Episode, ActionType
    for i in range(5):
        ep = Episode(
            agent_id="test",
            session_id="s1",
            action=f"test action {i}",
            action_type=ActionType.TOOL_CALL,
            content_text=f"test content {i}",
            success=True,
            domain="testing",
        )
        episodic.record(ep)

    model = SelfModel(db, episodic, semantic, procedural)
    import asyncio
    result = asyncio.run(model.execute())
    assert result["domains_assessed"] >= 1
    assert "testing" in result.get("competence_map", {})
