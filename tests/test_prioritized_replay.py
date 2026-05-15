"""Tests for PrioritizedReplay — PER sampling, IS weights, staleness prevention.

Uses in-memory SQLite; no side effects.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from myelin.core.database import Database
from myelin.core.schema import SCHEMA_SQL
from myelin.cognitive.prioritized_replay import (
    PrioritizedReplay,
    _hours_since,
    FRESHPER_DECAY,
    PRIORITY_FLOOR,
    MAX_REPLAY_COUNT,
    W_TD_ERROR,
    W_SURPRISE,
    W_IMPORTANCE,
    IMPORTANCE_HALF_LIFE_HOURS,
    PER_ALPHA,
    IS_BETA_START,
    IS_BETA_END,
    BATCH_SIZE,
)
from myelin.knowledge.entities import EntityStore
from myelin.knowledge.graph import KnowledgeGraph


def _new_id() -> str:
    return uuid4().hex[:16]


def _make_db() -> Database:
    db = Database(":memory:")
    db.conn.executescript(SCHEMA_SQL)
    return db


def _add_episode(
    db: Database,
    action: str = "test_action",
    content: str = "using git and docker",
    domain: str = "testing",
    success: bool = True,
    priority_score: float | None = None,
    importance_score: float = 0.5,
    td_error: float | None = None,
    surprise_score: float | None = None,
    replay_count: int = 0,
    timestamp: str | None = None,
):
    ep_id = _new_id()
    db.insert(
        "episodes",
        {
            "id": ep_id,
            "agent_id": "test_agent",
            "session_id": "test_session",
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "action": action,
            "action_type": "tool_call",
            "content_text": content,
            "success": int(success),
            "priority_score": priority_score,
            "importance_score": importance_score,
            "td_error": td_error,
            "surprise_score": surprise_score,
            "replay_count": replay_count,
            "access_times": json.dumps([time.time()]),
            "access_count": 1,
            "last_accessed": datetime.utcnow().isoformat(),
            "tags": "[]",
            "domain": domain,
        },
    )
    return ep_id


def _add_entity(
    db: Database,
    name: str = "test_entity",
    entity_type: str = "tool",
    canonical_name: str | None = None,
    domain: str | None = "testing",
):
    ent_id = _new_id()
    db.insert(
        "entities",
        {
            "id": ent_id,
            "name": name,
            "entity_type": entity_type,
            "canonical_name": canonical_name or name.lower(),
            "mention_count": 1,
            "domain": domain,
            "access_times": "[]",
            "source_episodes": "[]",
            "first_seen": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
        },
    )
    return ent_id


def _make_replay(tmp_db) -> PrioritizedReplay:
    return PrioritizedReplay(
        db=tmp_db,
        entity_store=EntityStore(tmp_db),
        graph=KnowledgeGraph(tmp_db),
    )


# ── _hours_since ───────────────────────────────────────────────────


def test_hours_since_recent():
    now = datetime.utcnow().isoformat()
    assert _hours_since(now) < 0.01


def test_hours_since_old():
    old = (datetime.utcnow() - timedelta(hours=48)).isoformat()
    h = _hours_since(old)
    assert 47.0 < h < 49.0


def test_hours_since_invalid():
    assert _hours_since("") == 0.0
    assert _hours_since("garbage") == 0.0


# ── Priority Scoring ───────────────────────────────────────────────


def test_compute_priority_td_error_only(tmp_db):
    ep_id = _add_episode(tmp_db, td_error=0.8, importance_score=0.5)
    ep = tmp_db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep_id,))
    replay = _make_replay(tmp_db)
    priority = replay._compute_priority(ep)
    # td_term: 0.35 * 0.8 = 0.28, importance term: 0.35*0.5*~1.0 = 0.175
    assert 0.40 < priority < 0.50


def test_compute_priority_surprise_only(tmp_db):
    ep_id = _add_episode(tmp_db, td_error=None, surprise_score=0.9, importance_score=0.5)
    ep = tmp_db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep_id,))
    replay = _make_replay(tmp_db)
    priority = replay._compute_priority(ep)
    # surprise_term: 0.30 * 0.9 = 0.27, importance_term: 0.35*0.5 = 0.175
    assert 0.40 < priority < 0.50


def test_compute_priority_no_signals(tmp_db):
    ep_id = _add_episode(tmp_db, td_error=None, surprise_score=None, importance_score=0.5)
    ep = tmp_db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep_id,))
    replay = _make_replay(tmp_db)
    priority = replay._compute_priority(ep)
    # Only importance term: 0.35 * 0.5 = 0.175
    assert 0.15 < priority < 0.20


def test_compute_priority_staleness_penalty(tmp_db):
    ep_id = _add_episode(tmp_db, td_error=1.0, importance_score=0.5, replay_count=5)
    ep = tmp_db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep_id,))
    replay = _make_replay(tmp_db)
    priority = replay._compute_priority(ep)
    staleness = math.pow(FRESHPER_DECAY, 5)
    expected_raw = W_TD_ERROR * 1.0 + W_IMPORTANCE * 0.5
    expected = expected_raw * staleness
    assert abs(priority - expected) < 0.01


def test_compute_priority_floor(tmp_db):
    """Even with zero signals and high replay, priority should be at floor."""
    ep_id = _add_episode(
        tmp_db,
        td_error=0.0,
        surprise_score=0.0,
        importance_score=0.0,
        replay_count=MAX_REPLAY_COUNT,
    )
    ep = tmp_db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep_id,))
    replay = _make_replay(tmp_db)
    priority = replay._compute_priority(ep)
    assert priority >= PRIORITY_FLOOR


def test_priority_decay_over_time(tmp_db):
    old_ts = (datetime.utcnow() - timedelta(days=14)).isoformat()
    ep_id = _add_episode(
        tmp_db,
        td_error=None,
        surprise_score=None,
        importance_score=1.0,
        timestamp=old_ts,
    )
    ep = tmp_db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep_id,))
    replay = _make_replay(tmp_db)
    priority = replay._compute_priority(ep)
    # 14 days = 336 hours, decay = exp(-336/168) = exp(-2) ≈ 0.135
    # importance_term = 0.35 * 1.0 * 0.135 ≈ 0.047
    # floor is 0.1, so priority should hit the floor
    assert priority == pytest.approx(PRIORITY_FLOOR, abs=0.01)


# ── _compute_priorities ─────────────────────────────────────────────


def test_compute_priorities_updates_db(tmp_db):
    _add_episode(tmp_db, td_error=0.9, importance_score=0.8)
    _add_episode(tmp_db, td_error=0.1, importance_score=0.2)
    replay = _make_replay(tmp_db)
    scored = replay._compute_priorities()
    assert len(scored) == 2
    # First should have higher priority
    assert scored[0]["priority_score"] > scored[1]["priority_score"]


def test_compute_priorities_excludes_high_replay(tmp_db):
    _add_episode(tmp_db, td_error=0.9, replay_count=0)
    _add_episode(tmp_db, td_error=0.8, replay_count=MAX_REPLAY_COUNT)
    replay = _make_replay(tmp_db)
    scored = replay._compute_priorities()
    assert len(scored) == 1  # only the one with replay_count < MAX


# ── IS Beta Annealing ───────────────────────────────────────────────


def test_beta_initial():
    replay = _make_replay(None)
    beta = replay._beta(1)
    assert beta == pytest.approx(IS_BETA_START, abs=0.01)


def test_beta_anneals():
    replay = _make_replay(None)
    b1 = replay._beta(1)
    b50 = replay._beta(50)
    b200 = replay._beta(200)
    assert b1 < b50
    assert b50 < b200
    assert b200 <= IS_BETA_END


def test_beta_caps():
    replay = _make_replay(None)
    assert replay._beta(999) <= IS_BETA_END


# ── Sampling ────────────────────────────────────────────────────────


def test_sample_batch_returns_correct_size(tmp_db):
    for _ in range(30):
        _add_episode(
            tmp_db,
            td_error=0.5,
            importance_score=0.5,
        )
    replay = _make_replay(tmp_db)
    replay._compute_priorities()
    batch = replay._sample_batch(cycle_count=1)
    assert len(batch) == BATCH_SIZE
    for ep in batch:
        assert "_is_weight" in ep
        assert "_rank" in ep
        assert "_prob" in ep


def test_sample_batch_less_than_batch_size(tmp_db):
    for _ in range(5):
        _add_episode(tmp_db, td_error=0.5, importance_score=0.5)
    replay = _make_replay(tmp_db)
    replay._compute_priorities()
    batch = replay._sample_batch(cycle_count=1)
    assert 1 <= len(batch) <= 5


def test_sample_batch_empty(tmp_db):
    replay = _make_replay(tmp_db)
    batch = replay._sample_batch(cycle_count=1)
    assert batch == []


@pytest.mark.asyncio
async def test_execute_no_episodes(tmp_db):
    replay = _make_replay(tmp_db)
    result = await replay.execute()
    assert result["sampled"] == 0
    assert result["cycle_count"] == 1


# ── Replay Loop ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_replays_and_increments_count(tmp_db):
    _add_episode(tmp_db, td_error=0.9, importance_score=0.9)
    _add_episode(tmp_db, td_error=0.8, importance_score=0.8)
    _add_episode(tmp_db, td_error=0.7, importance_score=0.7)
    replay = _make_replay(tmp_db)
    result = await replay.execute()
    assert result["sampled"] > 0
    assert result["replayed"] > 0
    # Check replay_count was incremented on at least one episode
    ep = tmp_db.fetchone("SELECT * FROM episodes LIMIT 1")
    assert ep["replay_count"] == 1


@pytest.mark.asyncio
async def test_execute_strengthens_entities(tmp_db):
    # Create two entities and an episode that mentions them
    # Use pytest (standalone match) and git commit (verb+noun pattern)
    ent1 = _add_entity(tmp_db, name="pytest", canonical_name="pytest")
    ent2 = _add_entity(tmp_db, name="git commit", canonical_name="git commit")
    _add_episode(tmp_db, content="run pytest and git commit together")
    replay = _make_replay(tmp_db)
    result = await replay.execute()
    assert result["entities_strengthened"] >= 1


@pytest.mark.asyncio
async def test_execute_cycle_count_increments(tmp_db):
    _add_episode(tmp_db, td_error=0.9, importance_score=0.9)
    replay = _make_replay(tmp_db)
    r1 = await replay.run()
    assert r1["cycle_count"] == 1
    r2 = await replay.run()
    assert r2["cycle_count"] == 2


@pytest.mark.asyncio
async def test_execute_schema_hints(tmp_db):
    # Create many episodes in same cluster, all high priority
    cluster_id = "test_cluster_1"
    for _ in range(10):
        ep_id = _new_id()
        tmp_db.insert(
            "episodes",
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "session_id": "test_session",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "test",
                "action_type": "tool_call",
                "content_text": "testing something",
                "success": 1,
                "priority_score": 0.9,
                "importance_score": 0.9,
                "td_error": 0.8,
                "replay_count": 0,
                "cluster_id": cluster_id,
                "access_times": "[]",
                "access_count": 1,
                "last_accessed": datetime.utcnow().isoformat(),
                "tags": "[]",
                "domain": "testing",
            },
        )
    replay = _make_replay(tmp_db)
    result = await replay.execute()
    # Schema hints are tracked but in practice may trigger with BATCH_SIZE=20
    assert result["sampled"] > 0


# ── End-to-end ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_replay_cycle(tmp_db):
    """End-to-end: add episodes, compute priorities, sample, replay, check results."""
    # Add diverse episodes
    for i in range(25):
        _add_episode(
            tmp_db,
            action=f"action_{i}",
            content=f"run pytest and git commit iteration {i}",
            td_error=0.1 + (i / 25) * 0.9,
            importance_score=0.3 + (i / 25) * 0.7,
        )

    # Add entities that match the content
    _add_entity(tmp_db, name="git", canonical_name="git")
    _add_entity(tmp_db, name="docker", canonical_name="docker")

    replay = _make_replay(tmp_db)
    result = await replay.run()

    assert result["sampled"] == BATCH_SIZE
    assert result["replayed"] == BATCH_SIZE
    assert result["cycle_count"] == 1
    assert result["beta"] == pytest.approx(IS_BETA_START, abs=0.01)
    assert result["total_scored"] >= 25

    # Run a second cycle to verify beta annealing
    r2 = await replay.run()
    assert r2["cycle_count"] == 2
    assert r2["beta"] > result["beta"]


@pytest.mark.asyncio
async def test_stale_episodes_excluded(tmp_db):
    """Episodes with replay_count >= MAX_REPLAY_COUNT should be excluded."""
    # Add one episode at max replay count
    _add_episode(tmp_db, td_error=1.0, replay_count=MAX_REPLAY_COUNT)
    _add_episode(tmp_db, td_error=0.9, replay_count=0)
    replay = _make_replay(tmp_db)
    result = await replay.execute()
    assert result["sampled"] == 1
    assert result["replayed"] == 1


@pytest.mark.asyncio
async def test_priority_floor_prevents_zero(tmp_db):
    """Even low-priority episodes should have a minimum priority."""
    ep_id = _add_episode(
        tmp_db,
        td_error=0.0,
        surprise_score=0.0,
        importance_score=0.0,
        replay_count=MAX_REPLAY_COUNT,
    )
    replay = _make_replay(tmp_db)
    priority = replay._compute_priority(
        tmp_db.fetchone("SELECT * FROM episodes WHERE id = ?", (ep_id,))
    )
    assert priority >= PRIORITY_FLOOR
