"""Schema migration coverage for pre-Learning OS databases."""

import json
import sqlite3

import pytest

from myelin.core.database import Database
from myelin.core.models import Procedure, ProcedureStatus, ProcedureStep
from myelin.core.schema import SCHEMA_VERSION
from myelin.memory.procedural import ProceduralMemory
from myelin.session import Session

OLD_SCHEMA_SQL = """
CREATE TABLE schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT INTO schema_meta (key, value) VALUES ('version', '3');

CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    action TEXT NOT NULL,
    action_type TEXT NOT NULL,
    input_context TEXT,
    output_result TEXT,
    success INTEGER NOT NULL DEFAULT 1,
    content_text TEXT NOT NULL,
    embedding BLOB,
    access_count INTEGER NOT NULL DEFAULT 1,
    access_times TEXT NOT NULL DEFAULT '[]',
    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
    consolidated INTEGER NOT NULL DEFAULT 0,
    cluster_id TEXT,
    tags TEXT DEFAULT '[]',
    domain TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE semantic_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    source_type TEXT NOT NULL,
    source_ids TEXT NOT NULL DEFAULT '[]',
    access_count INTEGER NOT NULL DEFAULT 1,
    access_times TEXT NOT NULL DEFAULT '[]',
    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
    confidence REAL NOT NULL DEFAULT 0.5,
    valid_from TEXT,
    valid_until TEXT,
    superseded_by TEXT,
    domain TEXT,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE procedures (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    trigger_pattern TEXT NOT NULL,
    trigger_embedding BLOB,
    steps TEXT NOT NULL,
    preconditions TEXT DEFAULT '[]',
    postconditions TEXT DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    predicted_success_rate REAL,
    actual_success_rate REAL,
    calibration_offset REAL DEFAULT 0.0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    modify_count INTEGER NOT NULL DEFAULT 0,
    activation_score REAL NOT NULL DEFAULT 0.0,
    access_times TEXT NOT NULL DEFAULT '[]',
    last_executed TEXT,
    source_agent TEXT NOT NULL,
    source_episodes TEXT DEFAULT '[]',
    promotion_method TEXT DEFAULT 'auto',
    is_composite INTEGER NOT NULL DEFAULT 0,
    component_procedures TEXT DEFAULT '[]',
    parent_procedures TEXT DEFAULT '[]',
    transferred_to TEXT DEFAULT '[]',
    transfer_success_rate REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'draft',
    version INTEGER NOT NULL DEFAULT 1,
    domain TEXT,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE learning_goals (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    goal TEXT NOT NULL,
    strategy TEXT,
    priority REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    episodes_needed INTEGER DEFAULT 3,
    episodes_collected INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT
);
"""


def _columns(db: Database, table: str) -> set[str]:
    return {row["name"] for row in db.fetchall(f"PRAGMA table_info({table})")}


def _create_v3_database(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO episodes "
        "(id, agent_id, session_id, action, action_type, content_text, access_times, domain) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-episode",
            "legacy-agent",
            "legacy-session",
            "legacy deploy",
            "tool_call",
            "legacy deploy workflow from old schema",
            json.dumps([]),
            "deployment",
        ),
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_v3_database_migrates_and_core_flows_work(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_v3_database(db_path)

    db = Database(path=db_path, enable_vec=False)
    _ = db.conn

    assert db.fetchone("SELECT value FROM schema_meta WHERE key = 'version'")["value"] == str(
        SCHEMA_VERSION
    )
    assert {
        "importance_score",
        "td_error",
        "surprise_score",
        "priority_score",
        "replay_count",
        "labile_until",
    } <= _columns(db, "episodes")
    assert {"labile_until", "prediction_error", "last_pe_raw", "last_update_mode"} <= _columns(
        db, "semantic_nodes"
    )
    assert {"prediction_error", "surprise_score", "total_pe_sum", "pe_count"} <= _columns(
        db, "procedures"
    )

    old_results = db.fts_search("episodes", "episodes_fts", "legacy deploy", limit=3)
    assert any(result["id"] == "legacy-episode" for result in old_results)

    session = Session(db, agent_id="migration-agent", session_id="migration-session")
    episode_id = await session.observe(
        action="pytest tests -q",
        action_type="tool_call",
        content_text="Ran pytest after migration",
        domain="testing",
    )
    assert episode_id
    end_result = await session.end()
    assert not [
        result
        for result in end_result["cognitive_results"]
        if "no such column" in result.get("error", "").lower()
    ]

    procedural = ProceduralMemory(db)
    procedure = Procedure(
        name="migration_check",
        trigger_pattern="run migrated procedure",
        steps=[ProcedureStep(order=1, description="run pytest tests -q")],
        source_agent="migration-agent",
        status=ProcedureStatus.ACTIVE,
        confidence=0.7,
    )
    proc_id = procedural.store(procedure)
    assert procedural.record_execution(proc_id, success=True) > 0.7

    db.close()
