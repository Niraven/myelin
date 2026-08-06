"""Myelin V4 SQLite schema — full learning OS foundation.

SCHEMA_VERSION 4 adds:
- Reconsolidation: lability windows, prediction error tracking, update modes
- Two-phase sleep: NREM (strengthening, downscaling) + REM (counterfactual simulation)
- Prioritized replay: priority scoring, replay tracking, surprise signals
- Prediction error learning: TD-error tracking, forward model logging
- Schema learning: abstract behavioral patterns from semantic clusters
"""

from contextlib import suppress

SCHEMA_VERSION = 4

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ============================================================
-- EPISODIC MEMORY
-- Raw observations with learning OS extensions (V4).
-- ============================================================

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),

    -- What happened
    action TEXT NOT NULL,
    action_type TEXT NOT NULL,
    input_context TEXT,
    output_result TEXT,
    success INTEGER NOT NULL DEFAULT 1,

    -- Semantic search
    content_text TEXT NOT NULL,
    embedding BLOB,

    -- ACT-R activation
    access_count INTEGER NOT NULL DEFAULT 1,
    access_times TEXT NOT NULL DEFAULT '[]',
    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),

    -- Consolidation state
    consolidated INTEGER NOT NULL DEFAULT 0,
    cluster_id TEXT,

    -- V3: Importance scoring
    importance_score REAL NOT NULL DEFAULT 0.5,

    -- V4: Learning OS extensions
    td_error REAL,                      -- Prediction error from procedure execution
    surprise_score REAL,                -- Surprise signal (0.0-1.0)
    priority_score REAL DEFAULT 0.5,    -- Composite priority for replay
    replay_count INTEGER DEFAULT 0,     -- Times consumed in replay
    labile_until TEXT,                  -- Reconsolidation lability window expiry

    -- Metadata
    tags TEXT DEFAULT '[]',
    domain TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(domain);
CREATE INDEX IF NOT EXISTS idx_episodes_cluster ON episodes(cluster_id);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated);
CREATE INDEX IF NOT EXISTS idx_episodes_labile ON episodes(labile_until);
CREATE INDEX IF NOT EXISTS idx_episodes_priority ON episodes(priority_score DESC);

-- ============================================================
-- SEMANTIC MEMORY
-- Facts and reflections with reconsolidation support (V4).
-- ============================================================

CREATE TABLE IF NOT EXISTS semantic_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,

    -- Source tracking
    source_type TEXT NOT NULL,
    source_ids TEXT NOT NULL DEFAULT '[]',

    -- ACT-R activation
    access_count INTEGER NOT NULL DEFAULT 1,
    access_times TEXT NOT NULL DEFAULT '[]',
    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),

    -- Confidence and validity
    confidence REAL NOT NULL DEFAULT 0.5,
    valid_from TEXT,
    valid_until TEXT,
    superseded_by TEXT,

    -- V4: Reconsolidation + prediction error
    labile_until TEXT,
    prediction_error REAL,
    last_pe_raw REAL,
    last_update_mode TEXT,

    -- Organization
    domain TEXT,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_semantic_domain ON semantic_nodes(domain);
CREATE INDEX IF NOT EXISTS idx_semantic_confidence ON semantic_nodes(confidence);
CREATE INDEX IF NOT EXISTS idx_semantic_labile ON semantic_nodes(labile_until);

-- ============================================================
-- PROCEDURAL MEMORY
-- Learned workflows with prediction error tracking (V4).
-- ============================================================

CREATE TABLE IF NOT EXISTS procedures (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,

    -- Trigger matching
    trigger_pattern TEXT NOT NULL,
    trigger_embedding BLOB,

    -- Procedure body
    steps TEXT NOT NULL,
    preconditions TEXT DEFAULT '[]',
    postconditions TEXT DEFAULT '[]',

    -- Bayesian confidence
    confidence REAL NOT NULL DEFAULT 0.5,
    predicted_success_rate REAL,
    actual_success_rate REAL,
    calibration_offset REAL DEFAULT 0.0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    modify_count INTEGER NOT NULL DEFAULT 0,

    -- ACT-R activation
    activation_score REAL NOT NULL DEFAULT 0.0,
    access_times TEXT NOT NULL DEFAULT '[]',
    last_executed TEXT,

    -- V4: Prediction error learning
    prediction_error REAL,              -- Last TD-error magnitude
    surprise_score REAL,                -- Surprise from last execution
    total_pe_sum REAL DEFAULT 0.0,      -- Running sum for PE tracking
    pe_count INTEGER DEFAULT 0,         -- Number of prediction errors recorded

    -- Provenance
    source_agent TEXT NOT NULL,
    source_episodes TEXT DEFAULT '[]',
    promotion_method TEXT DEFAULT 'auto',

    -- Composition
    is_composite INTEGER NOT NULL DEFAULT 0,
    component_procedures TEXT DEFAULT '[]',
    parent_procedures TEXT DEFAULT '[]',

    -- Transfer tracking
    transferred_to TEXT DEFAULT '[]',
    transfer_success_rate REAL DEFAULT 0.0,

    -- Lifecycle
    status TEXT NOT NULL DEFAULT 'draft',
    version INTEGER NOT NULL DEFAULT 1,
    domain TEXT,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_procedures_status ON procedures(status);
CREATE INDEX IF NOT EXISTS idx_procedures_domain ON procedures(domain);
CREATE INDEX IF NOT EXISTS idx_procedures_confidence ON procedures(confidence);
CREATE INDEX IF NOT EXISTS idx_procedures_activation ON procedures(activation_score);
CREATE INDEX IF NOT EXISTS idx_procedures_source_agent ON procedures(source_agent);
CREATE INDEX IF NOT EXISTS idx_procedures_composite ON procedures(is_composite);

-- ============================================================
-- PROCEDURE EVIDENCE / TRUST LIFECYCLE
-- Tracks each procedure execution, feedback, or approval event.
-- ============================================================

CREATE TABLE IF NOT EXISTS procedure_evidence (
    id TEXT PRIMARY KEY,
    procedure_id TEXT NOT NULL,
    source TEXT NOT NULL,                 -- 'execution', 'feedback', 'approval'
    outcome TEXT NOT NULL,                -- 'success', 'failure', 'partial'
    confidence_delta REAL NOT NULL DEFAULT 0.0,
    episode_id TEXT,
    prediction_id TEXT,                -- Provenance: links to prediction_log for verified feedback
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (procedure_id) REFERENCES procedures(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pe_procedure ON procedure_evidence(procedure_id);
CREATE INDEX IF NOT EXISTS idx_pe_source ON procedure_evidence(source);
CREATE INDEX IF NOT EXISTS idx_pe_timestamp ON procedure_evidence(timestamp);
-- Provenance must be unique: a prediction_log id may back at most one evidence row.
-- Partial so legacy NULL (unbound) evidence is unaffected.
CREATE UNIQUE INDEX IF NOT EXISTS idx_pe_prediction_unique
    ON procedure_evidence(prediction_id) WHERE prediction_id IS NOT NULL;

-- ============================================================
-- V4: RECONSOLIDATION LOG
-- Tracks every reconsolidation event with before/after snapshots.
-- ============================================================

CREATE TABLE IF NOT EXISTS reconsolidation_log (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,           -- 'episode', 'semantic_node', 'procedure'
    memory_id TEXT NOT NULL,
    pe_raw REAL NOT NULL,               -- Raw prediction error (Jaccard distance)
    pe_eff REAL NOT NULL,               -- Effective PE after neuromodulation
    update_mode TEXT NOT NULL,           -- 'confirmed', 'selective_edit', 'integration', 'new_episode'
    labile_duration_minutes REAL,        -- How long the lability window was open
    snapshot_before TEXT,                -- JSON: memory content before update
    snapshot_after TEXT,                 -- JSON: memory content after update
    trigger_episode_id TEXT,             -- What episode triggered reconsolidation
    agent_id TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_recon_memory ON reconsolidation_log(memory_type, memory_id);
CREATE INDEX IF NOT EXISTS idx_recon_mode ON reconsolidation_log(update_mode);
CREATE INDEX IF NOT EXISTS idx_recon_timestamp ON reconsolidation_log(timestamp);

-- ============================================================
-- V4: PREDICTION LOG
-- Tracks every prediction vs actual outcome for calibration.
-- ============================================================

CREATE TABLE IF NOT EXISTS prediction_log (
    id TEXT PRIMARY KEY,
    procedure_id TEXT NOT NULL,
    episode_id TEXT,                     -- Episode that triggered prediction
    predicted_success INTEGER,           -- What we predicted (0 or 1)
    predicted_confidence REAL,           -- Confidence in prediction
    actual_outcome INTEGER,              -- What actually happened (0 or 1)
    td_error REAL NOT NULL,              -- Temporal difference error
    surprise_score REAL,                 -- Normalized surprise (0.0-1.0)
    domain TEXT,
    agent_id TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (procedure_id) REFERENCES procedures(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pred_procedure ON prediction_log(procedure_id);
CREATE INDEX IF NOT EXISTS idx_pred_td_error ON prediction_log(td_error DESC);
CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON prediction_log(timestamp);

-- ============================================================
-- V4: SCHEMA LAYER
-- Behavioral patterns emerging from semantic cluster inductions.
-- Agenternal-inspired: 3+ semantic memories cluster -> schema.
-- ============================================================

CREATE TABLE IF NOT EXISTS schemas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    behavioral_pattern TEXT NOT NULL,    -- The abstracted pattern
    schema_type TEXT NOT NULL DEFAULT 'behavioral', -- 'behavioral', 'preference', 'domain_model'

    -- Induction metadata
    semantic_source_ids TEXT NOT NULL DEFAULT '[]',  -- JSON: IDs of semantic nodes that induced this
    episode_source_ids TEXT NOT NULL DEFAULT '[]',   -- JSON: IDs of episodes that support this
    confidence REAL NOT NULL DEFAULT 0.5,
    induction_count INTEGER DEFAULT 1,   -- How many times this schema was re-induced

    -- Applicability
    domain TEXT,
    conditions TEXT DEFAULT '[]',        -- JSON: when this schema applies
    exceptions TEXT DEFAULT '[]',        -- JSON: when this schema does NOT apply

    -- Lifecycle
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active', -- 'active', 'hypothesis', 'refuted', 'archived'
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_schemas_domain ON schemas(domain);
CREATE INDEX IF NOT EXISTS idx_schemas_type ON schemas(schema_type);
CREATE INDEX IF NOT EXISTS idx_schemas_confidence ON schemas(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_schemas_status ON schemas(status);

-- ============================================================
-- COGNITIVE PROCESSES
-- ============================================================

CREATE TABLE IF NOT EXISTS process_runs (
    id TEXT PRIMARY KEY,
    process_name TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    items_processed INTEGER DEFAULT 0,
    items_created INTEGER DEFAULT 0,
    items_modified INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_process_runs_name ON process_runs(process_name);
CREATE INDEX IF NOT EXISTS idx_process_runs_status ON process_runs(status);

-- ============================================================
-- KNOWLEDGE GRAPH
-- ============================================================

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    description TEXT,
    embedding BLOB,
    mention_count INTEGER NOT NULL DEFAULT 1,
    access_times TEXT NOT NULL DEFAULT '[]',
    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
    domain TEXT,
    source_episodes TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name, entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain);
CREATE INDEX IF NOT EXISTS idx_entities_mention_count ON entities(mention_count DESC);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    context_snippet TEXT,
    role TEXT DEFAULT 'subject',
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_source ON entity_mentions(source_type, source_id);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    evidence_episodes TEXT NOT NULL DEFAULT '[]',
    domain TEXT,
    first_observed TEXT NOT NULL DEFAULT (datetime('now')),
    last_observed TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_pair ON relationships(source_entity_id, target_entity_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type);
CREATE INDEX IF NOT EXISTS idx_relationships_strength ON relationships(strength DESC);

-- Temporal index
CREATE TABLE IF NOT EXISTS temporal_states (
    id TEXT PRIMARY KEY,
    entity_id TEXT,
    semantic_node_id TEXT,
    state_description TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    confidence REAL NOT NULL DEFAULT 0.5,
    source_episode_id TEXT,
    domain TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_temporal_entity ON temporal_states(entity_id);
CREATE INDEX IF NOT EXISTS idx_temporal_valid ON temporal_states(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_temporal_current ON temporal_states(valid_until) WHERE valid_until IS NULL;
CREATE INDEX IF NOT EXISTS idx_temporal_domain ON temporal_states(domain);

-- FTS for entities
CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    name, canonical_name, description, entity_type,
    content='entities', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
    INSERT INTO entities_fts(rowid, name, canonical_name, description, entity_type)
    VALUES (new.rowid, new.name, new.canonical_name, new.description, new.entity_type);
END;

CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
    INSERT INTO entities_fts(entities_fts, rowid, name, canonical_name, description, entity_type)
    VALUES ('delete', old.rowid, old.name, old.canonical_name, old.description, old.entity_type);
END;

CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
    INSERT INTO entities_fts(entities_fts, rowid, name, canonical_name, description, entity_type)
    VALUES ('delete', old.rowid, old.name, old.canonical_name, old.description, old.entity_type);
    INSERT INTO entities_fts(rowid, name, canonical_name, description, entity_type)
    VALUES (new.rowid, new.name, new.canonical_name, new.description, new.entity_type);
END;

-- ============================================================
-- FULL-TEXT SEARCH
-- ============================================================

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content_text, action, domain, tags,
    content='episodes', content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
    content, domain, tags,
    content='semantic_nodes', content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
    name, description, trigger_pattern, domain,
    content='procedures', content_rowid='rowid'
);

-- FTS sync triggers for episodes
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, content_text, action, domain, tags)
    VALUES (new.rowid, new.content_text, new.action, new.domain, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, content_text, action, domain, tags)
    VALUES ('delete', old.rowid, old.content_text, old.action, old.domain, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, content_text, action, domain, tags)
    VALUES ('delete', old.rowid, old.content_text, old.action, old.domain, old.tags);
    INSERT INTO episodes_fts(rowid, content_text, action, domain, tags)
    VALUES (new.rowid, new.content_text, new.action, new.domain, new.tags);
END;

-- FTS sync triggers for semantic_nodes
CREATE TRIGGER IF NOT EXISTS semantic_ai AFTER INSERT ON semantic_nodes BEGIN
    INSERT INTO semantic_fts(rowid, content, domain, tags)
    VALUES (new.rowid, new.content, new.domain, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS semantic_ad AFTER DELETE ON semantic_nodes BEGIN
    INSERT INTO semantic_fts(semantic_fts, rowid, content, domain, tags)
    VALUES ('delete', old.rowid, old.content, old.domain, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS semantic_au AFTER UPDATE ON semantic_nodes BEGIN
    INSERT INTO semantic_fts(semantic_fts, rowid, content, domain, tags)
    VALUES ('delete', old.rowid, old.content, old.domain, old.tags);
    INSERT INTO semantic_fts(rowid, content, domain, tags)
    VALUES (new.rowid, new.content, new.domain, new.tags);
END;

-- FTS sync triggers for procedures
CREATE TRIGGER IF NOT EXISTS procedures_ai AFTER INSERT ON procedures BEGIN
    INSERT INTO procedures_fts(rowid, name, description, trigger_pattern, domain)
    VALUES (new.rowid, new.name, new.description, new.trigger_pattern, new.domain);
END;

CREATE TRIGGER IF NOT EXISTS procedures_ad AFTER DELETE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, name, description, trigger_pattern, domain)
    VALUES ('delete', old.rowid, old.name, old.description, old.trigger_pattern, old.domain);
END;

CREATE TRIGGER IF NOT EXISTS procedures_au AFTER UPDATE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, name, description, trigger_pattern, domain)
    VALUES ('delete', old.rowid, old.name, old.description, old.trigger_pattern, old.domain);
    INSERT INTO procedures_fts(rowid, name, description, trigger_pattern, domain)
    VALUES (new.rowid, new.name, new.description, new.trigger_pattern, new.domain);
END;

-- ============================================================
-- SEMANTIC FACTS (myelin_memorize / myelin_facts)
-- Durable key-value facts independent of episodes.
-- ============================================================

CREATE TABLE IF NOT EXISTS semantic_facts (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    domain TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_facts_agent_key ON semantic_facts(agent_id, key);
CREATE INDEX IF NOT EXISTS idx_semantic_facts_domain ON semantic_facts(domain);
CREATE INDEX IF NOT EXISTS idx_semantic_facts_deleted ON semantic_facts(deleted_at);

-- Profile facts
CREATE TABLE IF NOT EXISTS profile_facts (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    category TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    is_static INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    last_observed TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profile_facts_agent ON profile_facts(agent_id);
CREATE INDEX IF NOT EXISTS idx_profile_facts_category ON profile_facts(agent_id, category);
CREATE INDEX IF NOT EXISTS idx_profile_facts_static ON profile_facts(agent_id, is_static);
CREATE INDEX IF NOT EXISTS idx_profile_facts_confidence ON profile_facts(agent_id, confidence DESC);

-- ============================================================
-- V3: METACOGNITION TABLES (confidence, learning goals, calibration)
-- ============================================================

CREATE TABLE IF NOT EXISTS confidence_map (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL UNIQUE,
    confidence REAL NOT NULL DEFAULT 0.0,
    episode_count INTEGER NOT NULL DEFAULT 0,
    procedure_count INTEGER NOT NULL DEFAULT 0,
    last_activity TEXT,
    trend TEXT DEFAULT 'stable',
    trend_delta REAL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS learning_goals (
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

CREATE INDEX IF NOT EXISTS idx_learning_goals_status ON learning_goals(status);
CREATE INDEX IF NOT EXISTS idx_learning_goals_domain ON learning_goals(domain);

CREATE TABLE IF NOT EXISTS calibration_log (
    id TEXT PRIMARY KEY,
    procedure_id TEXT NOT NULL,
    predicted_confidence REAL NOT NULL,
    actual_outcome INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS self_evaluations (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    top_domains TEXT NOT NULL,
    weak_domains TEXT NOT NULL,
    improving TEXT NOT NULL,
    declining TEXT NOT NULL,
    insights TEXT
);

-- ============================================================
-- V3: TRANSFER TABLES (agent profiles, transfer log, tool usage)
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_profiles (
    agent_id TEXT PRIMARY KEY,
    agent_name TEXT,
    tools TEXT NOT NULL DEFAULT '[]',
    context_format TEXT,
    model_family TEXT,
    max_context INTEGER,
    supports_images INTEGER DEFAULT 0,
    capabilities TEXT DEFAULT '{}',
    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS transfer_log (
    id TEXT PRIMARY KEY,
    procedure_id TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    transfer_confidence REAL NOT NULL,
    adapted INTEGER NOT NULL DEFAULT 0,
    adaptation_details TEXT,
    outcome TEXT DEFAULT 'pending',
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (procedure_id) REFERENCES procedures(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transfer_log_procedure ON transfer_log(procedure_id);
CREATE INDEX IF NOT EXISTS idx_transfer_log_agents ON transfer_log(source_agent, target_agent);

CREATE TABLE IF NOT EXISTS tool_usage (
    agent_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (agent_id, tool_name),
    FOREIGN KEY (agent_id) REFERENCES agent_profiles(agent_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tool_usage_agent ON tool_usage(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_count ON tool_usage(usage_count DESC);
"""


MIGRATION_COLUMNS = {
    "episodes": [
        ("importance_score", "REAL NOT NULL DEFAULT 0.5"),
        ("td_error", "REAL"),
        ("surprise_score", "REAL"),
        ("priority_score", "REAL DEFAULT 0.5"),
        ("replay_count", "INTEGER DEFAULT 0"),
        ("labile_until", "TEXT"),
        ("procedure_id", "TEXT"),
        ("is_exploration", "INTEGER NOT NULL DEFAULT 0"),
        ("intrinsic_reward", "REAL"),
        ("deleted_at", "TEXT"),
    ],
    "semantic_nodes": [
        ("labile_until", "TEXT"),
        ("prediction_error", "REAL"),
        ("last_pe_raw", "REAL"),
        ("last_update_mode", "TEXT"),
        ("deleted_at", "TEXT"),
    ],
    "procedures": [
        ("prediction_error", "REAL"),
        ("surprise_score", "REAL"),
        ("total_pe_sum", "REAL DEFAULT 0.0"),
        ("pe_count", "INTEGER DEFAULT 0"),
        ("execution_count", "INTEGER NOT NULL DEFAULT 0"),
        ("deleted_at", "TEXT"),
        ("trust_state", "TEXT NOT NULL DEFAULT 'seed'"),
        ("last_evidence_timestamp", "TEXT"),
    ],
    "learning_goals": [
        ("gap_type", "TEXT"),
        ("target_id", "TEXT"),
    ],
    "procedure_evidence": [
        ("prediction_id", "TEXT"),
    ],
}


FTS_TABLES = ("entities_fts", "episodes_fts", "semantic_fts", "procedures_fts")


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _column_names(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_existing_columns(conn) -> None:
    """Add V4 columns to existing V3 databases before indexes are created."""
    for table, columns in MIGRATION_COLUMNS.items():
        if not _table_exists(conn, table):
            continue
        existing = _column_names(conn, table)
        for name, definition in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _rebuild_fts(conn) -> None:
    """Populate external-content FTS tables for rows created before triggers existed."""
    for table in FTS_TABLES:
        if _table_exists(conn, table):
            with suppress(Exception):
                conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")


def init_db(conn) -> None:
    """Initialize the database schema.

    Creates all tables, indexes, FTS virtual tables, and triggers.
    Safe to call multiple times and upgrades V3 databases in place.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    _migrate_existing_columns(conn)
    conn.executescript(SCHEMA_SQL)
    _migrate_existing_columns(conn)
    _rebuild_fts(conn)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
