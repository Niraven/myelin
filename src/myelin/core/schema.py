"""Myelin V2 SQLite schema. Grounded in ACT-R, SOAR, and Generative Agents."""

SCHEMA_VERSION = 3

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
-- Raw observations of agent behavior with full context.
-- ============================================================

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),

    -- What happened
    action TEXT NOT NULL,
    action_type TEXT NOT NULL,          -- 'tool_call', 'response', 'error', 'user_input'
    input_context TEXT,                 -- JSON: what triggered this action
    output_result TEXT,                 -- JSON: what the action produced
    success INTEGER NOT NULL DEFAULT 1, -- 0 or 1

    -- Semantic search
    content_text TEXT NOT NULL,         -- Flattened text for FTS
    embedding BLOB,                    -- Vector embedding (768d float32)

    -- ACT-R activation tracking
    access_count INTEGER NOT NULL DEFAULT 1,
    access_times TEXT NOT NULL DEFAULT '[]', -- JSON array of unix timestamps
    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),

    -- Consolidation state
    consolidated INTEGER NOT NULL DEFAULT 0,
    cluster_id TEXT,

    -- Metadata
    tags TEXT DEFAULT '[]',            -- JSON array
    domain TEXT,                       -- Inferred domain (e.g. 'deployment', 'testing')
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(domain);
CREATE INDEX IF NOT EXISTS idx_episodes_cluster ON episodes(cluster_id);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated);

-- ============================================================
-- SEMANTIC MEMORY
-- Facts, reflections, and higher-order knowledge.
-- ============================================================

CREATE TABLE IF NOT EXISTS semantic_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,            -- 'fact', 'reflection', 'meta_reflection', 'preference'
    content TEXT NOT NULL,
    embedding BLOB,

    -- Source tracking
    source_type TEXT NOT NULL,          -- 'observation', 'reflection', 'teaching', 'transfer'
    source_ids TEXT NOT NULL DEFAULT '[]', -- JSON: episode/node IDs this was derived from

    -- ACT-R activation
    access_count INTEGER NOT NULL DEFAULT 1,
    access_times TEXT NOT NULL DEFAULT '[]',
    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),

    -- Confidence and validity
    confidence REAL NOT NULL DEFAULT 0.5,
    valid_from TEXT,
    valid_until TEXT,                   -- NULL = still valid
    superseded_by TEXT,                 -- ID of replacement node

    -- Organization
    domain TEXT,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_semantic_domain ON semantic_nodes(domain);
CREATE INDEX IF NOT EXISTS idx_semantic_confidence ON semantic_nodes(confidence);

-- ============================================================
-- PROCEDURAL MEMORY
-- Learned workflows with branching, variants, and composition.
-- ============================================================

CREATE TABLE IF NOT EXISTS procedures (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,

    -- Trigger matching
    trigger_pattern TEXT NOT NULL,      -- Natural language trigger description
    trigger_embedding BLOB,            -- Vector for semantic matching

    -- Procedure body (V2: supports branching)
    steps TEXT NOT NULL,                -- JSON: ordered steps with CORE/VARIANT/OPTIONAL flags
    preconditions TEXT DEFAULT '[]',    -- JSON: what must be true before execution
    postconditions TEXT DEFAULT '[]',   -- JSON: what is true after execution

    -- Bayesian confidence (V2)
    confidence REAL NOT NULL DEFAULT 0.5,
    predicted_success_rate REAL,        -- What we predict
    actual_success_rate REAL,           -- What actually happens
    calibration_offset REAL DEFAULT 0.0, -- predicted - actual (for correction)
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    modify_count INTEGER NOT NULL DEFAULT 0,

    -- ACT-R activation
    activation_score REAL NOT NULL DEFAULT 0.0,
    access_times TEXT NOT NULL DEFAULT '[]',
    last_executed TEXT,

    -- Provenance
    source_agent TEXT NOT NULL,
    source_episodes TEXT DEFAULT '[]',  -- JSON: episode IDs that generated this
    promotion_method TEXT DEFAULT 'auto', -- 'auto', 'taught', 'transferred', 'composed'

    -- Composition (V2: SOAR chunking)
    is_composite INTEGER NOT NULL DEFAULT 0,
    component_procedures TEXT DEFAULT '[]', -- JSON: IDs of sub-procedures
    parent_procedures TEXT DEFAULT '[]',    -- JSON: IDs of meta-procedures containing this

    -- Transfer tracking
    transferred_to TEXT DEFAULT '[]',   -- JSON: agent IDs
    transfer_success_rate REAL DEFAULT 0.0,

    -- Lifecycle
    status TEXT NOT NULL DEFAULT 'draft', -- 'draft', 'active', 'reflexive', 'archived'
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
-- METACOGNITION
-- Confidence maps, learning goals, calibration, self-evaluation.
-- ============================================================

CREATE TABLE IF NOT EXISTS confidence_map (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL UNIQUE,
    confidence REAL NOT NULL DEFAULT 0.0,
    episode_count INTEGER NOT NULL DEFAULT 0,
    procedure_count INTEGER NOT NULL DEFAULT 0,
    last_activity TEXT,
    trend TEXT DEFAULT 'stable',       -- 'improving', 'stable', 'declining'
    trend_delta REAL DEFAULT 0.0,      -- Confidence change over 30 days
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- SOAR-inspired impasse detection and learning goals
CREATE TABLE IF NOT EXISTS learning_goals (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    goal TEXT NOT NULL,                 -- What we need to learn
    strategy TEXT,                      -- How we plan to learn it
    priority REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active', -- 'active', 'achieved', 'abandoned'
    episodes_needed INTEGER DEFAULT 3,
    episodes_collected INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_learning_goals_status ON learning_goals(status);
CREATE INDEX IF NOT EXISTS idx_learning_goals_domain ON learning_goals(domain);

-- Calibration tracking: predicted vs actual
CREATE TABLE IF NOT EXISTS calibration_log (
    id TEXT PRIMARY KEY,
    procedure_id TEXT NOT NULL,
    predicted_confidence REAL NOT NULL,
    actual_outcome INTEGER NOT NULL,   -- 0 or 1
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (procedure_id) REFERENCES procedures(id) ON DELETE CASCADE
);

-- Self-evaluation snapshots
CREATE TABLE IF NOT EXISTS self_evaluations (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    top_domains TEXT NOT NULL,          -- JSON: highest confidence
    weak_domains TEXT NOT NULL,         -- JSON: lowest confidence with activity
    improving TEXT NOT NULL,            -- JSON: positive trend
    declining TEXT NOT NULL,            -- JSON: negative trend
    insights TEXT                       -- Generated self-reflection text
);

-- ============================================================
-- TRANSFER
-- Agent profiles and cross-agent procedure adaptation.
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_profiles (
    agent_id TEXT PRIMARY KEY,
    agent_name TEXT,
    tools TEXT NOT NULL DEFAULT '[]',   -- JSON: available tool names
    context_format TEXT,                -- 'mcp_stdio', 'mcp_sse', 'custom'
    model_family TEXT,                  -- 'claude', 'gpt', 'llama', etc.
    max_context INTEGER,
    supports_images INTEGER DEFAULT 0,
    capabilities TEXT DEFAULT '{}',     -- JSON: additional capability flags
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
    adaptation_details TEXT,            -- JSON: what was changed
    outcome TEXT DEFAULT 'pending',     -- 'pending', 'success', 'failure', 'modified'
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (procedure_id) REFERENCES procedures(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transfer_log_procedure ON transfer_log(procedure_id);
CREATE INDEX IF NOT EXISTS idx_transfer_log_agents ON transfer_log(source_agent, target_agent);

-- ============================================================
-- COGNITIVE PROCESSES
-- Background process state and scheduling.
-- ============================================================

CREATE TABLE IF NOT EXISTS process_runs (
    id TEXT PRIMARY KEY,
    process_name TEXT NOT NULL,         -- 'consolidator', 'reflector', 'promoter', 'composer', 'decayer', 'challenger'
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    items_processed INTEGER DEFAULT 0,
    items_created INTEGER DEFAULT 0,
    items_modified INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running', -- 'running', 'completed', 'failed'
    error TEXT,
    details TEXT                        -- JSON: process-specific output
);

CREATE INDEX IF NOT EXISTS idx_process_runs_name ON process_runs(process_name);
CREATE INDEX IF NOT EXISTS idx_process_runs_status ON process_runs(status);

-- ============================================================
-- KNOWLEDGE GRAPH
-- Entity extraction, typed relationships, temporal index.
-- Stolen from: mem0 (entity linking), Supermemory (ontology-aware graph).
-- Made better: entities feed procedure discovery, relationships are
-- learned from behavior not just similarity, temporal reasoning
-- resolves "current state" vs "historical" queries.
-- ============================================================

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,           -- 'tool', 'service', 'concept', 'file', 'person', 'config', 'error'
    canonical_name TEXT NOT NULL,        -- Deduplicated canonical form
    description TEXT,
    embedding BLOB,

    -- Frequency and activation
    mention_count INTEGER NOT NULL DEFAULT 1,
    access_times TEXT NOT NULL DEFAULT '[]',
    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),

    -- Provenance
    domain TEXT,
    source_episodes TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name, entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain);
CREATE INDEX IF NOT EXISTS idx_entities_mention_count ON entities(mention_count DESC);

-- Links entities to the episodes/nodes where they appear
CREATE TABLE IF NOT EXISTS entity_mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    source_type TEXT NOT NULL,           -- 'episode', 'semantic_node', 'procedure'
    source_id TEXT NOT NULL,
    context_snippet TEXT,                -- Surrounding text where entity was found
    role TEXT DEFAULT 'subject',         -- 'subject', 'object', 'tool', 'target'
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_source ON entity_mentions(source_type, source_id);

-- Typed edges between entities (the knowledge graph)
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,         -- 'uses', 'requires', 'produces', 'causes', 'contradicts', 'supersedes', 'related_to', 'part_of', 'triggers'
    strength REAL NOT NULL DEFAULT 1.0,  -- Co-occurrence / confidence weight
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

-- Temporal index: tracks state changes over time for entities/facts
CREATE TABLE IF NOT EXISTS temporal_states (
    id TEXT PRIMARY KEY,
    entity_id TEXT,                      -- NULL for general facts
    semantic_node_id TEXT,               -- The semantic node this state comes from
    state_description TEXT NOT NULL,     -- What was true at this time
    valid_from TEXT NOT NULL,
    valid_until TEXT,                    -- NULL = currently valid
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
    name,
    canonical_name,
    description,
    entity_type,
    content='entities',
    content_rowid='rowid'
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
-- FULL-TEXT SEARCH (FTS5)
-- ============================================================

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content_text,
    action,
    domain,
    tags,
    content='episodes',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
    content,
    domain,
    tags,
    content='semantic_nodes',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
    name,
    description,
    trigger_pattern,
    domain,
    content='procedures',
    content_rowid='rowid'
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
"""


def init_db(conn) -> None:
    """Initialize the database with the V2 schema."""
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )
    conn.commit()
