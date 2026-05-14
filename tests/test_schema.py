"""Test SQLite schema initialization."""

from myelin.core.database import Database
from myelin.core.schema import SCHEMA_VERSION


def test_schema_creates_all_tables(tmp_db):
    tables = tmp_db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    table_names = {t["name"] for t in tables}

    expected = {
        "schema_meta",
        "episodes",
        "semantic_nodes",
        "procedures",
        "confidence_map",
        "learning_goals",
        "calibration_log",
        "self_evaluations",
        "agent_profiles",
        "transfer_log",
        "process_runs",
    }
    assert expected.issubset(table_names), f"Missing tables: {expected - table_names}"


def test_schema_version(tmp_db):
    row = tmp_db.fetchone("SELECT value FROM schema_meta WHERE key = 'version'")
    assert row is not None
    assert row["value"] == str(SCHEMA_VERSION)


def test_fts_tables_exist(tmp_db):
    tables = tmp_db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts%'"
    )
    fts_names = {t["name"] for t in tables}
    assert "episodes_fts" in fts_names
    assert "semantic_fts" in fts_names
    assert "procedures_fts" in fts_names


def test_wal_mode(tmp_db):
    row = tmp_db.fetchone("PRAGMA journal_mode")
    assert row is not None
    assert row["journal_mode"] == "wal"


def test_foreign_keys_enabled(tmp_db):
    row = tmp_db.fetchone("PRAGMA foreign_keys")
    assert row is not None
    assert row["foreign_keys"] == 1
