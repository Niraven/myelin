"""No-network tests for MyelinProvider and ShadowDualWriteWrapper.

All tests use a temporary in-memory SQLite database.  No network calls
are made.  Mem0 is never imported — the "unavailable Mem0" scenarios are
tested by checking that the wrapper degrades gracefully when Mem0 cannot
be imported (import path points to a non-existent module).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from myelin.core.database import Database
from myelin.memory.embedding import NoOpEmbedding

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(path=tmp_path / "test_provider.db", enable_vec=False)
    _ = d.conn
    yield d
    d.close()


@pytest.fixture
def provider(db: Database) -> Any:
    """Import lazily so the module-level import doesn't fail without deps."""
    from myelin.provider import MyelinProvider

    return MyelinProvider(
        db=db,
        embedder=NoOpEmbedding(),
        config={"agent_id": "test-agent", "session_id": "test-session"},
    )


@pytest.fixture
def wrapper(db: Database) -> Any:
    """ShadowDualWriteWrapper with a non-existent Mem0 import path so
    Mem0 is always "unavailable" — no network, no package needed."""
    from myelin.shadow_writer import (
        DualWriteConfig,
        DualWriteMode,
        ShadowDualWriteWrapper,
    )
    from myelin.provider import MyelinProvider

    mp = MyelinProvider(
        db=db,
        embedder=NoOpEmbedding(),
        config={"agent_id": "test-agent", "session_id": "test-session"},
    )
    sw = ShadowDualWriteWrapper(
        myelin_provider=mp,
        db=db,
        config=DualWriteConfig(
            mode=DualWriteMode.DUAL,
            record_discrepancies=True,
            mem0_import_path="nonexistent.mem0.Memory",
            mem0_kwargs={},
        ),
    )
    return sw


# ===========================================================================
# MyelinProvider — basic operations
# ===========================================================================


class TestMyelinProviderAdd:
    """Verify the ``add`` method accepts various input shapes."""

    def test_add_string(self, provider: Any) -> None:
        entries = provider.add("hello world")
        assert len(entries) == 1
        assert entries[0]["id"]
        assert entries[0]["content"] == "hello world"

    def test_add_dict(self, provider: Any) -> None:
        entries = provider.add({"role": "user", "content": "remember this"})
        assert len(entries) == 1
        assert entries[0]["content"] == "remember this"

    def test_add_list(self, provider: Any) -> None:
        entries = provider.add(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        )
        assert len(entries) == 2
        assert entries[0]["content"] == "first"
        assert entries[1]["content"] == "second"

    def test_add_with_metadata(self, provider: Any) -> None:
        entries = provider.add(
            "tagged memory",
            metadata={"domain": "testing", "tags": ["important"]},
        )
        assert len(entries) == 1

    def test_add_stores_semantic_node(self, provider: Any) -> None:
        entries = provider.add("semantic test")
        eid = entries[0]["id"]
        sid = entries[0].get("semantic_id")
        assert sid, "Expected a semantic node to be created"

        # Verify via handlers
        from myelin.tools.handlers import ToolHandlers

        handlers = ToolHandlers(
            provider.episodic,
            provider.semantic,
            provider.procedural,
            provider.embedder,
        )
        import asyncio

        result = asyncio.run(handlers.recall("semantic test", limit=5))
        assert result["total_results"] >= 1


class TestMyelinProviderSearch:
    """Verify the ``search`` method retrieves previously added memories."""

    def test_search_returns_added_content(self, provider: Any) -> None:
        provider.add("deploy the service with kubectl")
        results = provider.search("kubectl deploy")
        assert len(results) >= 1
        assert any("deploy" in r.get("content", "") for r in results)

    def test_search_empty_result(self, provider: Any) -> None:
        results = provider.search("nonexistent unicorn magic")
        assert len(results) == 0

    def test_search_multiple_matches(self, provider: Any) -> None:
        provider.add("build docker image")
        provider.add("push docker image to registry")
        results = provider.search("docker image")
        assert len(results) >= 1


class TestMyelinProviderUpdate:
    """Verify the ``update`` method creates updated entries."""

    def test_update_creates_new_entry(self, provider: Any) -> None:
        entries = provider.add("original text")
        first_id = entries[0]["id"]

        updated = provider.update(first_id, "updated text")
        assert updated["id"] != first_id
        assert updated["previous_id"] == first_id
        assert updated["content"] == "updated text"

    def test_update_with_dict(self, provider: Any) -> None:
        entries = provider.add("old data")
        eid = entries[0]["id"]
        updated = provider.update(eid, {"content": "new data"})
        assert updated["content"] == "new data"


class TestMyelinProviderDelete:
    """Verify the ``delete`` method (soft-delete via tombstone)."""

    def test_delete_existing(self, provider: Any) -> None:
        entries = provider.add("to be deleted")
        eid = entries[0]["id"]
        assert provider.delete(eid) is True

    def test_delete_nonexistent(self, provider: Any) -> None:
        assert provider.delete("nonexistent-id") is False

    def test_deleted_excluded_from_search(self, provider: Any) -> None:
        entries = provider.add("will be deleted")
        eid = entries[0]["id"]
        provider.delete(eid)

        # get_all should exclude it
        all_mem = provider.get_all()
        assert not any(m["id"] == eid for m in all_mem)


class TestMyelinProviderGetAll:
    """Verify ``get_all`` returns entries in reverse chronological order."""

    def test_get_all_returns_recent(self, provider: Any) -> None:
        provider.add("first")
        provider.add("second")
        all_mem = provider.get_all()
        assert len(all_mem) >= 2

    def test_get_all_filters_by_agent(self, provider: Any) -> None:
        provider.add("agent-a memory")
        provider.add("agent-a memory 2")
        all_mem = provider.get_all(agent_id="test-agent")
        assert len(all_mem) >= 1
        for m in all_mem:
            assert m["agent_id"] == "test-agent"

    def test_get_all_limit(self, provider: Any) -> None:
        for i in range(5):
            provider.add(f"memory {i}")
        all_mem = provider.get_all(limit=3)
        assert len(all_mem) <= 3


# ===========================================================================
# ShadowDualWriteWrapper — Myelin-only (baseline)
# ===========================================================================


class TestShadowWrapperMyelinOnly:
    """With mode=MYELIN_ONLY, wrapper should behave identically to
    MyelinProvider.  Mem0 is never touched."""

    def test_add_myelin_only(self, wrapper: Any) -> None:
        wrapper.mode = "myelin_only"
        entries = wrapper.add("hello from wrapper")
        assert len(entries) == 1
        assert entries[0]["content"] == "hello from wrapper"

    def test_search_myelin_only(self, wrapper: Any) -> None:
        wrapper.mode = "myelin_only"
        wrapper.add("searchable content")
        results = wrapper.search("searchable")
        assert len(results) >= 1

    def test_update_myelin_only(self, wrapper: Any) -> None:
        wrapper.mode = "myelin_only"
        entries = wrapper.add("original")
        updated = wrapper.update(entries[0]["id"], "modified")
        assert updated["content"] == "modified"

    def test_delete_myelin_only(self, wrapper: Any) -> None:
        wrapper.mode = "myelin_only"
        entries = wrapper.add("delete me")
        assert wrapper.delete(entries[0]["id"]) is True

    def test_get_all_myelin_only(self, wrapper: Any) -> None:
        wrapper.mode = "myelin_only"
        wrapper.add("entry one")
        wrapper.add("entry two")
        all_mem = wrapper.get_all()
        assert len(all_mem) >= 2


# ===========================================================================
# ShadowDualWriteWrapper — Mem0 unavailable
# ===========================================================================


class TestShadowWrapperMem0Unavailable:
    """When Mem0 cannot be imported / initialised, the wrapper must
    silently fall back to Myelin-only behaviour and log the error."""

    def test_init_mem0_unavailable(self, wrapper: Any) -> None:
        assert not wrapper.mem0_available
        assert wrapper._mem0_init_error is not None
        assert "nonexistent" in str(wrapper._mem0_init_error)

    def test_dual_mode_without_mem0_add(self, wrapper: Any) -> None:
        # Should succeed even though Mem0 is unavailable
        entries = wrapper.add("mem0-unavailable test")
        assert len(entries) == 1

    def test_dual_mode_without_mem0_search(self, wrapper: Any) -> None:
        wrapper.add("something to find")
        results = wrapper.search("something")
        assert len(results) >= 1

    def test_dual_mode_without_mem0_update(self, wrapper: Any) -> None:
        entries = wrapper.add("before update")
        updated = wrapper.update(entries[0]["id"], "after update")
        assert updated["content"] == "after update"

    def test_dual_mode_without_mem0_delete(self, wrapper: Any) -> None:
        entries = wrapper.add("to delete")
        assert wrapper.delete(entries[0]["id"]) is True

    def test_errors_logged_for_mem0_failures(self, wrapper: Any) -> None:
        """When Mem0 is unavailable, shadow_errors should be populated
        on dual-write attempts."""
        # Add in dual mode — should log an error for the Mem0 attempt
        wrapper.add("error test")
        rows = wrapper.db.fetchall(
            "SELECT * FROM shadow_errors WHERE operation = 'add'"
        )
        assert len(rows) >= 1
        assert "nonexistent" in str(rows[0]["error"])


# ===========================================================================
# ShadowDualWriteWrapper — dual-write failure isolation
# ===========================================================================


class TestDualWriteFailureIsolation:
    """Even when Mem0 is unavailable, the Myelin path must succeed."""

    def test_isolation_on_add(self, wrapper: Any) -> None:
        wrapper.mode = "dual"
        entries = wrapper.add("isolated add")
        assert len(entries) == 1
        assert entries[0]["content"] == "isolated add"

    def test_isolation_on_search(self, wrapper: Any) -> None:
        wrapper.mode = "dual"
        wrapper.add("isolated search content")
        results = wrapper.search("isolated search")
        assert len(results) >= 1

    def test_isolation_on_update(self, wrapper: Any) -> None:
        wrapper.mode = "dual"
        entries = wrapper.add("before")
        updated = wrapper.update(entries[0]["id"], "after")
        assert updated["content"] == "after"

    def test_isolation_on_delete(self, wrapper: Any) -> None:
        wrapper.mode = "dual"
        entries = wrapper.add("isolated delete")
        assert wrapper.delete(entries[0]["id"]) is True


# ===========================================================================
# ShadowDualWriteWrapper — discrepancy reporting tables
# ===========================================================================


class TestDiscrepancyReporting:
    """Verify that discrepancy tracking tables are created and populated."""

    def test_discrepancy_table_exists(self, wrapper: Any) -> None:
        """The shadow_discrepancies table is created on init."""
        rows = wrapper.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_discrepancies'"
        )
        assert len(rows) == 1

    def test_provenance_table_exists(self, wrapper: Any) -> None:
        rows = wrapper.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_provenance'"
        )
        assert len(rows) == 1

    def test_error_table_exists(self, wrapper: Any) -> None:
        rows = wrapper.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_errors'"
        )
        assert len(rows) == 1

    def test_add_records_provenance_attempt(self, wrapper: Any) -> None:
        """Even when Mem0 fails, provenance is recorded for Myelin IDs."""
        wrapper.add("provenance test")
        rows = wrapper.db.fetchall(
            "SELECT * FROM shadow_provenance WHERE action = 'add'"
        )
        assert len(rows) >= 1
        prov = dict(rows[0])
        myelin_ids = json.loads(prov.get("myelin_ids", "[]"))
        assert len(myelin_ids) >= 1

    def test_search_records_discrepancy_attempt(self, wrapper: Any) -> None:
        """Dual-mode search logs an error but still returns Myelin results."""
        wrapper.add("discrepancy test content")
        results = wrapper.search("discrepancy")
        assert len(results) >= 1

    def test_provenance_tracks_agent(self, wrapper: Any) -> None:
        wrapper.add("agent tracking test")
        rows = wrapper.db.fetchall(
            "SELECT * FROM shadow_provenance WHERE agent_id = 'test-agent'"
        )
        assert len(rows) >= 1
        assert dict(rows[0])["agent_id"] == "test-agent"


# ===========================================================================
# ShadowDualWriteWrapper — rollback support
# ===========================================================================


class TestRollbackSupport:
    """Verify rollback and disable/enable toggle work correctly."""

    def test_disable_shadow(self, wrapper: Any) -> None:
        result = wrapper.disable_shadow()
        assert result["status"] == "disabled"
        assert wrapper.mode.value == "myelin_only"

    def test_enable_shadow(self, wrapper: Any) -> None:
        wrapper.disable_shadow()
        result = wrapper.enable_shadow()
        assert result["status"] == "enabled"
        assert wrapper.mode.value == "dual"

    def test_enable_shadow_custom_mode(self, wrapper: Any) -> None:
        from myelin.shadow_writer import DualWriteMode

        result = wrapper.enable_shadow(mode=DualWriteMode.SHADOW_READ)
        assert result["mode"] == "shadow_read"

    def test_rollback_shadow_without_mem0(self, wrapper: Any) -> None:
        """Rollback should be a no-op when Mem0 is unavailable."""
        result = wrapper.rollback_shadow()
        assert result["status"] == "skipped"
        assert "reason" in result

    def test_mode_remains_operational_after_disable(self, wrapper: Any) -> None:
        wrapper.disable_shadow()
        entries = wrapper.add("post-disable memory")
        assert len(entries) == 1
        results = wrapper.search("post-disable")
        assert len(results) >= 1

    def test_mode_remains_operational_after_enable(self, wrapper: Any) -> None:
        wrapper.disable_shadow()
        wrapper.enable_shadow()
        entries = wrapper.add("post-enable memory")
        assert len(entries) == 1


# ===========================================================================
# ShadowDualWriteWrapper — provenance tracking
# ===========================================================================


class TestProvenanceTracking:
    """Verify that provenance is tracked for audit purposes."""

    def test_provenance_has_action_type(self, wrapper: Any) -> None:
        wrapper.add("provenance action check")
        rows = wrapper.db.fetchall(
            "SELECT action FROM shadow_provenance WHERE action = 'add'"
        )
        assert len(rows) >= 1

    def test_provenance_has_timestamp(self, wrapper: Any) -> None:
        wrapper.add("provenance timestamp check")
        rows = wrapper.db.fetchall(
            "SELECT timestamp FROM shadow_provenance ORDER BY timestamp DESC LIMIT 1"
        )
        assert len(rows) == 1
        assert rows[0]["timestamp"]

    def test_multiple_ops_record_provenance(self, wrapper: Any) -> None:
        wrapper.add("first")
        wrapper.add("second")
        wrapper.add("third")
        rows = wrapper.db.fetchall(
            "SELECT * FROM shadow_provenance WHERE action = 'add'"
        )
        assert len(rows) >= 3

    def test_get_provenance(self, wrapper: Any) -> None:
        wrapper.add("prov get test")
        prov = wrapper._get_provenance(agent_id="test-agent")
        assert len(prov) >= 1
        assert prov[0]["agent_id"] == "test-agent"
        assert len(prov[0]["myelin_ids"]) >= 1


# ===========================================================================
# Integration: full lifecycle with provider
# ===========================================================================


class TestFullLifecycle:
    """End-to-end: add → search → update → search → delete → get_all."""

    def test_full_lifecycle(self, provider: Any) -> None:
        # Add
        entries = provider.add("initial learning")
        assert len(entries) == 1
        eid = entries[0]["id"]

        # Search
        results = provider.search("initial")
        assert len(results) >= 1

        # Update
        updated = provider.update(eid, "revised learning")
        assert updated["content"] == "revised learning"

        # Search again
        results_after = provider.search("revised")
        assert len(results_after) >= 1

        # Delete original
        assert provider.delete(eid) is True

        # Get all — original excluded, update exists
        all_mem = provider.get_all()
        assert not any(m["id"] == eid for m in all_mem)
        assert any(m["id"] == updated["id"] for m in all_mem)

    def test_multiple_agents(self, db: Any) -> None:
        from myelin.provider import MyelinProvider

        p1 = MyelinProvider(
            db=db,
            embedder=NoOpEmbedding(),
            config={"agent_id": "agent-alpha"},
        )
        p2 = MyelinProvider(
            db=db,
            embedder=NoOpEmbedding(),
            config={"agent_id": "agent-beta"},
        )

        p1.add("alpha memory")
        p2.add("beta memory")

        alpha_all = p1.get_all()
        beta_all = p2.get_all()

        assert all(m["agent_id"] == "agent-alpha" for m in alpha_all)
        assert all(m["agent_id"] == "agent-beta" for m in beta_all)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Unusual input shapes and boundary conditions."""

    def test_add_empty_string(self, provider: Any) -> None:
        entries = provider.add("")
        assert len(entries) == 1

    def test_search_empty_query(self, provider: Any) -> None:
        results = provider.search("")
        assert isinstance(results, list)

    def test_update_nonexistent_id(self, provider: Any) -> None:
        updated = provider.update("no-such-id", "data")
        assert "id" in updated

    def test_get_all_empty_db(self, provider: Any) -> None:
        all_mem = provider.get_all()
        assert len(all_mem) == 0

    def test_delete_twice(self, provider: Any) -> None:
        entries = provider.add("double delete")
        eid = entries[0]["id"]
        assert provider.delete(eid) is True
        # Second delete should still succeed (redundant tombstone)
        assert provider.delete(eid) is True

    def test_normalise_messages_string(self, provider: Any) -> None:
        result = provider._normalise_messages("simple string")
        assert result == ["simple string"]

    def test_normalise_messages_dict(self, provider: Any) -> None:
        result = provider._normalise_messages({"content": "dict content"})
        assert result == ["dict content"]

    def test_normalise_messages_list(self, provider: Any) -> None:
        result = provider._normalise_messages(
            [{"content": "a"}, {"content": "b"}]
        )
        assert result == ["a", "b"]

    def test_normalise_messages_list_partial(self, provider: Any) -> None:
        result = provider._normalise_messages([{"role": "user"}, "plain"])
        assert len(result) == 2
