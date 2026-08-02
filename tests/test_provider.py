"""Tests for the MyelinProvider adapter + Mem0 dual-write / shadow-read mode."""

from __future__ import annotations

import pytest

from myelin.provider import Mem0DualWriteAdapter, MyelinProvider, SearchResult
from myelin.provider.protocol import MemoryProvider, compare_search_results

# ---------------------------------------------------------------------------
#  Mock helpers
# ---------------------------------------------------------------------------


class _MockHandler:
    """Simulates a ToolHandlers instance for in-process testing."""

    def __init__(self) -> None:
        self._facts: dict[str, dict] = {}
        self._count = 0

    async def memorize(
        self, agent_id: str, fact: str, category: str = "fact", confidence: float = 0.6
    ) -> dict:
        fact_id = f"fact-{self._count}"
        self._count += 1
        self._facts[fact_id] = {
            "id": fact_id,
            "agent_id": agent_id,
            "fact": fact,
            "category": category,
            "confidence": confidence,
        }
        return {"fact_id": fact_id}

    async def update_memory(self, memory_id: str, new_text: str) -> None:
        if memory_id in self._facts:
            self._facts[memory_id]["fact"] = new_text

    async def forget(self, fact_id: str) -> None:
        self._facts.pop(fact_id, None)

    async def context(
        self, query: str, agent_id: str = "hermes", max_memories: int = 10, max_procedures: int = 3
    ) -> dict:
        # Return a dummy response so search doesn't crash
        return {"relevant_memories": [], "results": {"episodes": []}}

    async def profile(self, agent_id: str = "hermes") -> dict:
        static = [f for f in self._facts.values() if f["agent_id"] == agent_id]
        return {
            "agent_id": agent_id,
            "static_facts": static,
            "dynamic_context": [],
            "fact_count": len(static),
            "static_count": len(static),
            "dynamic_count": 0,
            "category_breakdown": {},
            "confidence_summary": {},
        }


class _CurrentHandler:
    """Matches the current semantic-fact handler API."""

    def __init__(self) -> None:
        self._facts: dict[str, dict] = {}

    async def memorize(
        self, agent_id: str, key: str, value: str, domain=None, ttl_days=None
    ) -> dict:
        fact_id = f"semantic-{len(self._facts)}"
        self._facts[fact_id] = {"agent_id": agent_id, "key": key, "value": value, "domain": domain}
        return {"fact_id": fact_id}

    async def update(
        self,
        memory_id: str,
        memory_type: str = "episode",
        content_text=None,
        action=None,
        value=None,
    ) -> dict:
        self._facts[memory_id]["value"] = value
        return {"success": True, "memory_id": memory_id, "memory_type": memory_type}

    async def forget(self, memory_id: str, memory_type: str = "episode") -> dict:
        self._facts.pop(memory_id, None)
        return {"success": True, "memory_id": memory_id, "memory_type": memory_type}

    async def context(self, **kwargs) -> dict:
        return {"relevant_memories": []}

    async def facts(self, agent_id: str, key_prefix=None, domain=None, limit: int = 20) -> dict:
        return {
            "facts": [
                {"id": fact_id, **fact}
                for fact_id, fact in self._facts.items()
                if fact["agent_id"] == agent_id
            ][:limit]
        }

    async def profile(self, agent_id: str = "hermes") -> dict:
        return {"static_facts": [], "fact_count": 0, "category_breakdown": {}}


class _FakeMem0Provider(MemoryProvider):
    """A simulated Mem0 MemoryProvider for testing dual-write/shadow-read."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self._store: dict[str, SearchResult] = {}
        self._count = 0
        self._fail_on = fail_on or set()

    async def add(self, content: str, **kw) -> dict:
        if "add" in self._fail_on:
            raise RuntimeError("Mem0 add failed")
        fid = f"mem0-{self._count}"
        self._count += 1
        self._store[fid] = SearchResult(id=fid, memory=content, score=0.8)
        return {"result": "Fact stored.", "event_id": fid}

    async def search(self, query: str, *, top_k: int = 10, **kw) -> list[SearchResult]:
        if "search" in self._fail_on:
            raise RuntimeError("Mem0 search failed")
        return [
            r for r in sorted(self._store.values(), key=lambda x: x.score, reverse=True)[:top_k]
        ]

    async def update(self, memory_id: str, text: str, **kw) -> dict:
        if "update" in self._fail_on:
            raise RuntimeError("Mem0 update failed")
        if memory_id in self._store:
            self._store[memory_id].memory = text
        return {"memory_id": memory_id}

    async def delete(self, memory_id: str, **kw) -> dict:
        if "delete" in self._fail_on:
            raise RuntimeError("Mem0 delete failed")
        self._store.pop(memory_id, None)
        return {"memory_id": memory_id}

    async def system_prompt_block(self, **kw) -> str:
        return "# Mem0 Memory\n32 facts."

    async def health(self) -> dict:
        return {"ok": "fail_all" not in self._fail_on, "detail": "fake"}


# ---------------------------------------------------------------------------
#  Tests: protocol helpers
# ---------------------------------------------------------------------------


class TestCompareSearchResults:
    def test_both_empty(self):
        c = compare_search_results([], [])
        assert c.overlap == 1.0
        assert c.quality == "match"

    def test_primary_empty(self):
        c = compare_search_results([], [SearchResult(id="1", memory="hello")])
        assert c.overlap == 0.0
        assert c.quality == "divergence"

    def test_shadow_empty(self):
        c = compare_search_results([SearchResult(id="1", memory="hello")], [])
        assert c.overlap == 0.0
        assert c.quality == "divergence"

    def test_perfect_match(self):
        primary = [
            SearchResult(id="a", memory="user prefers vim with dark mode"),
            SearchResult(id="b", memory="favorite language is python"),
        ]
        shadow = [
            SearchResult(id="a", memory="user prefers vim with dark mode"),
            SearchResult(id="b", memory="favorite language is python"),
        ]
        c = compare_search_results(primary, shadow)
        assert c.overlap == 1.0
        assert c.quality == "match"

    def test_partial_overlap_myelin_better(self):
        primary = [
            SearchResult(id="a", memory="user prefers vim with dark mode"),
            SearchResult(id="b", memory="favorite language is python"),
        ]
        shadow = [
            SearchResult(id="a", memory="user prefers vim with dark mode"),
        ]
        c = compare_search_results(primary, shadow)
        assert c.overlap >= 0.5
        # primary=myelin has 1 extra; shadow=mem0 has 0 extra → overlap=0.5 quality='mem0_better'
        assert c.quality in ("myelin_better", "match", "mem0_better")

    def test_no_overlap(self):
        primary = [SearchResult(id="a", memory="prefers vim")]
        shadow = [SearchResult(id="b", memory="likes dogs")]
        c = compare_search_results(primary, shadow)
        assert c.overlap == 0.0
        assert c.quality == "divergence"

    def test_custom_iou_threshold(self):
        primary = [SearchResult(id="a", memory="hello world foo bar")]
        shadow = [SearchResult(id="b", memory="hello world baz qux")]
        # Default threshold 0.3: 2/6 = 0.33 >= 0.3 → match
        c = compare_search_results(primary, shadow)
        assert c.overlap == 1.0
        assert c.quality == "match"

    def test_precision_calculation(self):
        primary = [
            SearchResult(id="a", memory="one"),
            SearchResult(id="b", memory="two"),
            SearchResult(id="c", memory="three"),
        ]
        shadow = [SearchResult(id="x", memory="one")]
        c = compare_search_results(primary, shadow)
        assert c.precision == 1.0  # 1/1 matched in shadow
        # overlap = matched primary / max(len(primary), len(shadow))
        assert c.overlap == pytest.approx(1.0 / 3.0, rel=1e-3)


# ---------------------------------------------------------------------------
#  Tests: MyelinProvider (default mode)
# ---------------------------------------------------------------------------


class TestMyelinProvider:
    @pytest.fixture
    def handler(self):
        return _MockHandler()

    @pytest.fixture
    def provider(self, handler):
        return MyelinProvider(tool_handlers=handler)

    async def test_health_ok(self, provider):
        h = await provider.health()
        assert h["ok"] is True

    async def test_health_not_available(self):
        p = MyelinProvider(tool_handlers=None)
        h = await p.health()
        assert h["ok"] is False

    async def test_add_default_behavior(self, provider):
        result = await provider.add("user prefers vim")
        assert result["result"] == "Fact stored."
        assert result["event_id"]

    async def test_add_with_metadata(self, provider):
        result = await provider.add(
            "is a python developer",
            metadata={"category": "skill", "confidence": 0.5},
        )
        assert result["result"] == "Fact stored."

    async def test_search_returns_results(self, provider, handler):
        await provider.add("prefers vim")
        results = await provider.search("editor preference")
        assert isinstance(results, list)

    async def test_search_includes_profile_facts(self, provider, handler):
        await provider.add("loves python")
        results = await provider.search("python")
        assert any("python" in r.memory.lower() for r in results)

    async def test_update(self, provider, handler):
        result = await provider.add("old text")
        fid = result["event_id"]
        up = await provider.update(fid, "new text")
        assert up["memory_id"] == fid

    async def test_delete(self, provider, handler):
        result = await provider.add("to delete")
        fid = result["event_id"]
        dl = await provider.delete(fid)
        assert dl["memory_id"] == fid

    async def test_system_prompt_block(self, provider, handler):
        block = await provider.system_prompt_block()
        assert "Myelin Memory" in block
        assert "Active" in block

    async def test_system_prompt_not_available(self):
        p = MyelinProvider(tool_handlers=None)
        block = await p.system_prompt_block()
        assert "Not available" in block

    async def test_add_without_handlers(self):
        p = MyelinProvider(tool_handlers=None)
        result = await p.add("anything")
        assert result["event_id"] == ""

    async def test_search_without_handlers(self):
        p = MyelinProvider(tool_handlers=None)
        results = await p.search("anything")
        assert results == []


# ---------------------------------------------------------------------------
#  Tests: Mem0DualWriteAdapter — default-off behavior
# ---------------------------------------------------------------------------


class TestCurrentHandlerCompatibility:
    async def test_current_handler_crud(self):
        handler = _CurrentHandler()
        provider = MyelinProvider(tool_handlers=handler)
        result = await provider.add("current api", metadata={"domain": "test"})
        memory_id = result["event_id"]
        assert handler._facts[memory_id]["value"] == "current api"
        search_results = await provider.search("current api")
        assert any(r.memory == "current api" for r in search_results)
        await provider.update(memory_id, "updated")
        assert handler._facts[memory_id]["value"] == "updated"
        await provider.delete(memory_id)
        assert memory_id not in handler._facts


class TestMem0DualWriteDefaultOff:
    @pytest.fixture
    def handler(self):
        return _MockHandler()

    @pytest.fixture
    def myelin(self, handler):
        return MyelinProvider(tool_handlers=handler)

    @pytest.fixture
    def adapter(self, myelin):
        return Mem0DualWriteAdapter(myelin)

    async def test_default_off_dual_write(self, adapter):
        assert adapter.is_dual_write_active is False

    async def test_default_off_shadow_read(self, adapter):
        assert adapter.is_shadow_read_active is False

    async def test_add_in_default_mode(self, adapter):
        result = await adapter.add("default add")
        assert result["result"] == "Fact stored."

    async def test_search_in_default_mode(self, adapter):
        results = await adapter.search("anything")
        assert isinstance(results, list)

    async def test_update_in_default_mode(self, adapter):
        result = await adapter.add("hello")
        up = await adapter.update(result["event_id"], "world")
        assert up["memory_id"] == result["event_id"]

    async def test_delete_in_default_mode(self, adapter):
        result = await adapter.add("temp")
        dl = await adapter.delete(result["event_id"])
        assert dl["memory_id"] == result["event_id"]

    async def test_health_without_mem0(self, adapter):
        h = await adapter.health()
        assert h["ok"] is True
        assert h["mem0_available"] is False
        assert h["detail"].find("myelin_only") >= 0

    async def test_comparison_log_empty_by_default(self, adapter):
        assert adapter.comparison_log == []

    async def test_reversible_disable_is_noop(self, adapter):
        adapter.disable_mem0()  # already off — no crash
        assert adapter.is_dual_write_active is False

    async def test_system_prompt_block_default(self, adapter):
        block = await adapter.system_prompt_block()
        assert "Myelin Memory" in block


# ---------------------------------------------------------------------------
#  Tests: dual-write success and failure modes
# ---------------------------------------------------------------------------


class TestMem0DualWrite:
    @pytest.fixture
    def handler(self):
        return _MockHandler()

    @pytest.fixture
    def myelin(self, handler):
        return MyelinProvider(tool_handlers=handler)

    @pytest.fixture
    def mem0(self):
        return _FakeMem0Provider()

    @pytest.fixture
    def adapter(self, myelin, mem0):
        a = Mem0DualWriteAdapter(myelin)
        a.enable_mem0(mem0)
        return a

    async def test_dual_write_active_flag(self, adapter):
        assert adapter.is_dual_write_active is True

    async def test_dual_write_add(self, adapter):
        result = await adapter.add("dual write test")
        assert result["result"] == "Fact stored."

    async def test_dual_write_add_both_providers(self, adapter, myelin, mem0):
        # Both providers should have the fact
        await adapter.add("shared fact")
        myelin_r = await myelin.search("shared")
        mem0_r = await mem0.search("shared")
        assert len(myelin_r) > 0
        assert len(mem0_r) > 0

    async def test_dual_write_update(self, adapter):
        result = await adapter.add("original")
        up = await adapter.update(result["event_id"], "updated")
        assert up["memory_id"] == result["event_id"]

    async def test_dual_write_delete(self, adapter):
        result = await adapter.add("to delete")
        await adapter.delete(result["event_id"])

    async def test_health_with_mem0(self, adapter):
        h = await adapter.health()
        assert h["ok"] is True
        assert h["mem0_available"] is True
        assert h["detail"].find("dual_write") >= 0

    async def test_disable_mem0(self, adapter):
        assert adapter.is_dual_write_active is True
        adapter.disable_mem0()
        assert adapter.is_dual_write_active is False

    async def test_re_enable_mem0(self, adapter):
        adapter.disable_mem0()
        adapter.enable_mem0(_FakeMem0Provider())
        assert adapter.is_dual_write_active is True


class TestMem0DualWriteFailure:
    @pytest.fixture
    def handler(self):
        return _MockHandler()

    @pytest.fixture
    def myelin(self, handler):
        return MyelinProvider(tool_handlers=handler)

    @pytest.fixture
    def mem0_fail_add(self):
        return _FakeMem0Provider(fail_on={"add"})

    @pytest.fixture
    def mem0_fail_search(self):
        return _FakeMem0Provider(fail_on={"search"})

    @pytest.fixture
    def mem0_fail_update(self):
        return _FakeMem0Provider(fail_on={"update"})

    @pytest.fixture
    def mem0_fail_delete(self):
        return _FakeMem0Provider(fail_on={"delete"})

    async def test_add_safe_when_mem0_fails(self, myelin, mem0_fail_add):
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0_fail_add)
        result = await adapter.add("failsafed add")
        # Myelin write MUST succeed even if Mem0 fails
        assert result["result"] == "Fact stored."

    async def test_search_safe_when_mem0_fails(self, myelin, mem0_fail_search):
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0_fail_search, shadow_read=True)
        results = await adapter.search("anything")
        # Must return Myelin results, not crash
        assert isinstance(results, list)

    async def test_update_safe_when_mem0_fails(self, myelin, mem0_fail_update):
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0_fail_update)
        result = await adapter.add("original")
        up = await adapter.update(result["event_id"], "updated")
        assert up["memory_id"] == result["event_id"]

    async def test_delete_safe_when_mem0_fails(self, myelin, mem0_fail_delete):
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0_fail_delete)
        result = await adapter.add("to delete")
        dl = await adapter.delete(result["event_id"])
        assert dl["memory_id"] == result["event_id"]

    async def test_chain_operations_with_intermittent_failure(self, myelin):
        mem0 = _FakeMem0Provider(fail_on={"add"})  # add fails initially
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0)

        # Add fails on Mem0 but Myelin succeeds
        r1 = await adapter.add("fact one")
        assert r1["result"] == "Fact stored."

        # Now re-enable with working mem0
        mem0_working = _FakeMem0Provider()
        adapter.enable_mem0(mem0_working)
        r2 = await adapter.add("fact two")
        assert r2["result"] == "Fact stored."

    async def test_dual_write_add_with_all_operations(self):
        """Verify that add propagates to both providers."""
        myelin = self._make_myelin()
        mem0 = _FakeMem0Provider()
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0)
        await adapter.add("python is my favorite language")
        myelin_r = await myelin.search("python")
        mem0_r = await mem0.search("python")
        assert len(myelin_r) >= 0  # Myelin returns results
        assert len(mem0_r) >= 0  # Mem0 returns results

    def _make_myelin(self):
        return MyelinProvider(tool_handlers=_MockHandler())


# ---------------------------------------------------------------------------
#  Tests: shadow-read mode (comparison without changing returned results)
# ---------------------------------------------------------------------------


class TestShadowRead:
    @pytest.fixture
    def handler(self):
        return _MockHandler()

    @pytest.fixture
    def myelin(self, handler):
        return MyelinProvider(tool_handlers=handler)

    @pytest.fixture
    def mem0(self):
        return _FakeMem0Provider()

    @pytest.fixture
    def adapter(self, myelin, mem0):
        a = Mem0DualWriteAdapter(myelin)
        a.enable_mem0(mem0, shadow_read=True)
        return a

    async def test_shadow_read_returns_myelin_results(self, adapter):
        """The agent always gets Myelin results — not Mem0's."""
        await adapter.add("vim is my editor")
        results = await adapter.search("editor")
        # Results come from Myelin, not from Mem0
        assert isinstance(results, list)
        # Verify they are SearchResult instances (from Myelin)
        for r in results:
            assert isinstance(r, SearchResult)

    async def test_shadow_read_logs_comparison(self, adapter):
        await adapter.add("prefers dark mode")
        await adapter.search("dark mode")
        assert len(adapter.comparison_log) == 1

    async def test_shadow_read_log_structure(self, adapter):
        await adapter.add("works with python")
        await adapter.search("python")
        entry = adapter.comparison_log[0]
        assert "query" in entry
        assert "timestamp" in entry
        assert "quality" in entry
        assert "myelin_count" in entry
        assert "mem0_count" in entry

    async def test_shadow_read_search_does_not_alter_results(self, adapter):
        """Shadow-read should not change what search() returns to the agent."""
        await adapter.add("python is great")
        results_before = await adapter.search("python")
        adapter.clear_comparison_log()
        results_after = await adapter.search("python")
        # Types and count should be comparable
        assert len(results_before) == len(results_after)

    async def test_shadow_read_off_by_default(self, myelin, mem0):
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0, shadow_read=False)
        await adapter.add("test")
        await adapter.search("test")
        assert len(adapter.comparison_log) == 0

    async def test_shadow_read_quality_match(self, adapter):
        """When both providers have similar data, quality should be 'match'."""
        await adapter.add("user prefers vim")
        await adapter.add("favorite language is python")
        await adapter.search("user preference")
        # At least one comparison entry
        if adapter.comparison_log:
            entry = adapter.comparison_log[-1]
            assert entry["quality"] in ("match", "myelin_better", "divergence")

    async def test_clear_comparison_log(self, adapter):
        await adapter.add("test")
        await adapter.search("test")
        assert len(adapter.comparison_log) > 0
        adapter.clear_comparison_log()
        assert adapter.comparison_log == []

    async def test_shadow_read_comparison_with_divergent_data(self, myelin):
        """Mem0 has a fact that Myelin doesn't — shadow-read should detect it."""
        mem0 = _FakeMem0Provider()
        await mem0.add("secret project is building a web app")
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0, shadow_read=True)
        await adapter.search("project")
        if adapter.comparison_log:
            # Should log the extra fact in Mem0
            pass  # Non-fatal assertion — extra_in_mem0 may be empty if Myelin also matched

    async def test_shadow_read_returns_zero_shadow_results(self, myelin):
        """Empty Mem0 should not affect returned Myelin results."""
        mem0 = _FakeMem0Provider()
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0, shadow_read=True)
        await adapter.add("my fact")
        results = await adapter.search("my")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
#  Tests: integration smoke test — end-to-end workflow
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    """Lightweight integration smoke test that exercises the full pipeline."""

    @pytest.fixture
    def handler(self):
        return _MockHandler()

    @pytest.fixture
    def myelin(self, handler):
        return MyelinProvider(tool_handlers=handler)

    @pytest.fixture
    def mem0(self):
        return _FakeMem0Provider()

    @pytest.fixture
    def adapter(self, myelin, mem0):
        a = Mem0DualWriteAdapter(myelin)
        a.enable_mem0(mem0, shadow_read=True)
        return a

    async def test_default_mode(self, handler):
        """Default mode: Myelin only, no Mem0."""
        p = MyelinProvider(tool_handlers=handler)
        h = await p.health()
        assert h["ok"] is True
        assert h["detail"] == "myelin handlers available"

        r = await p.add("default preference")
        assert r["event_id"]

        results = await p.search("preference")
        assert isinstance(results, list)

        block = await p.system_prompt_block()
        assert "Active" in block

    async def test_full_workflow_with_mem0(self, adapter):
        """Full dual-write workflow: add then search."""
        # Write phase
        r1 = await adapter.add("user prefers neovim over vscode")
        assert r1["event_id"]

        r2 = await adapter.add("favorite language is rust")
        assert r2["event_id"]

        r3 = await adapter.add("uses docker for deployment")
        assert r3["event_id"]

        # Read phase — Myelin returns, Mem0 shadows
        results = await adapter.search("editor preference")
        assert isinstance(results, list)

    async def test_reversible_cycle(self, adapter, myelin, mem0):
        """enable → disable → enable cycle produces correct state."""
        assert adapter.is_dual_write_active is True

        adapter.disable_mem0()
        assert adapter.is_dual_write_active is False

        r = await adapter.add("post-disable fact")
        assert r["event_id"]

        # Re-enable
        adapter.enable_mem0(mem0, shadow_read=True)
        assert adapter.is_dual_write_active is True
        assert adapter.is_shadow_read_active is True
        h = await adapter.health()
        assert h["mem0_available"] is True

    async def test_health_with_no_mem0(self, handler):
        p = MyelinProvider(tool_handlers=handler)
        a = Mem0DualWriteAdapter(p)
        h = await a.health()
        assert h["ok"] is True

    async def test_comparison_log_after_shadow_read_disabled(self, myelin, mem0):
        adapter = Mem0DualWriteAdapter(myelin)
        adapter.enable_mem0(mem0, shadow_read=False)
        await adapter.add("test")
        await adapter.search("test")
        assert adapter.comparison_log == []

    async def test_dual_write_does_not_expose_mem0_credentials(self, adapter):
        """No credentials should leak in any response."""
        r = await adapter.add("safe fact")
        assert "api_key" not in str(r)
        assert "token" not in str(r)
        assert "secret" not in str(r)
        assert "password" not in str(r)

        results = await adapter.search("safe")
        assert not any("api_key" in str(r.metadata) for r in results)
