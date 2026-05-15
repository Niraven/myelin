# Task: Cross-Agent Context

**Status:** Planned
**Priority:** P1
**Dependencies:** Agent Profile Learning (profiling.py)
**Domain:** development

## Goal

Allow agents to query across all connected agents in one call. "What did Zo learn about deployment?" should return results from Zo's episodes even when queried by Hermes.

## Why

Currently Myelin scopes all queries to a single agent. If Hermes asks about "deployment", it only sees Hermes' own episodes. But Zo might have learned a better deployment pattern. Cross-agent context makes the entire ecosystem smarter.

## Implementation

### 1. Multi-agent query scope

**File:** `memory/retriever.py`

Add `agent_ids` parameter to `retrieve()` — when `None` or `["*"]`, search across all agents:

```python
def retrieve(self, query, query_embedding=None, domain=None, limit=10,
             weights=None, agent_ids: list[str] | None = None):
    """If agent_ids is None or ['*'], search all agents.
       Otherwise filter to specific agents."""
```

### 2. Filter episodes by agent

**File:** `memory/episodic.py`

```python
def search_hybrid(self, text_query, query_vec=None, limit=10, 
                  agent_ids: list[str] | None = None):
    """Hybrid search optionally filtered by agent IDs."""
    # When agent_ids is None or ['*'], no filter
    # When specified, add WHERE agent_id IN (...) clause
```

### 3. Tag results by source agent

**File:** `tools/handlers.py`, `memory/retriever.py`

```python
# In retriever, add agent_id to result metadata
result["source_agent"] = candidate.get("agent_id", "unknown")
```

### 4. New MCP tool parameter

**File:** `server.py`

Add `agent_ids` parameter to `myelin_query` and `myelin_context`:

```python
"agent_ids": {
    "type": "array",
    "items": {"type": "string"},
    "description": "Filter by agent IDs. Omit or ['*'] for all agents.",
}
```

### 5. Per-agent confidence calibration

When returning results from other agents, apply a confidence discount:

```python
confidence_multiplier = {
    "self": 1.0,       # Same agent = full confidence
    "sibling": 0.85,   # Different agent in same system = slight discount
    "unknown": 0.7,    # Unknown agent = significant discount
}
```

## Acceptance Criteria

- [ ] `myelin_query("deploy", agent_ids=["*"])` returns results from ALL agents
- [ ] Each result includes `source_agent` field
- [ ] Cross-agent results confidence-discounted
- [ ] Default behavior (no agent_ids) unchanged — backward compatible
- [ ] Integration test: query across 2 simulated agents

## Files Changed

| File | Action |
|------|--------|
| `memory/retriever.py` | Add agent_ids parameter to retrieve() |
| `memory/episodic.py` | Add agent_id filtering to search_hybrid |
| `tools/handlers.py` | Pass agent_ids through query/context handlers |
| `server.py` | Add agent_ids to tool schemas |
| `tests/test_retrieval.py` | Add multi-agent query tests |

## Estimated Effort

- Implementation: 2-3 hours
- Testing: 1 hour

## Task: Transfer Marketplace

**Priority:** P2 (stretch)
**Dependencies:** Transfer Protocol v2, Agent Profile Learning
**Domain:** development

### Goal

Proactive procedure discovery. When Agent A connects, Myelin checks: "Agent B has 3 procedures you don't, at 0.7+ confidence. Want to import them?"

### Implementation

**File:** `transfer/discovery.py` (new)

```python
class TransferDiscovery:
    def discover_for_agent(self, agent_id: str, min_confidence: float = 0.6) -> list[dict]:
        """Find procedures from other agents this agent doesn't have."""
    
    def suggest_import(self, agent_id: str, procedure_id: str) -> dict:
        """Suggest importing a procedure from another agent."""
        # Returns: estimated benefit, confidence, adaptation needed
```

### MCP Tools

- `myelin_transfer_suggest(agent_id)` — return suggested imports
- `myelin_transfer_accept(procedure_id, target_agent)` — import with one call

### Acceptance Criteria

- [ ] Agent sees suggested procedures from other agents
- [ ] Suggestion includes: procedure name, source agent, confidence, adaptation needed
- [ ] One-click import with full adaptation pipeline

## Files Changed (Marketplace)

| File | Action |
|------|--------|
| `transfer/discovery.py` | New — procedure discovery |
| `transfer/protocol.py` | Add suggest/accept methods |
| `server.py` | Wire marketplace MCP tools |
| `tests/test_marketplace.py` | New — marketplace tests |
