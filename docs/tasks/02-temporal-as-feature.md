# Task: Temporal as a Feature

**Status:** Planned
**Priority:** P1
**Assignee:** builder
**Dependencies:** None
**Domain:** development

## Goal

Surface Myelin's temporal state tracking as visible MCP tools. The temporal index already tracks entity state transitions internally, but there's no way for an agent to query "what changed since yesterday" or "what's the current state of X."

This is Myelin's unique advantage over mem0 — make it visible.

## Why

Myelin maintains full state transition history for every entity (`temporal_states` table). This is something mem0 doesn't do. But it's invisible to the agent unless they dig into the database.

Two new MCP tools:
- `myelin_what_changed(domain, since)` — returns state transitions in a domain since a timestamp
- `myelin_entity_status(entity_name)` — returns current state + recent transitions for an entity

## Architecture

```
Agent query:
  myelin_what_changed("infrastructure", "2026-05-14")
      ↓
  TemporalIndex.get_domain_transitions_since(domain, cutoff)
      ↓
  Returns: [{"entity": "cloudflared", "from": "running", "to": "restarted", "when": "2026-05-15T09:00"}, ...]
      ↓
  Formatted as markdown table for agent consumption

Agent query:
  myelin_entity_status("cloudflared")
      ↓
  TemporalIndex.get_current_state(entity_id) + TemporalIndex.get_state_history(entity_id, limit=5)
      ↓
  Returns: {"current": "running since 2026-05-15", "history": [...], "transitions": 3}
```

## Implementation Plan

### 1. Add domain query methods to TemporalIndex

**File:** `/tmp/myelin/src/myelin/knowledge/temporal.py`

```python
def get_domain_transitions_since(self, domain: str, since: str) -> list[dict]:
    """Get all state transitions in a domain since a timestamp."""
    return self.db.fetchall("""
        SELECT ts.*, e.canonical_name as entity_name
        FROM temporal_states ts
        LEFT JOIN entities e ON e.id = ts.entity_id
        WHERE ts.domain = ? AND ts.created_at >= ?
        ORDER BY ts.created_at DESC
    """, (domain, since))

def get_state_history(self, entity_id: str, limit: int = 10) -> list[dict]:
    """Get full state history for an entity."""
    return self.db.fetchall("""
        SELECT * FROM temporal_states
        WHERE entity_id = ?
        ORDER BY valid_from DESC
        LIMIT ?
    """, (entity_id, limit))
```

### 2. Add MCP tool definitions

**File:** `/tmp/myelin/src/myelin/server.py`

```python
Tool(
    name="myelin_what_changed",
    description="Get state transitions in a domain since a timestamp. Shows what changed and when.",
    inputSchema={
        "type": "object",
        "properties": {
            "domain": {"type": "string", "description": "Domain to query (e.g. 'infrastructure', 'development')"},
            "since": {"type": "string", "description": "ISO timestamp or date (e.g. '2026-05-14' or '2026-05-14T09:00:00')"},
        },
        "required": ["domain", "since"],
    },
),
Tool(
    name="myelin_entity_status",
    description="Get current state and recent transitions for an entity.",
    inputSchema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Entity name (e.g. 'cloudflared', 'obsidian')"},
        },
        "required": ["name"],
    },
),
```

### 3. Add handler methods

**File:** `/tmp/myelin/src/myelin/tools/handlers.py`

```python
async def what_changed(self, domain: str, since: str) -> dict:
    """Return state transitions in a domain since a timestamp."""
    states = self.temporal.get_domain_transitions_since(domain, since)
    
    if not states:
        return {"domain": domain, "since": since, "changes": [], "message": "No changes found"}
    
    return {
        "domain": domain,
        "since": since,
        "change_count": len(states),
        "changes": [
            {
                "entity": s.get("entity_name", "unknown"),
                "state": s["state_description"],
                "when": s.get("valid_from"),
                "confidence": s.get("confidence", 0.5),
            }
            for s in states[:50]  # limit to 50
        ],
    }

async def entity_status(self, name: str) -> dict:
    """Get current state and history for an entity."""
    found = self.entities.search(name)
    if not found:
        return {"found": False, "message": f"Entity '{name}' not found"}
    
    entity = found[0]
    current = self.temporal.get_current_state(entity["id"])
    history = self.temporal.get_state_history(entity["id"], limit=5)
    transitions = self.temporal.get_state_transitions(entity["id"])
    
    return {
        "found": True,
        "entity": {
            "id": entity["id"],
            "name": entity["canonical_name"],
            "type": entity["entity_type"],
            "mention_count": entity.get("mention_count", 0),
        },
        "current_state": {
            "description": current["state_description"],
            "since": current.get("valid_from"),
            "confidence": current.get("confidence", 0.5),
        } if current else None,
        "recent_history": [
            {"state": h["state_description"], "from": h.get("valid_from"), "until": h.get("valid_until")}
            for h in history
        ],
        "transitions": len(transitions),
    }
```

## Acceptance Criteria

- [ ] `myelin_what_changed("infrastructure", "2026-05-14")` returns all state transitions in infrastructure domain since that date
- [ ] `myelin_entity_status("cloudflared")` returns current state + last 5 transitions
- [ ] Unknown entities return `found: False` with helpful message
- [ ] Empty domains return "No changes found"
- [ ] Results are structured JSON (agent-readable) — no markdown formatting needed
- [ ] Works with existing temporal index — no schema changes
- [ ] Zero regression on existing tests

## Files Changed

| File | Change |
|------|--------|
| `knowledge/temporal.py` | Add `get_domain_transitions_since`, `get_state_history` |
| `server.py` | Add 2 new MCP tool definitions |
| `tools/handlers.py` | Add `what_changed`, `entity_status` handlers |
| `server.py` | Wire handlers into call_tool dispatch map |

## Estimated Effort

- Implementation: ~2 hours
- Testing: ~1 hour
- Documentation: ~30 min
