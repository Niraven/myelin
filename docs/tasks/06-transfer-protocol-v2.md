# Task: Transfer Protocol v2

**Status:** Planned
**Priority:** P0 — Core moat
**Dependencies:** None (builds on existing transfer/ module)
**Domain:** development

## Goal

Upgrade the current basic export/import to a capability-aware transfer protocol that:
1. Introspects what tools the source procedure requires
2. Checks if the target agent has those tools
3. Adapts steps when tools don't match
4. Produces confidence-calibrated procedures on import

## Why

Current transfer just serializes a procedure and deserializes it. If the source used `git push` and the target has no git, the procedure breaks silently. For transfer to be useful across different agent types (Hermes ↔ Zo ↔ Codex), it must adapt.

## Architecture

```
export(procedure_id, target_agent)
  │
  ├─ 1. Load procedure + its tool requirements
  ├─ 2. Load target agent profile (tools, capabilities)
  ├─ 3. Compare tool sets → gap analysis
  ├─ 4. For each step:
  │      if step.tool in target.tools → keep as-is
  │      else → search for alternative
  │             if found → rewrite step
  │             else → flag for human review
  └─ 5. Return package with adapted steps + confidence delta

import(package, agent_id)
  │
  ├─ 1. Load adapted steps
  ├─ 2. Calculate confidence discount:
  │      base_confidence × adaptation_quality
  │       - 1.0 if no adaptation needed
  │       - 0.8 if minor tool substitution
  │       - 0.6 if multiple substitutions
  │       - 0.4 if steps flagged for review
  ├─ 3. Store as draft procedure with discounted confidence
  └─ 4. Return procedure_id + confidence + review_needed flags
```

## Implementation Plan

### 1. Extend AgentProfile model

**File:** `transfer/profiling.py`

```python
@dataclass
class AgentCapability:
    tool_name: str
    tool_type: str  # 'terminal', 'web', 'file', 'api', etc.
    usage_count: int
    last_used: str | None
    
class AgentProfiler:
    def get_toolset(self, agent_id: str) -> list[AgentCapability]:
        """Get tools this agent uses, ranked by frequency."""
    
    def has_tool(self, agent_id: str, tool_name: str) -> bool:
        """Check if agent has a specific tool capability."""
    
    def find_alternative(self, required_tool: str, target_agent: str) -> str | None:
        """Find the closest available tool on the target agent."""
```

### 2. Add tool mapping table

**File:** `transfer/tool_map.py` (new)

Maps common tool names to alternatives across agents:

```python
TOOL_ALIASES = {
    "git": ["gh", "github-cli"],
    "docker": ["podman", "nerdctl"],
    "npm": ["yarn", "pnpm"],
    "pip": ["pip3", "conda"],
    "kubectl": ["oc", "k"],
    "psql": ["mysql", "sqlite3"],
    "aws": ["gcloud", "az"],
}

TOOL_TYPE_MAP = {
    "git push": {"type": "git", "action": "push"},
    "docker build": {"type": "docker", "action": "build"},
    "npm test": {"type": "npm", "action": "test"},
    "pip install": {"type": "pip", "action": "install"},
}
```

### 3. Build adaptation engine

**File:** `transfer/adaptation.py` (new)

```python
class StepAdaptationEngine:
    def adapt(self, step: dict, target_tools: list[str]) -> dict:
        """Adapt a single step for the target agent's toolset.
        
        Returns adapted step + adaptation_quality (0.0-1.0).
        """
    
    def analyze_requirements(self, procedure: dict) -> list[str]:
        """Extract tool requirements from procedure steps."""
    
    def calculate_confidence(self, base: float, adaptations: list[float]) -> float:
        """Calculate post-adaptation confidence."""
```

### 4. Upgrade transfer protocol

**File:** `transfer/protocol.py`

```python
class TransferProtocol:
    async def export_procedure(
        self, procedure_id: str, source_agent: str, target_agent: str
    ) -> dict:
        """Export with capability-aware adaptation."""
    
    async def import_procedure(self, package: dict, agent_id: str) -> dict:
        """Import with confidence calibration."""
```

### 5. Wire MCP tools

**File:** `server.py`, `tools/handlers.py`

New MCP tools:
- `myelin_transfer_export(procedure_id, target_agent)` — adapted export
- `myelin_transfer_import(package, agent_id)` — calibrated import
- `myelin_transfer_discover(agent_id, min_confidence)` — find transferable procedures

## Acceptance Criteria

- [ ] Export analyzes tool requirements from procedure steps
- [ ] Import adapts steps when tools don't match
- [ ] Confidence is discounted proportionally to adaptation complexity
- [ ] Full adaptation (no changes needed) → confidence preserved
- [ ] Major adaptation (3+ substitutions) → draft status, flagged for review
- [ ] No tool found for a step → step flagged, procedure stays draft
- [ ] All existing transfer tests pass
- [ ] New integration test: transfer Hermes→Zo with tool mismatch

## Files Changed

| File | Action |
|------|--------|
| `transfer/protocol.py` | Upgrade export/import with adaptation |
| `transfer/profiling.py` | Add AgentCapability, get_toolset, find_alternative |
| `transfer/adaptation.py` | New — step-level adaptation engine |
| `transfer/tool_map.py` | New — tool aliases and type mapping |
| `server.py` | Wire new MCP tools |
| `tools/handlers.py` | Add handler methods |
| `tests/test_transfer.py` | Add adaptation tests |

## Estimated Effort

- Design: 1 hour
- Implementation: 4-5 hours
- Testing: 1-2 hours
