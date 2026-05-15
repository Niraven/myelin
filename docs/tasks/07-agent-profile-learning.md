# Task: Agent Profile Learning

**Status:** Planned
**Priority:** P0 — Required for transfer adaptation
**Dependencies:** None
**Domain:** development

## Goal

Auto-discover what tools each agent has by observing their behavior. Build a capability matrix that the transfer protocol uses for step adaptation.

## Why

Transfer needs to know: "does Zo have `docker`?" Without agent profiles, adaptation has nothing to work with. Currently, agent profiles exist as a schema (`agent_profiles` table) but are never populated from observations.

## Implementation

### 1. Learn tools from episodes

**File:** `transfer/profiling.py`

Every time an episode is recorded, extract tool names from the action text and content. Update the agent profile's tool list:

```python
class AgentProfiler:
    def learn_from_episode(self, episode: dict) -> None:
        """Extract tool usage from an episode and update agent profile."""
        tools = extract_tools_from_text(
            f"{episode.get('action', '')} {episode.get('content_text', '')}"
        )
        for tool in tools:
            self.record_tool_usage(
                agent_id=episode.get("agent_id", "unknown"),
                tool_name=tool,
            )
    
    def record_tool_usage(self, agent_id: str, tool_name: str) -> None:
        """Increment tool usage count, update last_seen."""
        profile = self.get_or_create_profile(agent_id)
        # Update tool frequency in agent_profiles.tools JSON
    
    def get_toolset(self, agent_id: str, min_usage: int = 3) -> list[str]:
        """Get tools an agent has used at least min_usage times."""
        # Returns: ["git", "docker", "python", ...]
```

### 2. Wire into episode recording

**File:** `tools/handlers.py` (inside `_record_episode`)

```python
def _record_episode(self, episode):
    episode_id = self.episodic.record(episode)
    self.entities.process_episode(...)
    self.profiler.learn_from_episode({
        "agent_id": episode.agent_id,
        "action": episode.action,
        "content_text": episode.content_text,
    })
    return episode_id
```

### 3. Tool extraction from text

**File:** `transfer/profiling.py`

Reuse the existing entity extraction patterns but focus on tool names:

```python
def extract_tools_from_text(text: str) -> list[str]:
    """Extract tool names from episode text.
    
    Catches: git, docker, npm, python, kubectl, etc.
    Uses existing TOOL_PATTERNS from entities.py
    """
    from ..knowledge.entities import TOOL_PATTERNS
    tools = set()
    for pattern in TOOL_PATTERNS:
        for match in pattern.finditer(text):
            tool = match.group(1).strip().lower()
            # Extract the base tool name (before subcommand)
            base_tool = tool.split()[0] if " " in tool else tool
            tools.add(base_tool)
    return sorted(tools)
```

### 4. Storage

Agent profiles already have a `tools` column (JSON array). Just update it:

```python
profile = self.get_or_create_profile(agent_id)
current_tools = json.loads(profile["tools"]) if isinstance(profile["tools"], str) else profile["tools"]
# Merge new tools
for tool in tools:
    if tool not in current_tools:
        current_tools.append(tool)
self.db.update("agent_profiles", profile["id"], {"tools": current_tools})
```

## Acceptance Criteria

- [ ] Agent profile auto-populates with tools from first 5 episodes
- [ ] `profiler.get_toolset("hermes")` returns ["git", "docker", "npm", "python", ...]
- [ ] Tools deduplicated (same tool recorded 10x = 1 entry, count incremented)
- [ ] Existing episodes backfilled with tool extraction
- [ ] Transfer protocol uses profiler for adaptation decisions
- [ ] Zero regression on existing tests

## Files Changed

| File | Action |
|------|--------|
| `transfer/profiling.py` | Add learn_from_episode, get_toolset, extract_tools_from_text |
| `tools/handlers.py` | Wire profiler.learn_from_episode into _record_episode |
| `tests/test_profiling.py` | New — test tool learning |

## Estimated Effort

- Implementation: 1-2 hours
- Testing: 30 min
