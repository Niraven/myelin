# Task: Hybrid Entity Extraction

**Status:** Planned
**Priority:** P0
**Assignee:** builder
**Dependencies:** None
**Domain:** development/mlops

## Goal

Replace regex-only entity extraction with a two-tier system:
1. **Regex tier** (write time): fast pattern matching for known tools, services, files, errors — same as today, <1ms
2. **LLM tier** (sleep time): lightweight concept extraction for entities the regex patterns miss — conceptual multi-word tools, domain-specific workflows, user-defined terms

## Why

Current entity extraction misses everything interesting. "Google Drive", "project-structure-sync script", "the deployment pipeline" are never captured as entities because they don't match the hardcoded regex patterns. This is the #1 reason the knowledge graph feels sparse.

Without richer entities, the relationship inference, temporal tracking, and context assembly are all starved of signal.

## Architecture

```
Write time (fast path):
  agent action → regex patterns → known entities
                 ↓
              episode stored with entity_mentions

Sleep time (deep path, optional):
  unprocessed episodes → LLM concept extraction → conceptual entities
                                                    ↓
                                                 dedup + merge → entity_mentions
```

## Implementation Plan

### 1. Add `HybridEntityExtractor` class

**File:** `/tmp/myelin/src/myelin/knowledge/entities.py`

```python
class HybridEntityExtractor:
    """Two-tier entity extraction: regex (fast) + LLM (deep)."""
    
    def __init__(self, llm_extract: callable | None = None):
        self.regex = PatternExtractor()  # existing logic
        self.llm = llm_extract  # None = LLM tier disabled
    
    def extract_fast(self, text: str, action: str = "") -> list[dict]:
        """Regex-only extraction. Called at write time. <1ms."""
        return extract_entities_from_text(text, action)
    
    def extract_concepts(self, texts: list[str]) -> list[dict]:
        """LLM-based concept extraction. Called during sleep.
        
        Takes batch of episode texts, returns new entity candidates.
        Each candidate has: name, entity_type (concept), canonical_name.
        """
        if not self.llm:
            return []
        return self.llm(texts)
```

### 2. Add `--llm-extraction` server flag

**File:** `/tmp/myelin/src/myelin/server.py`

Add `--llm-extraction` parser argument (default: None). When set, it points to the LLM endpoint/config for concept extraction. Pass to `HybridEntityExtractor`.

### 3. Add concept extraction provider

**File:** `/tmp/myelin/src/myelin/knowledge/concept_extractor.py` (new)

```
class ConceptExtractor:
    """Extracts conceptual entities from episode text using an LLM.
    
    Takes N episode texts, returns structured entity candidates.
    Uses the configured LLM provider or the agent's own model.
    
    Prompt designed for:
    - Low cost (< 200 tokens per episode batch)
    - Structured JSON output
    - Focus on user-specific tools, services, workflows
    """
    
    PROMPT = """Extract named entities from these agent episode descriptions.
    Focus on tools, services, workflows, and concepts that a regex would miss
    (e.g. "Google Drive", "project sync", "deployment pipeline").
    
    Episodes:
    {episodes}
    
    Return JSON array: [{"name": "...", "entity_type": "concept|tool|service"}]
    Only return entities NOT already in this list: {existing_entities}
    """
```

### 4. Integrate into sleep cycle

**File:** `/tmp/myelin/src/myelin/cognitive/sleep.py`

During sleep's entity phase (currently removed as redundant), add LLM concept extraction if the flag is set:

```python
if self.concept_extractor:
    unprocessed = self._get_episodes_without_concept_extraction()
    if unprocessed:
        candidates = self.concept_extractor.extract_concepts(
            [ep["content_text"] for ep in unprocessed]
        )
        for candidate in candidates:
            entity_store.upsert_entity(...)
```

### 5. Wire through to server startup

**File:** `/tmp/myelin/src/myelin/server.py`

```
def create_server(db_path, embedding_provider, llm_extraction=None):
    ...
    entity_store = EntityStore(db)
    
    if llm_extraction:
        extractor = ConceptExtractor(provider=llm_extraction)
    else:
        extractor = None
    
    hybrid_extractor = HybridEntityExtractor(llm_extract=extractor.extract_concepts if extractor else None)
```

## Acceptance Criteria

- [ ] Regex tier runs in <1ms, unchanged for existing patterns
- [ ] LLM tier extracts conceptual entities like "Google Drive", "project-structure-sync"
- [ ] LLM tier is gated behind `--llm-extraction` flag, off by default
- [ ] Entities are deduplicated against existing canonical forms
- [ ] Zero regression on existing test suite
- [ ] Flag default behavior: server starts and runs identically to today

## Files Changed

| File | Change |
|------|--------|
| `knowledge/entities.py` | Add `HybridEntityExtractor` class |
| `knowledge/concept_extractor.py` | New — LLM concept extraction provider |
| `cognitive/sleep.py` | Add concept extraction phase when enabled |
| `server.py` | Add `--llm-extraction` flag, wire through |

## Estimated Effort

- Implementation: ~3-4 hours
- Testing: ~1 hour
- Documentation: ~30 min
