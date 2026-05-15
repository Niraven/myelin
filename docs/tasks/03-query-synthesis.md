# Task: Query-Time Synthesis

**Status:** Planned
**Priority:** P1
**Assignee:** builder
**Dependencies:** Hybrid Entity Extraction (t_d5d55316)
**Domain:** development

## Goal

Instead of dumping raw ranked episodes when an agent queries Myelin, return a synthesized answer with citations. This makes Myelin feel like a thinking partner instead of a search engine.

**Before (current):**
```json
{"results": [
  {"content": "Built sync script...", "composite_score": 0.82},
  {"content": "Fixed 7 mismatches...", "composite_score": 0.75},
]}

**After (target):**
```json
{
  "synthesis": "Based on 3 episodes about Obsidian/Drive sync (avg confidence: 0.85): You built a project-structure-sync script (id: abc), fixed 7 naming mismatches between vault and Drive (id: def), and deleted 9 duplicate folders (id: ghi). Key entities involved: Obsidian, Google Drive.",
  "sources": [
    {"id": "abc", "content": "Built sync script...", "score": 0.82, "role": "primary"},
    {"id": "def", "content": "Fixed 7 mismatches...", "score": 0.75, "role": "supporting"},
  ]
}
```

## Why

Agents (and humans) don't want a firehose of ranked documents. They want an answer they can verify. mem0 returns facts with citations. Myelin should too.

## Architecture

```
myelin_query("obsidian drive sync")
    ↓
MultiSignalRetriever.retrieve() → ranked episodes/procedures
    ↓
if synthesis_enabled:
    → Synthesizer.synthesize(query, top_results)
    → Uses LLM (agent's model or a tiny local model)
    → Returns: {"synthesis": str, "sources": [...]}
else:
    → Raw results (current behavior)

Synthesis prompt (target <500 tokens total):
  "Given this query: {query}
   And these relevant episodes: {episodes_json}
   Produce a 3-5 sentence summary of what happened, citing episode IDs.
   Focus on: what was done, what changed, what entities were involved."
```

## Implementation Plan

### 1. Create Synthesizer module

**File:** `/tmp/myelin/src/myelin/intelligence/synthesizer.py` (new)

```python
class Synthesizer:
    """Synthesizes query results into concise answers with citations."""
    
    def __init__(self, llm_complete: callable | None = None):
        self.llm = llm_complete  # None = synthesis disabled, fall back to raw
    
    def synthesize(self, query: str, results: list[dict], max_sources: int = 5) -> dict:
        """Produce a synthesized answer from ranked retrieval results.
        
        Returns: {"synthesis": str, "sources": [...], "mode": "synthesized"|"raw"}
        """
        if not self.llm or not results:
            return self._fallback(query, results)
        
        top = results[:max_sources]
        prompt = self._build_prompt(query, top)
        
        summary = self.llm(prompt)
        sources = [
            {"id": r.get("id"), "content": r.get("content_text", "")[:100],
             "score": r.get("_composite_score", 0)}
            for r in top
        ]
        
        return {
            "synthesis": summary,
            "sources": sources,
            "source_count": len(sources),
            "mode": "synthesized",
        }
    
    def _fallback(self, query: str, results: list[dict]) -> dict:
        """Raw results when synthesis is unavailable."""
        return {
            "synthesis": None,
            "results": results,
            "source_count": len(results),
            "mode": "raw",
            "message": "Synthesis unavailable. Returning raw results.",
        }
```

### 2. Wire into query handler

**File:** `/tmp/myelin/src/myelin/tools/handlers.py`

Modify `query()` method to accept optional `synthesize=True` parameter:

```python
async def query(self, query: str, limit: int = 10, domain: str | None = None,
                weights: dict | None = None, synthesize: bool = False) -> dict:
    query_vec = self.embedder.embed(query) or None
    results = self.retriever.retrieve(query, query_embedding=query_vec, ...)
    
    if synthesize and self.synthesizer:
        synthesis = self.synthesizer.synthesize(query, results)
        return synthesis  # includes .results as fallback
    
    return {"query": query, "results": [...], "total": len(results)}
```

### 3. Wire LLM provider into server

**File:** `/tmp/myelin/src/myelin/server.py`

Add `--synthesis-model` argument (default: None). When set, use that model endpoint for synthesis calls:

```python
if synthesis_model:
    llm = lambda prompt: call_llm(synthesis_model, prompt)
    synthesizer = Synthesizer(llm_complete=llm)
else:
    synthesizer = Synthesizer()  # disabled, falls back to raw
```

For the initial implementation, the LLM can be:
- A simple subprocess call to an agent's CLI
- An HTTP POST to a local endpoint (e.g., Ollama, LiteLLM)
- Or disabled entirely (raw results are still useful)

### 4. Add MCP tool parameter

**File:** `/tmp/myelin/src/myelin/server.py`

Update `myelin_query` tool schema to include optional `synthesize` boolean:

```python
"synthesize": {
    "type": "boolean",
    "default": False,
    "description": "When true, synthesize top results into a concise answer with citations",
},
```

## Acceptance Criteria

- [ ] When `synthesize=True` and an LLM is configured, returns `{"synthesis": "...", "sources": [...]}`
- [ ] When `synthesize=False` (default), returns raw results (backward compatible)
- [ ] When no LLM configured, synthesize=True falls back to raw results with a message
- [ ] Synthesis prompt is under 500 tokens
- [ ] Each cited fact includes the source episode ID
- [ ] Zero regression on existing query behavior
- [ ] Works without any additional dependencies

## Files Changed

| File | Change |
|------|--------|
| `intelligence/synthesizer.py` | New — synthesis orchestration |
| `tools/handlers.py` | Add synthesize parameter to query() |
| `server.py` | Add `--synthesis-model` flag, wire Synthesizer |

## Estimated Effort

- Implementation: ~3-4 hours
- Testing: ~1 hour
- Documentation: ~30 min
