# Myelin Promotion Kit

**Tagline:** Agents should not relearn workflows. Myelin learns.

**One-liner:** Myelin turns repeated agent behavior into reusable procedures via MCP. No LLM extraction — deterministic algorithms learn from what you already do.

---

## The Problem

Every time an agent deploys code, searches a job board, audits a system, or generates leads — it starts from zero. No memory of how it succeeded last time. No transfer between agents.

**Skills fix this for hand-authored cases. Myelin fixes it for everything else.**

---

## What Myelin Is

A **procedural learning layer** that sits under any agent runtime (Hermes, Zo, Codex, Claude Code, any MCP-compatible agent):

- **Observes** agent actions automatically
- **Clusters** repeated behavior into patterns
- **Promotes** patterns into executable procedures with Bayesian confidence
- **Assembles context** — fuses episodic memory + procedures + entity graph + temporal state + activation scoring
- **Transfers** learned procedures between agents with capability-aware adaptation

---

## What It Is NOT

- NOT a memory database (use mem0, Honcho, Supermemory for facts)
- NOT an orchestrator (use Kanban, DAG, n8n for workflows)
- NOT a RAG system (use vector DBs for document retrieval)
- NOT a skill system — skills are hand-authored, Myelin procedures are auto-discovered

**It's the bridge between "we did this once" and "we know how to do this."**

---

## Key Capabilities

| Capability | What it does |
|------------|-------------|
| Procedure learning | Watches actions, clusters sequences, aligns (ClustalW), promotes patterns |
| Multi-signal retrieval | Fuses 5 signals: text + vector + entity + temporal + ACT-R activation |
| Context assembly | One call = top memories + matching procedures + entity context + suggestions |
| Knowledge graph | Entity extraction from actions, typed edges, BFS/DFS traversal |
| Temporal tracking | Entity state over time with transition history |
| Transfer protocol | Package procedures for cross-agent reuse with capability adaptation |
| Cognitive sleep | Background NREM/REM consolidation, promotion, decay cycles |
| Bayesian confidence | Asymptotic confidence updates with success/failure feedback |
| FSRS scheduling | Spaced repetition for memory optimization |

---

## Architecture (One Diagram)

```
Agent (MCP client)
     │
     ▼
  Myelin MCP Server (21 tools)
     │
     ├── Intelligence Layer (context assembly, multi-signal retriever)
     ├── Memory Layer (episodic, semantic, procedural)
     ├── Knowledge Layer (entities, graph, temporal)
     └── Cognitive Layer (consolidation, sleep, promotion, FSRS)
               │
               ▼
         SQLite + FTS5 + sqlite-vec
         (~/.hermes/data/myelin-hermes.db)
```

Zero external dependencies. Local-first. No cloud required.

---

## Integration (3 Minutes)

```bash
# 1. Install
pip install myelin-memory

# 2. Run MCP server
python -m myelin.server --db ~/.hermes/data/myelin.db --embeddings local

# 3. Add to agent config (MCP client)
# In your agent's MCP config:
{
  "myelin": {
    "command": "python",
    "args": ["-m", "myelin.server", "--db", "~/.hermes/data/myelin.db", "--embeddings", "local"]
  }
}
```

That's it. Start observing, and procedures emerge automatically.

---

## Numbers from Production (Hermes + Myelin, May 2026)

| Metric | Value |
|--------|-------|
| Tests | 636/636 passing |
| CI runs | 24 on main, all green |
| MCP tools | 21 |
| Procedures learned | 9 (growing) |
| Entities extracted | 55 (tools, services, files, concepts) |
| Latency (p50) | 2ms (context), 29ms (query), 55ms (p95 query) |
| Procedure hit rate | 100% (benchmark) |
| Confidence per feedback cycle | +4.5% (Bayesian) |
| Storage | ~2MB for 18 episodes + 10 procedures + 55 entities |

---

## Use Cases We've Proven

1. **Session context** — one call replaces 5 manual lookups
2. **Procedure lifecycle** — teach → execute → feedback → confidence 0.70→0.745
3. **Cross-agent transfer** — Hermes→Zo procedures packaged with adaptation
4. **Pattern discovery** — repeated deployment sequences auto-promoted
5. **Knowledge graph** — GitHub→Linear relationships detected from observations

---

## Comparison

| | Myelin | mem0 | Skills | Custom |
|---|---|---|---|---|
| Auto-learns from behavior | ✅ | ❌ | ❌ | ❌ |
| Procedures with confidence | ✅ | ❌ | ❌ | ❌ |
| Cross-agent transfer | ✅ | ❌ | Manual | ❌ |
| Entity graph | ✅ | ❌ | ❌ | ❌ |
| Temporal tracking | ✅ | ❌ | ❌ | ❌ |
| Background consolidation | ✅ | ❌ | ❌ | ❌ |
| No LLM cost per action | ✅ | ✅ | ✅ | Depends |
| MCP-native | ✅ | ✅ | ❌ | Depends |

---

## Links

- **GitHub:** https://github.com/Niraven/myelin
- **Docs:** https://github.com/Niraven/myelin/blob/main/README.md
- **License:** MIT
- **Author:** Nino (@Niraven) — nino@niraven.dev
