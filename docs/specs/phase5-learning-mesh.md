# Myelin Phase 5 — Multi-Agent Learning Mesh
## Architecture Specification

**Status:** Draft v1 | **Author:** Hermes Agent | **Date:** 2026-05-15  
**Applies:** System Design ⨯ Security ⨯ AI/ML ⨯ Psychology ⨯ Product

---

## 1. Problem Space

Myelin currently learns from single-agent observation streams. As agent systems grow to swarms (Hermes orchestrator, Kanban profiles, A2A protocol), Myelin faces three scaling challenges:

1. **Observation overload** — N agents × M observations/minute → write amplification, context pollution, storage bloat
2. **Cross-agent knowledge conflict** — Agent A learns a procedure for FastAPI auth that contradicts what Agent B learned. Myelin needs reconciliation, not just accumulation.
3. **Confidence calibration without a single ground truth** — When 10 agents report different success rates for the same procedure, which confidence do you trust?

---

## 2. Architecture: Learning Mesh

```
┌─────────────────────────────────────────────────────────┐
│                     LEARNING MESH                        │
│                                                          │
│  ┌──────┐   ┌──────┐   ┌──────┐            ┌────────┐ │
│  │Agent │   │Agent │   │Agent │    ...      │Orch.   │ │
│  │  A   │   │  B   │   │  C   │            │Loop    │ │
│  └──┬───┘   └──┬───┘   └──┬───┘            └───┬────┘ │
│     │          │          │                     │      │
│     └──────────┴──────────┴─────────────────────┘      │
│                              │                          │
│                         ┌────▼────┐                     │
│                         │ INGEST  │                     │
│                         │ QUEUE   │                     │
│                         │ (Kafka) │                     │
│                         └────┬────┘                     │
│                              │                          │
│              ┌───────────────┼───────────────┐         │
│              ▼               ▼               ▼         │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│     │ OBSERVATION│  │  PROCEDURE│  │  ENTITY    │    │
│     │  STORE    │  │  LEARNER  │  │  TRACKER   │    │
│     └────────────┘  └────────────┘  └────────────┘    │
│              │               │               │          │
│              └───────────────┼───────────────┘         │
│                              │                          │
│                         ┌────▼────┐                     │
│                         │ RECON-  │                     │
│                         │ CILIATE │                     │
│                         │ (merge) │                     │
│                         └────┬────┘                     │
│                              │                          │
│                    ┌─────────┴─────────┐               │
│                    ▼                   ▼               │
│           ┌──────────────┐  ┌────────────────┐        │
│           │  CONFIDENCE   │  │  KNOWLEDGE     │        │
│           │  FUSION      │  │  GRAPH         │        │
│           └──────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

**Key architectural decision:** The ingest queue decouples observation producers from consumers. This is the Kappa architecture pattern — streaming ingest, batch-learned procedures. Single-agent observation.write() is the simplest path; multi-agent goes through the queue.

---

## 3. System Design: Ingest Queue (Observation Gateway)

### Current State
Agents call `myelin.observe(action, ...)` directly. This is synchronous — if Myelin is slow, agents block.

### Phase 5 Design
```
Agent → observe() → IngestQueue → [buffer] → BatchWriter → SQLite
                                    ↓ async
                              ObservationStore (parquet)
```

**Decision driver:** Backpressure from CAP theorem — availability trumps consistency for observations. If an observation is lost, the agent retries (idempotency key). This makes the ingest path AP (available, eventually consistent).

### Queue Backend Decision

| Criteria | Kafka | Redis Streams | SQLite WAL | Winner |
|----------|-------|---------------|------------|--------|
| Ops overhead | High | Low | Zero | SQLite |
| Persistence | Configurable | Configurable | Full | SQLite |
| Replay | ✅ Seek | ✅ Range | ✅ Timestamp | Tie |
| Multi-producer | ✅ | ✅ | ✅ (WAL) | Tie |
| **Already running** | ❌ | ❌ | ✅ (Myelin uses it) | **SQLite** |

**Decision:** Use SQLite WAL mode with a dedicated `observation_queue` table. Poll every 100ms in a bg thread. This avoids adding Kafka ops overhead for a single-node system.

### Security: Observation ACL

Not all agents should see all knowledge. The Learning Mesh adds:

```
Observation {
    agent_id: str,
    agent_profile: str,         # "hermes-default", "kanban-builder", etc.
    tenant: str | None,         # Multi-tenant isolation
    sensitivity: Literal["public", "internal", "restricted"],
    action: str,
    ...
}
```

**Access rules:**
- `public`: any agent can learn from
- `internal`: same-profile agents only (research → research, builder → builder)
- `restricted`: same agent_id only (personal procedures)

This prevents a kanban-builder from being flooded with metacognition procedures from the orchestrator, while still allowing cross-profile transfer of public patterns (e.g., "how to structure a FastAPI app" is public).

---

## 4. AI/ML: Confidence Fusion

### Current State
Single confidence per procedure, updated via Bayesian rule on success/failure feedback.

### Phase 5: Multi-Source Fusion

When N agents report the same procedure with different confidences, Myelin uses **weighted evidence fusion**:

```
P(procedure_is_correct | E_1, ..., E_N) ∝
    P(procedure)^(1-N) × ∏_{i=1}^N P(E_i | procedure)^w_i

where w_i = log(1 + total_observations_from_agent_i)
```

This weights agents with more experience higher. A builder who's run a deploy procedure 50 times gets more say than a researcher who ran it once.

### Psychology: Curiosity Engine Update

The curiosity engine currently uses information gain (uncertainty reduction) to prioritize what to learn. Phase 5 adds **information gap theory** (Loewenstein, 1994) — curiosity spikes when an agent knows *enough to recognize what it doesn't know*.

**Implementation:** After confidence fusion, if any source's confidence deviates from the fused mean by >0.2, that procedure gets flagged for re-observation. This creates a curiosity loop: "I thought I knew this, but Agent X disagrees — let me watch more."

---

## 5. Reconciliation Engine

The reconsolidator currently handles single-agent memory updates. Phase 5 adds **multi-source reconciliation**:

### Conflict Types

| Type | Example | Resolution |
|------|---------|------------|
| **Sequence divergence** | Agent A deploys via Docker, Agent B via k8s | Merge into branching procedure with precondition selectors |
| **Parameter divergence** | Agent A uses 4 workers, Agent B uses 8 | Abstract parameter; record both as observations with context |
| **Contradictory feedback** | Agent A says procedure succeeded, Agent B says it failed | Re-run with both contexts flagged for LLM review |

### Implementation: Branching Procedures

```python
class ProcedureStep:
    action: str
    preconditions: list[Condition]    # NEW
    branches: list[Branch]            # NEW
    
class Branch:
    context: str                      # "when deploying to kubernetes"
    steps: list[ProcedureStep]
    confidence: float
    observations_count: int
```

This mirrors how the system-design domain handles tradeoffs — "it depends on context" encoded directly into the procedure structure.

---

## 6. Product: Agent-Facing API

### Current
```python
myelin.observe(...)       # record
myelin.context(query)     # retrieve
myelin.execute(proc_id)   # run
```

### Phase 5 Additions
```python
myelin.observe_batch(events)       # batch ingest (from orchestrator fan-out)
myelin.reconcile(agent_id)         # trigger reconciliation for an agent's observations
myelin.transfer(source, target,    # transfer with context adaptation
               context={"tools": [...]})
myelin.fused_confidence(proc_id)   # returns calibrated multi-source confidence
myelin.knowledge_gaps(agent_id)    # returns what the agent should be curious about
```

---

## 7. Quality Gates (from software engineering knowledge)

Before shipping Phase 5:

- [ ] **System design:** Read-after-write consistency on procedure learn → agent sees their own procedure within 500ms. Eventual consistency for cross-agent transfer is acceptable.
- [ ] **Security:** All observations validate `sensitivity` classification. Tenant isolation enforced at query time via WHERE agent_id/tenant. No cross-tenant data leakage.
- [ ] **AI/ML:** Confidence fusion tested with synthetic multi-agent data — verify that high-experience agents dominate initially, and confidence converges with more observations.
- [ ] **Performance:** Batching 100 observations should take <50ms total. Ingest queue should handle 1000 obs/s without backpressure on agents.
- [ ] **Testing:** Property-based tests for reconciliation (fuzzing with divergent procedure sequences), integration tests for multi-agent scenario (3 agents with different procedure variants), benchmark for confidence fusion vs single-agent baseline.

---

## 8. Implementation Plan

| Step | What | Depends On | Est. Effort |
|------|------|------------|-------------|
| 1 | ObservationQueue class (SQLite WAL, bg poller, batch flush) | Nothing | 2 days |
| 2 | Sensitivity ACL on observations (public/internal/restricted) | Step 1 | 1 day |
| 3 | Confidence fusion (weighted evidence from N sources) | Nothing | 2 days |
| 4 | Information gap curiosity trigger | Step 3 | 1 day |
| 5 | Reconciliation: branching procedures with preconditions | Step 1 + 3 | 3 days |
| 6 | API surface (observe_batch, reconcile, transfer, fused_confidence, knowledge_gaps) | Steps 1-5 | 2 days |
| 7 | Property-based tests + multi-agent integration tests | Steps 1-6 | 2 days |

**Total:** ~13 days for a single engineer. The subsytems are decoupled — Steps 1, 3, and 6 can start in parallel.

---

## 9. References

This spec applies knowledge from:
- **System Design:** Kappa architecture (streaming ingest), CAP theorem (AP path for observations), backpressure patterns
- **Security:** ACL-based access control, least-privilege knowledge sharing, tenant isolation via scoped queries
- **AI/ML:** Bayesian evidence fusion with experience weighting, conformal prediction confidence bounds
- **Psychology:** Information Gap Theory (Loewenstein 1994), optimal uncertainty for curiosity-driven learning
- **Product:** API surface that reveals the mental model — knowledge_gaps() makes the learning OS visible to agents
