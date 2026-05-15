# Phase 2: Transfer & Multi-Agent

Myelin's uncontested moat. No other OSS memory system enables cross-agent skill transfer.

## The Vision

An agent that has used Myelin for 3 months can transfer its learned procedures to a fresh agent in under 5 seconds. The new agent adapts the procedure to its own toolset automatically, with no manual re-teaching.

## Architecture

```
Agent A (Hermes)                  Agent B (Zo)
     │                                │
     │ learns procedure "deploy"      │
     │ (confidence: 0.85)             │
     │                                │
     │ ──► myelin_transfer_export ──► │
     │     {procedure_id,             │
     │      target_agent: "zo"}       │
     │                                │
     │     ◄── package ────────────── │
     │     {steps, confidence,        │
     │      tool_requirements,        │
     │      adapted_steps}            │
     │                                │
     │                                │──► myelin_transfer_import ──►
     │                                    {package, agent_id: "zo"}
     │
     │                                    ◄── procedure_id ─────────
     │                                    (confidence: 0.6, needs
     │                                     1 human verification)
     │
     │ Agent B now has "deploy" adapted
     │ to its available tools
```

## What We're Building

| Feature | Priority | Description |
|---------|----------|-------------|
| Transfer Protocol v2 | P0 | Capability-aware export/adapt/import with step-level transformation |
| Agent Profile Learning | P0 | Auto-detect agent tools from observation sequences |
| Cross-Agent Context | P1 | Query across multiple agents in one call |
| Transfer Marketplace | P2 | Discover and recommend transferable procedures |

## How It's Different From Phase 1

Phase 1 made Myelin *smarter* — better entities, synthesis, importance, temporal tools.

Phase 2 makes Myelin *connected* — agents trade learned skills. This is the feature that makes Myelin a platform, not just a library.
