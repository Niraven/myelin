# Lineage: Sigil to Myelin

Myelin is the focused successor to [Sigil](https://github.com/Niraven/sigil-memory).

Sigil explored a broad local-first agent substrate: memory, knowledge graphs, personas, orchestration, project management, compression, and multi-agent coordination. That exploration was useful, but the surface area was too wide for a sharp open-source wedge.

Myelin takes the strongest idea from that work and goes deep: agents should learn reusable procedures from behavior.

## Why Myelin Is the Flagship

Sigil stores procedural memory when a developer explicitly teaches it a workflow. Myelin discovers procedures automatically:

1. Agents perform actions across sessions.
2. Myelin records those actions as episodic memory.
3. Similar sessions are clustered.
4. Action sequences are aligned with a ClustalW-inspired algorithm.
5. Consensus steps are promoted into executable procedures.
6. Success/failure feedback updates procedure confidence.

That is the difference between storing a workflow and learning a workflow.

## What Myelin Absorbs From Sigil

Good Sigil ideas that belong in Myelin:

- **Proactive Knowledge Activation:** fold into `myelin_context` as proactive briefings before complex tasks.
- **CLI ergonomics:** add `myelin observe`, `myelin recall`, `myelin context`, `myelin teach`, and `myelin demo`.
- **Working memory with TTL:** add short-lived observations that expire or consolidate into episodic memory.
- **Context compression:** compress assembled context when token budget is tight.
- **Persona-aware thresholds:** optionally tune confidence thresholds by agent role or operating mode.

Sigil ideas that should stay out of Myelin core:

- Swarm orchestration.
- Project management.
- General A2A event bus.
- Broad persona framework as a first-class product.

Myelin should integrate with agent frameworks that handle orchestration instead of becoming an orchestration framework itself.

## How to Refer to Sigil

Use this language:

> Sigil was the broad research prototype. Myelin is the focused product: procedural memory for AI agents.

Avoid positioning Sigil as a failed project. It is better framed as the prototype that identified the real moat.
