# Comparisons

Myelin is not trying to replace every memory product. It is focused on procedural learning: turning repeated agent behavior into executable, confidence-tracked procedures.

## Myelin and mem0

[mem0](https://github.com/mem0ai/mem0) is the strongest reference point for production agent memory. Its public README positions it as a universal memory layer for AI agents, with multi-level memory, entity linking, multi-signal retrieval, temporal reasoning, SDKs, a managed platform, and strong benchmark claims.

Myelin should not compete by saying “mem0 is bad.” The stronger framing is:

| Dimension | mem0 | Myelin |
|-----------|------|--------|
| Primary job | Remember user/session/agent facts | Learn reusable agent procedures |
| Core loop | Add/search memories | Observe/cluster/align/promote/feedback |
| Extraction | LLM-centered memory extraction | Algorithmic core learning; optional embeddings |
| Retrieval | Semantic/BM25/entity/temporal retrieval | Context assembly plus procedure matching |
| Best fit | Personalized assistants, support, user memory | Coding agents, operational agents, repeatable workflows |
| Distribution | SDKs, CLI, cloud, self-hosted, agent skills | MCP-native local learning layer |

The public line:

> mem0 remembers. Myelin learns.

Use it as a category distinction, not an attack.

## Myelin and Sigil

[Sigil](https://github.com/Niraven/sigil-memory) is the predecessor. It explored local memory, knowledge graphs, personas, orchestration, working memory, and multi-agent coordination in one package.

Myelin is narrower and stronger:

| Dimension | Sigil | Myelin |
|-----------|-------|--------|
| Product shape | Broad agent substrate | Focused procedural learning layer |
| Procedural memory | Stores taught procedures | Learns procedures from repeated behavior |
| Distribution | Library/CLI prototype | MCP-native agent integration |
| Moat | Local-first breadth | ClustalW-inspired procedure discovery |
| Future | Idea bank and predecessor | Flagship project |

Recommended status:

> Sigil should point to Myelin as its successor. Myelin should absorb only the parts that make procedural memory better.

## What Not to Claim Yet

Avoid these claims until benchmarked and documented:

- “Better than mem0.”
- “Fastest memory system.”
- “State of the art.”
- “Beats LoCoMo/LongMemEval.”

Prefer claims that are true from the repository:

- Local-first SQLite memory.
- MCP-native tool surface.
- Procedure learning from repeated workflows.
- Deterministic sequence-alignment core.
- Confidence feedback after execution.

## Myelin And Agent Orchestrators

Agent orchestrators such as Hermes, LangGraph, CrewAI, AutoGen, and swarm runners coordinate work. Myelin should sit underneath them as a learning layer.

| Layer | Owns |
|---|---|
| Orchestrator | routing, scheduling, tools, approvals, delegation |
| Myelin | observation, procedure discovery, confidence, recall, transfer |

Public line:

> Orchestrators coordinate agents. Myelin learns from what agents repeatedly do.

For Hermes:

> Hermes operates. Myelin learns the operating procedures.
