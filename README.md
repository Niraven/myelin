# Myelin

Cognitive memory layer for AI agents. Learns procedures from behavior, transfers knowledge across agents via MCP.

## What it does

Myelin observes how AI agents work, detects patterns, and promotes repeated behaviors into reusable procedures. One agent learns, all agents benefit.

**Core capabilities:**
- **Procedural learning** -- automatically extracts workflows from repeated agent behavior using ACT-R activation math
- **Procedure composition** -- chains simple procedures into complex workflows (SOAR-inspired chunking)
- **Cross-agent transfer** -- shares learned procedures between agents with capability-aware similarity scoring
- **Active metacognition** -- identifies knowledge gaps and creates learning goals (SOAR impasse detection)
- **Bayesian confidence** -- asymptotic confidence tracking with calibration correction

## Architecture

Three memory types (inspired by cognitive science):
- **Episodic** -- raw observations of agent actions with full context
- **Semantic** -- facts, reflections, and higher-order insights
- **Procedural** -- learned step-by-step workflows with branching

Six background cognitive processes:
- **Consolidator** -- merges similar episodes into semantic summaries
- **Reflector** -- generates higher-order insights (Stanford Generative Agents)
- **Promoter** -- detects patterns and promotes to procedures (ACT-R activation)
- **Composer** -- chains compatible procedures (SOAR chunking)
- **Decayer** -- Ebbinghaus forgetting curve for unused memories
- **Challenger** -- tests beliefs against contradictory evidence

## Quick start

```bash
pip install myelin
```

### As an MCP server

```json
{
  "mcpServers": {
    "myelin": {
      "command": "myelin",
      "args": ["--embeddings", "none"]
    }
  }
}
```

### With local embeddings (better search)

```bash
pip install myelin[embeddings]
myelin --embeddings local
```

## MCP tools

| Tool | What it does |
|------|-------------|
| `myelin_observe` | Record an agent action as episodic memory |
| `myelin_recall` | Search across all memory types |
| `myelin_execute_procedure` | Find and return matching learned procedure |
| `myelin_procedure_feedback` | Report execution outcome (updates Bayesian confidence) |
| `myelin_confidence` | Query confidence levels by domain or procedure |
| `myelin_teach` | Manually teach a procedure |
| `myelin_status` | System status: episodes, procedures, learning goals |

## How promotion works

```
Agent actions --> Episodes --> Cluster detection --> ACT-R activation scoring
  --> Sequence alignment --> Procedure extraction --> Bayesian validation
  --> Composition check --> Active procedure
```

Activation equation (ACT-R, Anderson et al.):
```
B(i) = ln(sum_j(t_j^(-0.5)))
```

A memory's activation depends on how recently AND how frequently it's been accessed. No arbitrary thresholds.

## Development

```bash
git clone https://github.com/Niraven/myelin.git
cd myelin
pip install -e ".[dev]"
pytest tests/ -v
```

## Tech stack

- Python 3.11+
- SQLite + FTS5 (full-text search) + sqlite-vec (vector search)
- MCP (Model Context Protocol) for agent integration
- Pydantic for data models
- Optional: nomic-embed-text-v1.5 for local embeddings

## Research

Myelin combines ideas from four cognitive architectures:
- **ACT-R** (Carnegie Mellon, 40 years) -- activation equations for promotion scoring
- **SOAR** (Michigan, 40 years) -- chunking for procedure composition, impasse detection
- **Stanford Generative Agents** (Park et al., 2023) -- reflection for higher-order knowledge
- **CoALA** (Princeton, 2023) -- modular memory with structured action spaces

## License

MIT
