# Myelin Promotion Kit

## Positioning

**Tagline:** Agents should not relearn workflows.

**One-liner:** Myelin turns repeated agent behavior into reusable, confidence-tracked procedures over MCP.

**Category line:** Procedural learning for AI agents.

> mem0 remembers. Myelin learns.

## Short pitch

Most agent memory systems help an agent recall text. Myelin helps an agent reuse how work gets done.

Agents and orchestrators explicitly send observations to Myelin. Repeated action sequences are clustered, aligned into consensus workflows, and promoted into procedures. When an agent executes one, prediction-linked feedback updates confidence and earns trust from verified outcomes.

Myelin runs locally on SQLite and exposes 25 tools over stdio MCP. Core procedure learning works without an LLM; embeddings and LLM-backed consolidation are optional.

## What it is

- A procedural learning layer under an agent runtime.
- A local store for episodic, semantic, and procedural memory.
- A trust-aware context assembler that shields unverified and cross-domain procedures.
- A transfer layer for adapting learned procedures between agents.

## What it is not

- Not an orchestrator or scheduler.
- Not an automatic shell, browser, or tool-log watcher.
- Not a hosted memory service.
- Not a replacement for fact memory, task management, or human approvals.

## How it works

```text
agent emits observations
  -> Myelin clusters repeated action sequences
  -> explicit myelin_sleep promotes a consensus procedure
  -> myelin_execute_procedure returns a procedure + prediction_id
  -> agent executes with its normal approval gates
  -> myelin_procedure_feedback binds the outcome to that prediction
  -> verified evidence updates confidence and trust
```

Context is deliberately stricter than diagnostic search: automatic context includes only validated or trusted procedures and enforces an exact domain match when a domain is supplied.

## Try it from source

Myelin is not yet published to PyPI.

```bash
git clone https://github.com/Niraven/myelin.git
cd myelin
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python examples/procedure_learning_demo.py
```

Hermes-style orchestrated demo:

```bash
python examples/hermes_procedure_demo.py
```

## MCP setup

Myelin currently uses stdio transport.

```json
{
  "mcpServers": {
    "myelin": {
      "command": "python",
      "args": ["-m", "myelin.server", "--embedding-model", "none"]
    }
  }
}
```

Start with `myelin_context`, `myelin_observe_batch`, `myelin_execute_procedure`, `myelin_procedure_feedback`, `myelin_sleep`, and `myelin_status`. Gate teaching, graph, profile, and transfer tools until the integration is trusted.

## Proof points

- 25 stdio MCP tools.
- Local SQLite + FTS5 default storage.
- Deterministic procedure discovery and confidence updates.
- Prediction-linked, replay-idempotent feedback.
- Trust- and domain-bounded automatic context.
- Capability-aware cross-agent procedure transfer.
- Optional embeddings and LLM-backed consolidation.

Run the repository demo and benchmark for current machine-specific proof rather than publishing stale performance or test counts:

```bash
python examples/procedure_learning_demo.py
python -m myelin.benchmark --counts 100,1000 --json
```

## Launch copy

### Hacker News

**Title:** Show HN: Myelin, procedural learning for AI agents

Myelin is a local-first procedural learning layer for AI agents. Instead of only retrieving facts, it learns reusable workflows from observations agents explicitly emit.

The demo observes repeated deployment runs, extracts their shared action sequence, executes the resulting procedure, and uses prediction-linked feedback to earn trust. The core loop runs locally on SQLite/FTS5 and does not require an LLM.

### LinkedIn / X

Most agent memory stores facts. Myelin learns procedures.

It turns repeated agent workflows into confidence-tracked procedures, then uses prediction-linked feedback to distinguish a successful run from an unverified claim.

Local-first. SQLite-backed. stdio MCP. LLM optional.

Demo: `python examples/procedure_learning_demo.py`

## Links

- GitHub: https://github.com/Niraven/myelin
- README: https://github.com/Niraven/myelin/blob/main/README.md
- License: MIT
