<p align="center">
  <img src="assets/brand/social-preview.png" alt="Myelin — procedural learning for AI agents" width="920">
</p>

<p align="center">
  <a href="https://github.com/Niraven/myelin/actions/workflows/ci.yml"><img src="https://github.com/Niraven/myelin/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/transport-stdio%20MCP-0F766E" alt="stdio MCP">
  <img src="https://img.shields.io/badge/storage-SQLite-1D4ED8" alt="SQLite">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-F8D36A" alt="MIT license"></a>
</p>

<p align="center">
  <strong>mem0 remembers. Myelin learns.</strong>
</p>

Agents should not relearn the same workflow twice. Myelin learns from observations agents emit, turns repeated work into reusable procedures, and exposes them over stdio MCP — local-first, in SQLite, with core learning that runs without an LLM.

## Why Myelin

Most agent memory systems retrieve text. Myelin learns how work gets done. Repeated actions are clustered, aligned into a consensus sequence, and promoted into an executable procedure that can be reused after it earns trust from verified outcomes.

- **Procedures, not just recall.** Repeated action sequences become executable workflows with Bayesian confidence tracking.
- **Verified trust.** Feedback bound to a `prediction_id` promotes a procedure from `candidate` to `validated`. Legacy feedback still updates confidence but cannot promote trust.
- **Context, not search results.** A context block fuses memories, matching procedures, entities, temporal state, and domain confidence.
- **Cross-agent transfer.** Learned procedures package intent and capability gaps so another agent can adapt and re-verify them.
- **Local-first.** SQLite + FTS5 on disk, stdio MCP in, no remote service. Core learning needs no LLM; embeddings and LLM consolidation are optional.

Named after the myelin sheath that accelerates neural signal transmission, Myelin accelerates agents by helping them reuse what already works.

## Install from source

The package is not yet published to PyPI. Install from a checkout:

```bash
git clone https://github.com/Niraven/myelin.git
cd myelin
pip install -e ".[dev]"
```

Runtime target: Python 3.11+.

## Try it

Run the proof demo. It observes five repeated deployment workflows, promotes the shared sequence into a procedure, executes it, and updates confidence from feedback:

```bash
python examples/procedure_learning_demo.py
```

Expected shape:

```text
Myelin procedure-learning demo
Episodes observed: 25
Procedures created: 1
Learned procedure: auto_git_npm_docker
Initial confidence: 75%
Verified feedback loop (3 × execute → bound feedback):
  evidence_quality: verified
  stored trust_state: trusted
Confidence after verified feedback: 85%
Same-domain context includes procedure: auto_git_npm_docker
```

For an orchestrated agent system, run the Hermes simulation:

```bash
python examples/hermes_procedure_demo.py
```

It simulates Hermes coordinating research, build, and release agents while Myelin learns the shared workflow. Hermes stays responsible for routing and approvals.

## Add Myelin as an MCP server

Myelin is a stdio MCP server — `python -m myelin.server`. Point an MCP client at it:

```json
{
  "mcpServers": {
    "myelin": {
      "command": "python",
      "args": ["-m", "myelin.server", "--embedding-model", "none"],
      "env": {}
    }
  }
}
```

It is stdio only today. Do not configure it as an HTTP URL unless you run it behind a separate MCP bridge.

## The learning loop

Agents and orchestrators call Myelin explicitly. It does not watch shells, browsers, or tool logs. Emit observations, ask for context, and run the loop:

```text
agent plans task
  -> myelin_context at task boundary
  -> myelin_observe_batch during the workflow
  -> myelin_sleep at session end or maintenance
  -> myelin_execute_procedure on a repeated task
  -> myelin_procedure_feedback after execution
```

`myelin_execute_procedure` returns a `prediction_id`. Pass it to `myelin_procedure_feedback` so the result is bound to that prediction: bound feedback is verified, idempotent, and atomic, and it can promote trust. Omit it and feedback still updates confidence, but it cannot promote trust.

Start with a small allowlist — `myelin_context`, `myelin_observe`, `myelin_observe_batch`, `myelin_execute_procedure`, `myelin_procedure_feedback`, `myelin_status`, `myelin_sleep` — and gate transfer, graph, teach, and profile tools until the integration is trusted.

## Key MCP tools

The server exports 25 tools. The ones you will reach for first:

| Tool | Purpose |
|------|---------|
| `myelin_observe` / `myelin_observe_batch` | Record one or many agent actions with entity extraction |
| `myelin_context` | Assemble context for a situation (primary tool) |
| `myelin_query` | Multi-signal retrieval across all memory types |
| `myelin_execute_procedure` | Find the best matching learned procedure |
| `myelin_procedure_feedback` | Report success/failure; bound via `prediction_id` |
| `myelin_sleep` | Run sleep consolidation and procedure promotion |
| `myelin_status` | System status overview |
| `myelin_teach` | Manually teach a procedure |
| `myelin_transfer_export` / `_import` / `_discover` | Package, load, and discover transferable procedures |

The full surface also covers facts, temporal state, graph queries, entity status, confidence, profile, update, forget, recall, and visualization.

## Architecture

```
                    MCP Interface (25 tools)
                           |
            +--------------+--------------+
            |              |              |
     Intelligence    Memory Layer    Knowledge Layer
     (context        (episodic,      (entities,
      assembly)       semantic,       graph,
                      procedural)     temporal)
            |              |              |
            +--------------+--------------+
                           |
                  Cognitive Processes
           (consolidation, reflection,
            promotion, composition,
            decay, sleep)
                           |
                    SQLite + FTS5
```

- **Memory Layer:** episodic (raw observations), semantic (distilled facts), procedural (learned workflows).
- **Knowledge Layer:** entity extraction, an evidence-weighted knowledge graph, and temporal state transitions — no LLM required for entity extraction.
- **Intelligence Layer:** the context assembler fuses all signals; the multi-signal retriever ranks results.
- **Cognitive Processes:** reconsolidation, reflection, promotion, composition, decay, and sleep-style consolidation. In the MCP server these run when the caller invokes `myelin_sleep` or maintenance; they are not an automatic background daemon.
- **Transfer Protocol:** capability-aware procedure packaging with confidence discounting across agents.

## How promotion works

```text
Agent actions -> Episodes -> Cluster detection -> Sequence alignment
  -> Consensus extraction -> Procedure creation -> Bayesian validation
  -> Active procedure
```

Confidence updates are Bayesian: success raises it, failure lowers it, bounded to [0.05, 0.99].

## Integrations

- [Agent integration guide](docs/integrations/README.md) — universal stdio MCP setup and trust bands
- [Hermes](docs/integrations/hermes.md) — flagship orchestrator integration
- [Codex](docs/integrations/codex.md)
- [Claude Code](docs/integrations/claude-code.md)
- [OpenClaw](docs/integrations/openclaw.md)
- [Generic MCP clients](docs/integrations/generic-mcp.md)

Backfill Hermes session history into Myelin (checkout-specific, Hermes only): `python scripts/myelin-backfill.py [--limit 1000] [--dry-run]`.

## Benchmark

Run a local benchmark:

```bash
python -m myelin.benchmark --counts 1000 --json
```

The full run uses multiple counts: `python -m myelin.benchmark --counts 1000,10000,50000 --json`. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for modes and how to read the results.

## Research foundation

Myelin adapts ideas from established cognitive architectures: ACT-R activation equations, SOAR chunking, Stanford Generative Agents reflection, CoALA memory structure, and ClustalW multiple sequence alignment.

## Development

```bash
pip install -e ".[dev]"
ruff format --check src/ tests/ examples/
ruff check src/ tests/ examples/
mypy src/myelin/ --ignore-missing-imports
pytest tests/ -q
python examples/procedure_learning_demo.py
python examples/hermes_procedure_demo.py
```

## Community

- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Brand guide: [docs/BRAND.md](docs/BRAND.md)
- Launch copy: [docs/LAUNCH_KIT.md](docs/LAUNCH_KIT.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## License

MIT
