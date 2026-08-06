# Performance

Myelin optimizes for agent acceleration first and raw local latency second.

The launch claim should be:

> Myelin makes agents faster by letting them reuse learned procedures.

Do not publish speed comparisons against Mnemosyne, Noxem, Supermemory, Honcho, Hindsight, or mem0 without a fresh benchmark run on the same machine and workload.

## Modes

| Mode | Embeddings | Best for | Notes |
|---|---|---|---|
| Fast trace | `--embedding-model none` | Hermes/Codex tool events, CI repair, deployment traces, swarm action logs | Default launch mode. Uses SQLite + FTS5 only. |
| Semantic | `--embedding-model local` | Richer natural-language recall over long notes or summaries | Requires `pip install -e ".[embeddings]"`. |
| Hybrid | FTS first, embeddings where useful | Mixed procedural traces and semantic notes | Keep action traces no-embedding unless semantic recall quality matters. |

## Benchmark

`--counts` takes one or more comma-separated observation counts and defaults to `1000`. Add `--json` for machine-readable output.

Run a smoke benchmark:

```bash
python -m myelin.benchmark --counts 1000 --json
```

Run the full local benchmark:

```bash
python -m myelin.benchmark --counts 1000,10000,50000 --json
```

The benchmark reports:

- `store.p50_ms`, `store.p95_ms`
- `recall.p50_ms`, `recall.p95_ms`
- `context.p50_ms`, `context.p95_ms`
- `execute_procedure.p50_ms`, `execute_procedure.p95_ms`
- `promotion_ms`
- `procedure_hit_rate`
- `agent_steps_saved`

## Interpreting Results

Raw DB latency proves Myelin is not the bottleneck. Agent acceleration proves the product value.

Use both:

- **Raw latency:** local store/search/context timings.
- **Agent speed:** fewer planning, investigation, and repeated workflow steps after Myelin learns a procedure.

The benchmark uses no embeddings by default because action traces are structured and should not pay model latency on every observation.

## MCP Token Budget

Myelin should make agents faster, not add constant context overhead.

- Observe frequently, but return small observation acknowledgements.
- Batch workflow events with `myelin_observe_batch`.
- Call `myelin_context` at task boundaries, not every model turn.
- Keep `max_memories` and `max_procedures` low for coding agents until the database has useful signal.
- Store concise summaries in `content_text`; keep huge logs, artifacts, and raw outputs outside Myelin.
- Use `myelin_execute_procedure` when the task resembles a repeated workflow, then always send feedback.
