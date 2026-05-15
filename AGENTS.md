# Myelin Agent Rules

## Project Identity

Myelin is a local-first procedural learning layer for AI agents. It is not a generic memory provider and it is not an orchestrator. The core claim is:

> mem0 remembers. Myelin learns.

Use this framing in docs, launch copy, and integration guidance.

## Current Technical Truth

- Python package: `myelin-memory`
- Runtime target: Python 3.11+
- Current transport: stdio MCP via `python -m myelin.server`
- Current tool count: 21 MCP tools
- Default mode: local SQLite + FTS5 with `--embedding-model none`
- Optional mode: local/API embeddings for semantic recall
- Core learning loop: observe -> cluster/align -> promote -> execute -> feedback
- Cross-agent transfer: capability-aware procedure packaging with confidence discounting

Do not claim that Myelin exposes a built-in HTTP, SSE, Streamable HTTP, or fixed-port MCP server until that transport exists in code.

## Engineering Standards

- Keep changes focused and behavior-preserving unless a task explicitly asks for product changes.
- Prefer the existing SQLite, MCP, Pydantic, and Python standard-library patterns.
- Use `apply_patch` for manual edits.
- Add tests when changing behavior, migrations, schema, tool contracts, or cognitive processes.
- Do not store or print secrets in examples, observations, tests, or docs.

## Validation

Run the narrowest useful checks first, then the full suite before release-oriented changes:

```bash
ruff format --check src/ tests/ examples/
ruff check src/ tests/ examples/
mypy src/myelin/ --ignore-missing-imports
pytest tests/ -q
python examples/procedure_learning_demo.py
python examples/hermes_procedure_demo.py
python -m myelin.benchmark --counts 100 --json
```

The known-good local command style is:

```bash
uv run --python /Users/niamamor/.local/bin/python3.11 --with-editable ".[dev]" <command>
```

## Documentation Rules

- Lead with procedural learning and agent acceleration, not generic recall.
- Explain that agents and orchestrators must explicitly emit observations.
- Use `--embedding-model none` in new docs.
- Mention Hermes as the flagship integration path, not the product boundary.
- Avoid live competitor statistics unless freshly verified.
- Keep benchmark claims tied to committed benchmark output.

## Launch Assets

- Primary README image: `assets/brand/social-preview.png`
- Brand guide: `docs/BRAND.md`
- Launch copy: `docs/LAUNCH_KIT.md`
- HyperFrames launch composition source: `assets/hyperframes/myelin-launch/`

