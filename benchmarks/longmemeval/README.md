# LongMemEval-S: Synthetic Long-Term Memory Evaluation

A deterministic, repeatable synthetic benchmark for evaluating long-term memory retrieval in AI agents. Part of the Myelin v1 evaluation harness.

## Overview

- **5 domains**: deployment, coding, debugging, security, devops
- **500 episodes**: 100 per domain, generated deterministically with seed=42
- **100 questions**: 20 per domain, with exact answers found in specific episodes
- **2 evaluation strategies**:
  - **BM25** (FTS5): SQLite full-text search baseline using Porter tokenizer
  - **Full-context**: Exhaustive substring match across all episodes (theoretical upper bound)

## Usage

```python
from benchmarks.longmemeval.dataset import LongMemEvalDataset
from benchmarks.longmemeval.harness import evaluate

dataset = LongMemEvalDataset(seed=42)
episodes = dataset.generate_episodes(episodes_per_domain=100)
questions = dataset.generate_questions()

report = evaluate(questions, episodes)
print(report)
```

### CLI via benchmark module

```bash
python -m myelin.benchmark --longmemeval
python -m myelin.benchmark --ci-subset
python -m myelin.benchmark --nightly
python -m myelin.benchmark --longmemeval --json
```

## Output format

```json
{
  "elapsed_seconds": 0.345,
  "strategies": {
    "bm25": { "accuracy_at_1": 0.42, "mrr": 0.51, "coverage": 0.83, "total": 100 },
    "full_context": { "accuracy_at_1": 1.0, "mrr": 1.0, "coverage": 1.0, "total": 100 }
  },
  "question_count": 100,
  "episode_count": 500
}
```

## ⚠️ Disclaimer

**This is a synthetic approximation, not the official LongMemEval dataset.** The official LongMemEval (https://github.com/zou-group/LongMemEval) is a curated benchmark with real-world episodes and human-written questions. This synthetic version is designed for rapid iteration on retrieval strategies within the Myelin project. Results from this benchmark are indicative of relative improvement but should not be cited as LongMemEval scores.
