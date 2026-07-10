"""CI subset of LongMemEval — 50 questions, <30s runtime."""

from __future__ import annotations

import json
import sys
import time

from .longmemeval.dataset import LongMemEvalDataset
from .longmemeval.harness import evaluate_bm25, evaluate_full_context


def run_ci_subset() -> dict:
    """Run a compact CI subset: 10 questions per domain (50 total)."""
    started = time.perf_counter()

    dataset = LongMemEvalDataset(seed=42)
    episodes = dataset.generate_episodes(episodes_per_domain=100)
    questions = dataset.generate_questions()

    # Take first 10 questions per domain (50 total)
    from .longmemeval.dataset import DOMAINS, QUESTIONS_PER_DOMAIN

    questions_subset: list[dict] = []
    per_domain = QUESTIONS_PER_DOMAIN // 2  # 10 per domain
    for domain in DOMAINS:
        count = 0
        for q in questions:
            if q.get("domain") == domain:
                questions_subset.append(q)
                count += 1
                if count >= per_domain:
                    break

    bm25_result = evaluate_bm25(questions_subset, episodes)
    fc_result = evaluate_full_context(questions_subset, episodes)

    elapsed = time.perf_counter() - started

    return {
        "benchmark": "ci_subset",
        "elapsed_seconds": round(elapsed, 3),
        "question_count": len(questions_subset),
        "episode_count": len(episodes),
        "strategies": {
            "bm25": bm25_result,
            "full_context": fc_result,
        },
    }


if __name__ == "__main__":
    result = run_ci_subset()
    print(json.dumps(result, indent=2))
