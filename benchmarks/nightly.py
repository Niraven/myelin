"""Full nightly evaluation: 500 episodes, 100 questions, all strategies."""

from __future__ import annotations

import json
import sys
import time

from .longmemeval.dataset import LongMemEvalDataset
from .longmemeval.harness import evaluate_bm25, evaluate_full_context, evaluate
from .locomo.harness import evaluate_locomo


def run_nightly() -> dict:
    """Run the full nightly evaluation suite."""
    started = time.perf_counter()

    # --- LongMemEval full ---
    dataset = LongMemEvalDataset(seed=42)
    episodes = dataset.generate_episodes(episodes_per_domain=100)
    questions = dataset.generate_questions()

    full_report = evaluate(questions, episodes)

    # --- LoCoMo ---
    locomo_report = evaluate_locomo()

    elapsed = time.perf_counter() - started

    return {
        "benchmark": "nightly",
        "elapsed_seconds": round(elapsed, 3),
        "episode_count": len(episodes),
        "question_count": len(questions),
        "longmemeval": full_report,
        "locomo": locomo_report,
    }


if __name__ == "__main__":
    result = run_nightly()
    print(json.dumps(result, indent=2))
