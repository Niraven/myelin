"""LongMemEval-S evaluation harness.

Provides BM25 (FTS5) and full-context substring-match evaluators
for the synthetic Myelin benchmark dataset.
"""

from __future__ import annotations

import json
import time
import sqlite3
import tempfile
import os
from typing import Any

from myelin.core.database import Database
from .dataset import LongMemEvalDataset


def _fts5_escape(query: str) -> str:
    """Escape user query text for FTS5 MATCH (OR-based)."""
    tokens: list[str] = []
    for token in query.split():
        cleaned = "".join(c for c in token if c.isalnum() or c in "._/@:+")
        if len(cleaned) >= 3:
            escaped = cleaned.replace('"', '""')
            tokens.append(f'"{escaped}"')
    if not tokens:
        # fallback: use whatever survived
        for token in query.split():
            cleaned = "".join(c for c in token if c.isalnum())
            if cleaned:
                tokens.append(f'"{cleaned}"')
    return " OR ".join(tokens) if tokens else query


def _make_episode_text(episode: dict) -> str:
    content = episode.get("content", "")
    tags = episode.get("tags", [])
    domain = episode.get("domain", "")
    tags_str = " ".join(t for t in tags if isinstance(t, str))
    return f"{content} {tags_str} {domain}"


def evaluate_bm25(
    questions: list[dict],
    episodes: list[dict],
    k: int = 1,
) -> dict[str, Any]:
    """Evaluate using BM25 via SQLite FTS5.

    Returns accuracy@k, MRR, and coverage.
    """
    if not questions or not episodes:
        return {"accuracy_at_1": 0.0, "mrr": 0.0, "coverage": 0.0, "total": 0}

    fd, db_path = tempfile.mkstemp(suffix=".eval.db")
    os.close(fd)

    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE VIRTUAL TABLE episodes_fts USING fts5(content, domain, tags, tokenize='porter unicode61')"
        )
        conn.execute("BEGIN")
        for ep in episodes:
            text = _make_episode_text(ep)
            domain = ep.get("domain", "")
            tags = " ".join(
                t for t in ep.get("tags", []) if isinstance(t, str)
            )
            conn.execute(
                "INSERT INTO episodes_fts(content, domain, tags) VALUES (?, ?, ?)",
                (text, domain, tags),
            )
        conn.execute("COMMIT")

        correct = 0
        reciprocal_ranks: list[float] = []
        covered = 0

        for q in questions:
            query_text = q.get("question", "")
            answer = q.get("answer", "")
            domain = q.get("domain", "")

            fts_query = _fts5_escape(query_text)
            if domain:
                fts_query = f"({fts_query}) AND domain:{domain}"

            try:
                rows = conn.execute(
                    "SELECT content, domain FROM episodes_fts WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, k),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

            best_rank = None
            for rank, row in enumerate(rows):
                row_text = row[0]
                if str(answer).lower() in row_text.lower():
                    if best_rank is None:
                        best_rank = rank + 1
                    break

            if best_rank is not None:
                if best_rank <= k:
                    correct += 1
                reciprocal_ranks.append(1.0 / best_rank)
                covered += 1
            else:
                reciprocal_ranks.append(0.0)

        n = len(questions)
        return {
            "accuracy_at_1": round(correct / max(n, 1), 4),
            "mrr": round(
                sum(reciprocal_ranks) / max(n, 1), 4
            ),
            "coverage": round(covered / max(n, 1), 4),
            "total": n,
        }
    finally:
        if conn:
            conn.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def evaluate_full_context(
    questions: list[dict],
    episodes: list[dict],
    k: int = 1,
) -> dict[str, Any]:
    """Full-context upper bound: scan all episode texts for answer substrings.

    This represents the theoretical maximum for a perfect retrieval model
    that can search the entire episode corpus exhaustively.
    """
    if not questions or not episodes:
        return {"accuracy_at_1": 0.0, "mrr": 0.0, "coverage": 0.0, "total": 0}

    episode_texts = [_make_episode_text(ep).lower() for ep in episodes]

    correct = 0
    reciprocal_ranks: list[float] = []
    covered = 0

    for q in questions:
        answer = q.get("answer", "")
        answer_lower = str(answer).lower()

        best_rank = None
        for rank_idx, text in enumerate(episode_texts):
            if answer_lower in text:
                if best_rank is None:
                    best_rank = rank_idx + 1
                break

        if best_rank is not None:
            if best_rank <= k:
                correct += 1
            reciprocal_ranks.append(1.0 / best_rank)
            covered += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(questions)
    return {
        "accuracy_at_1": round(correct / max(n, 1), 4),
        "mrr": round(sum(reciprocal_ranks) / max(n, 1), 4),
        "coverage": round(covered / max(n, 1), 4),
        "total": n,
    }


def evaluate(
    questions: list[dict],
    episodes: list[dict],
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Run all evaluation strategies and return a JSON report.

    Args:
        questions: List of question dicts (must have 'question', 'answer', 'domain').
        episodes: List of episode dicts (must have 'content', 'domain', 'tags').
        strategies: List of strategy names to run (default: all).

    Returns:
        Dict with timing, per-strategy results, and metadata.
    """
    if strategies is None:
        strategies = ["bm25", "full_context"]

    started = time.perf_counter()
    results: dict[str, dict] = {}

    if "bm25" in strategies:
        results["bm25"] = evaluate_bm25(questions, episodes)

    if "full_context" in strategies:
        results["full_context"] = evaluate_full_context(questions, episodes)

    elapsed = time.perf_counter() - started

    return {
        "elapsed_seconds": round(elapsed, 3),
        "strategies": results,
        "question_count": len(questions),
        "episode_count": len(episodes),
    }
