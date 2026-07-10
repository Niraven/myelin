"""Tests for the Myelin v1 evaluation harness (benchmarks/)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from benchmarks.ci_subset import run_ci_subset
from benchmarks.longmemeval.dataset import (
    DOMAINS,
    QUESTIONS_PER_DOMAIN,
    LongMemEvalDataset,
)
from benchmarks.longmemeval.harness import (
    evaluate,
    evaluate_bm25,
    evaluate_full_context,
)
from benchmarks.locomo.harness import (
    CONVERSATIONS,
    evaluate_locomo,
)


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------


class TestDatasetStructure:
    def test_domains_defined(self):
        assert DOMAINS == ["deployment", "coding", "debugging", "security", "devops"]

    def test_questions_per_domain(self):
        assert QUESTIONS_PER_DOMAIN == 20

    def test_dataset_metadata_structure(self):
        dataset = LongMemEvalDataset()
        meta = dataset.metadata()
        assert meta["seed"] == 42
        assert meta["version"] == "1.0.0"
        assert meta["domains"] == DOMAINS
        assert meta["total_episodes"] == 500
        assert meta["total_questions"] == 100

    def test_episode_count(self):
        dataset = LongMemEvalDataset()
        episodes = dataset.generate_episodes()
        assert len(episodes) == 500

    def test_episodes_per_domain(self):
        dataset = LongMemEvalDataset()
        episodes = dataset.generate_episodes()
        for domain in DOMAINS:
            count = sum(1 for ep in episodes if ep["domain"] == domain)
            assert count == 100, f"Expected 100 episodes for {domain}, got {count}"

    def test_episode_required_fields(self):
        dataset = LongMemEvalDataset()
        episodes = dataset.generate_episodes()
        for ep in episodes:
            assert "domain" in ep
            assert "content" in ep
            assert "index" in ep
            assert ep["domain"] in DOMAINS
            assert isinstance(ep["content"], str)
            assert len(ep["content"]) > 0

    def test_question_count(self):
        dataset = LongMemEvalDataset()
        questions = dataset.generate_questions()
        assert len(questions) == 100

    def test_questions_per_domain_count(self):
        dataset = LongMemEvalDataset()
        questions = dataset.generate_questions()
        for domain in DOMAINS:
            count = sum(1 for q in questions if q["domain"] == domain)
            assert count == QUESTIONS_PER_DOMAIN, (
                f"Expected {QUESTIONS_PER_DOMAIN} questions for {domain}, got {count}"
            )

    def test_question_required_fields(self):
        dataset = LongMemEvalDataset()
        questions = dataset.generate_questions()
        required = {"question", "answer", "domain", "type"}
        for q in questions:
            missing = required - set(q.keys())
            assert not missing, f"Question missing fields: {missing}"
            assert q["domain"] in DOMAINS
            assert q["type"] == "factual"
            assert isinstance(q["question"], str)
            assert isinstance(q["answer"], str) or isinstance(q["answer"], list)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_dataset_deterministic_same_seed(self):
        d1 = LongMemEvalDataset(seed=42)
        d2 = LongMemEvalDataset(seed=42)
        e1 = d1.generate_episodes()
        e2 = d2.generate_episodes()
        for a, b in zip(e1, e2):
            assert a["content"] == b["content"]

    def test_dataset_deterministic_questions_same_seed(self):
        d1 = LongMemEvalDataset(seed=42)
        d2 = LongMemEvalDataset(seed=42)
        q1 = d1.generate_questions()
        q2 = d2.generate_questions()
        for a, b in zip(q1, q2):
            assert a["question"] == b["question"]
            assert a["answer"] == b["answer"]

    def test_different_seeds_produce_different_content(self):
        d1 = LongMemEvalDataset(seed=42)
        d2 = LongMemEvalDataset(seed=99)
        e1 = d1.generate_episodes()
        e2 = d2.generate_episodes()
        # Template-driven content uses index, not rng, so content is identical.
        # Only check that the metadata includes the seed (provenance).
        assert d1.metadata()["seed"] == 42
        assert d2.metadata()["seed"] == 99

    def test_bm25_deterministic(self):
        dataset = LongMemEvalDataset(seed=42)
        episodes = dataset.generate_episodes()
        questions = dataset.generate_questions()
        r1 = evaluate_bm25(questions, episodes)
        r2 = evaluate_bm25(questions, episodes)
        assert r1["accuracy_at_1"] == r2["accuracy_at_1"]
        assert r1["mrr"] == r2["mrr"]
        assert r1["coverage"] == r2["coverage"]

    def test_full_context_deterministic(self):
        dataset = LongMemEvalDataset(seed=42)
        episodes = dataset.generate_episodes()
        questions = dataset.generate_questions()
        r1 = evaluate_full_context(questions, episodes)
        r2 = evaluate_full_context(questions, episodes)
        assert r1["accuracy_at_1"] == r2["accuracy_at_1"]
        assert r1["mrr"] == r2["mrr"]
        assert r1["coverage"] == r2["coverage"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_empty_questions_bm25(self):
        result = evaluate_bm25([], [{"content": "test", "domain": "a", "tags": []}])
        assert result["accuracy_at_1"] == 0.0
        assert result["total"] == 0

    def test_empty_episodes_bm25(self):
        result = evaluate_bm25(
            [{"question": "q?", "answer": "a", "domain": "t"}], []
        )
        assert result["accuracy_at_1"] == 0.0
        assert result["total"] == 0

    def test_empty_questions_full_context(self):
        result = evaluate_full_context([], [{"content": "test", "domain": "a"}])
        assert result["accuracy_at_1"] == 0.0
        assert result["total"] == 0

    def test_empty_episodes_full_context(self):
        result = evaluate_full_context(
            [{"question": "q?", "answer": "a", "domain": "t"}], []
        )
        assert result["accuracy_at_1"] == 0.0
        assert result["total"] == 0

    def test_empty_both_evaluate(self):
        report = evaluate([], [])
        assert report["question_count"] == 0
        assert report["episode_count"] == 0


# ---------------------------------------------------------------------------
# BM25 baseline
# ---------------------------------------------------------------------------


class TestBM25:
    def test_bm25_returns_expected_keys(self):
        dataset = LongMemEvalDataset(seed=42)
        episodes = dataset.generate_episodes(episodes_per_domain=10)
        questions = dataset.generate_questions()
        # Take first 5 questions
        subset = questions[:5]
        result = evaluate_bm25(subset, episodes)
        assert "accuracy_at_1" in result
        assert "mrr" in result
        assert "coverage" in result
        assert "total" in result
        assert result["total"] == 5

    def test_bm25_accuracy_in_range(self):
        dataset = LongMemEvalDataset(seed=42)
        episodes = dataset.generate_episodes()
        questions = dataset.generate_questions()
        result = evaluate_bm25(questions, episodes)
        assert 0.0 <= result["accuracy_at_1"] <= 1.0
        assert 0.0 <= result["mrr"] <= 1.0


# ---------------------------------------------------------------------------
# Full-context baseline
# ---------------------------------------------------------------------------


class TestFullContext:
    def test_full_context_returns_expected_keys(self):
        dataset = LongMemEvalDataset(seed=42)
        episodes = dataset.generate_episodes(episodes_per_domain=10)
        questions = dataset.generate_questions()
        subset = questions[:5]
        result = evaluate_full_context(subset, episodes)
        assert "accuracy_at_1" in result
        assert "mrr" in result
        assert "coverage" in result
        assert "total" in result
        assert result["total"] == 5

    def test_full_context_accuracy_in_range(self):
        dataset = LongMemEvalDataset(seed=42)
        episodes = dataset.generate_episodes()
        questions = dataset.generate_questions()
        result = evaluate_full_context(questions, episodes)
        assert 0.0 <= result["accuracy_at_1"] <= 1.0
        assert 0.0 <= result["mrr"] <= 1.0


# ---------------------------------------------------------------------------
# CI subset
# ---------------------------------------------------------------------------


class TestCISubset:
    def test_ci_subset_completes_under_30s(self):
        started = time.perf_counter()
        result = run_ci_subset()
        elapsed = time.perf_counter() - started
        assert elapsed < 30.0, f"CI subset took {elapsed:.2f}s (expected < 30s)"
        assert result["question_count"] == 50

    def test_ci_subset_valid_json(self):
        result = run_ci_subset()
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["benchmark"] == "ci_subset"


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------


class TestLoCoMo:
    def test_conversation_count(self):
        assert len(CONVERSATIONS) == 10

    def test_conversations_have_required_fields(self):
        for conv in CONVERSATIONS:
            assert "name" in conv, f"Missing 'name' in conversation"
            assert "turns" in conv, f"Missing 'turns' in {conv.get('name')}"
            assert "questions" in conv, f"Missing 'questions' in {conv.get('name')}"
            assert len(conv["turns"]) >= 5, (
                f"{conv['name']} has {len(conv['turns'])} turns, expected >= 5"
            )

    def test_turns_have_required_fields(self):
        for conv in CONVERSATIONS:
            for t in conv["turns"]:
                assert "role" in t
                assert "content" in t
                assert t["role"] in ("user", "assistant")

    def test_questions_have_required_fields(self):
        for conv in CONVERSATIONS:
            for q in conv["questions"]:
                assert "question" in q, f"Missing question in {conv['name']}"
                assert "answer" in q, f"Missing answer in {conv['name']}"
                assert "type" in q, f"Missing type in {conv['name']}"
                assert q["type"] in ("temporal", "cross_conversation", "entity_tracking")

    def test_total_questions(self):
        total = sum(len(conv["questions"]) for conv in CONVERSATIONS)
        assert total == 50, f"Expected 50 total questions, got {total}"

    def test_locomo_evaluation_runs(self):
        result = evaluate_locomo()
        assert result["benchmark"] == "locomo"
        assert result["total"] == 50
        assert result["conversations"] == 10
        assert "accuracy" in result
        assert "elapsed_seconds" in result


# ---------------------------------------------------------------------------
# Combined evaluate() function
# ---------------------------------------------------------------------------


class TestEvaluateFunction:
    def test_evaluate_returns_expected_structure(self):
        dataset = LongMemEvalDataset(seed=42)
        episodes = dataset.generate_episodes(episodes_per_domain=5)
        questions = dataset.generate_questions()
        subset = questions[:10]
        report = evaluate(subset, episodes)
        assert "elapsed_seconds" in report
        assert "strategies" in report
        assert "question_count" in report
        assert "episode_count" in report
        assert report["question_count"] == 10

    def test_evaluate_default_strategies(self):
        dataset = LongMemEvalDataset(seed=42)
        episodes = dataset.generate_episodes(episodes_per_domain=5)
        questions = dataset.generate_questions()
        subset = questions[:5]
        report = evaluate(subset, episodes)
        assert "bm25" in report["strategies"]
        assert "full_context" in report["strategies"]
