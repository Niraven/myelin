"""Test ClustalW-inspired sequence alignment."""

from myelin.memory.alignment import (
    AlignedStep,
    GAP,
    action_match_score,
    extract_consensus,
    needleman_wunsch,
    pairwise_distances,
    progressive_align,
)


class TestActionMatchScore:
    def test_exact_match(self):
        assert action_match_score("git pull", "git pull") == 2.0

    def test_gap(self):
        assert action_match_score("git pull", GAP) == -1.0

    def test_partial_match(self):
        score = action_match_score("npm run test", "npm run build")
        assert score > -1.0  # "npm" and "run" overlap

    def test_no_match(self):
        score = action_match_score("git pull", "docker compose up")
        assert score <= 0.0


class TestNeedlemanWunsch:
    def test_identical_sequences(self):
        seq = ["git pull", "npm test", "npm build"]
        aligned_a, aligned_b, score = needleman_wunsch(seq, seq)
        assert aligned_a == aligned_b
        assert score > 0

    def test_one_insertion(self):
        seq_a = ["git pull", "npm test", "npm build"]
        seq_b = ["git pull", "npm install", "npm test", "npm build"]
        aligned_a, aligned_b, _ = needleman_wunsch(seq_a, seq_b)
        assert len(aligned_a) == len(aligned_b)
        assert GAP in aligned_a or GAP in aligned_b

    def test_completely_different(self):
        seq_a = ["action_a", "action_b"]
        seq_b = ["action_x", "action_y"]
        aligned_a, aligned_b, score = needleman_wunsch(seq_a, seq_b)
        assert len(aligned_a) == len(aligned_b)

    def test_empty_sequence(self):
        aligned_a, aligned_b, _ = needleman_wunsch([], ["a", "b"])
        assert all(a == GAP for a in aligned_a)

    def test_single_element(self):
        aligned_a, aligned_b, _ = needleman_wunsch(["git pull"], ["git pull"])
        assert aligned_a == ["git pull"]
        assert aligned_b == ["git pull"]


class TestProgressiveAlign:
    def test_two_sequences(self):
        seqs = [
            ["git pull", "npm test", "npm build"],
            ["git pull", "npm test", "npm build"],
        ]
        result = progressive_align(seqs)
        assert len(result) == 2
        assert len(result[0]) == len(result[1])

    def test_three_similar_sequences(self):
        seqs = [
            ["git pull", "npm test", "npm build", "git push"],
            ["git pull", "npm install", "npm test", "npm build", "git push"],
            ["git pull", "npm test", "npm build", "notify slack", "git push"],
        ]
        result = progressive_align(seqs)
        assert len(result) == 3
        # All rows should be same length
        lengths = set(len(row) for row in result)
        assert len(lengths) == 1

    def test_single_sequence(self):
        seqs = [["git pull", "npm test"]]
        result = progressive_align(seqs)
        assert len(result) == 1

    def test_empty(self):
        assert progressive_align([]) == []


class TestExtractConsensus:
    def test_all_core_steps(self):
        alignment = [
            ["git pull", "npm test", "npm build"],
            ["git pull", "npm test", "npm build"],
            ["git pull", "npm test", "npm build"],
        ]
        consensus = extract_consensus(alignment)
        assert len(consensus) == 3
        assert all(s.step_type == "core" for s in consensus)

    def test_mixed_types(self):
        alignment = [
            ["git pull", "npm test", "npm build", "notify"],
            ["git pull", "npm test", "npm build", GAP],
            ["git pull", "npm test", "npm build", GAP],
            ["git pull", "npm test", "npm build", "notify"],
            ["git pull", "npm test", "npm build", GAP],
        ]
        consensus = extract_consensus(alignment)
        core = [s for s in consensus if s.step_type == "core"]
        optional = [s for s in consensus if s.step_type == "optional"]
        assert len(core) == 3  # git pull, npm test, npm build
        assert len(optional) == 1  # notify (2/5 = 40%)

    def test_variants(self):
        alignment = [
            ["git pull", "npm install", "npm test"],
            ["git pull", "npm ci", "npm test"],
            ["git pull", "npm install", "npm test"],
        ]
        consensus = extract_consensus(alignment)
        variant_step = next(
            (s for s in consensus if s.has_variants), None
        )
        assert variant_step is not None

    def test_empty_alignment(self):
        assert extract_consensus([]) == []

    def test_min_frequency_filter(self):
        alignment = [
            ["a", "b", "c"],
            ["a", GAP, "c"],
            ["a", GAP, "c"],
            ["a", GAP, "c"],
            ["a", GAP, "c"],
        ]
        # "b" appears 1/5 = 20%, below default min_frequency of 0.3
        consensus = extract_consensus(alignment, min_frequency=0.3)
        actions = [s.primary_action for s in consensus]
        assert "b" not in actions
