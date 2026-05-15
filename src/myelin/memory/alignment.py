"""ClustalW-inspired progressive multiple sequence alignment for procedure extraction.

In bioinformatics, ClustalW aligns multiple DNA/protein sequences to find conserved regions.
We adapt this for action sequences: find the conserved steps across multiple sessions
where an agent performed a similar workflow.

The key insight: just like conserved DNA regions indicate functional importance,
conserved action sequences indicate procedural steps the agent consistently follows.

Algorithm:
1. Pairwise alignment (Needleman-Wunsch) of all session action sequences
2. Build guide tree from pairwise distances (UPGMA)
3. Progressive alignment following the guide tree
4. Extract consensus: CORE (>80%), OPTIONAL (40-80%), VARIANT (multiple actions at same position)
"""

from __future__ import annotations

from collections import Counter

GAP = "-"

# ── Scoring ────────────────────────────────────────────────────


def action_match_score(a: str, b: str) -> float:
    """Score for matching two actions. Uses token overlap for fuzzy matching."""
    if a == b:
        return 2.0
    if a == GAP or b == GAP:
        return -1.0

    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return -1.0

    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    if jaccard > 0.5:
        return 1.0
    elif jaccard > 0.25:
        return 0.0
    return -1.0


# ── Pairwise Alignment (Needleman-Wunsch) ─────────────────────


def needleman_wunsch(
    seq_a: list[str],
    seq_b: list[str],
    gap_penalty: float = -1.0,
) -> tuple[list[str], list[str], float]:
    """Global alignment of two action sequences.

    Returns (aligned_a, aligned_b, score).
    """
    m, n = len(seq_a), len(seq_b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i * gap_penalty
    for j in range(1, n + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + action_match_score(seq_a[i - 1], seq_b[j - 1])
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(match, delete, insert)

    # Traceback
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    i, j = m, n

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + action_match_score(seq_a[i - 1], seq_b[j - 1])
        ):
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap_penalty:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(GAP)
            i -= 1
        else:
            aligned_a.append(GAP)
            aligned_b.append(seq_b[j - 1])
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    return aligned_a, aligned_b, dp[m][n]


# ── Distance Matrix ───────────────────────────────────────────


def pairwise_distances(sequences: list[list[str]]) -> list[list[float]]:
    """Compute pairwise distance matrix from alignment scores."""
    n = len(sequences)
    scores = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            _, _, score = needleman_wunsch(sequences[i], sequences[j])
            max_len = max(len(sequences[i]), len(sequences[j]))
            normalized = score / (2.0 * max_len) if max_len > 0 else 0.0
            distance = 1.0 - max(0.0, min(1.0, normalized))
            scores[i][j] = distance
            scores[j][i] = distance

    return scores


# ── Guide Tree (UPGMA) ────────────────────────────────────────


def upgma_tree(distances: list[list[float]]) -> list[tuple[int | tuple, int | tuple]]:
    """Build a UPGMA guide tree from a distance matrix.

    Returns merge order as list of (cluster_a, cluster_b) pairs.
    """
    n = len(distances)
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    dist = {(i, j): distances[i][j] for i in range(n) for j in range(i + 1, n)}
    active = set(range(n))
    merges: list[tuple] = []
    next_id = n

    while len(active) > 1:
        # Find closest pair
        best_dist = float("inf")
        best_pair = (-1, -1)
        active_list = sorted(active)

        for idx, ci in enumerate(active_list):
            for cj in active_list[idx + 1 :]:
                key = (min(ci, cj), max(ci, cj))
                d = dist.get(key, float("inf"))
                if d < best_dist:
                    best_dist = d
                    best_pair = (ci, cj)

        ci, cj = best_pair
        merges.append((ci, cj))

        # Create new cluster
        new_cluster = clusters[ci] + clusters[cj]
        clusters[next_id] = new_cluster

        # Update distances (average linkage)
        for ck in active:
            if ck in (ci, cj):
                continue
            d_ci = dist.get((min(ci, ck), max(ci, ck)), 0.0)
            d_cj = dist.get((min(cj, ck), max(cj, ck)), 0.0)
            ni, nj = len(clusters[ci]), len(clusters[cj])
            new_dist = (d_ci * ni + d_cj * nj) / (ni + nj)
            dist[(min(next_id, ck), max(next_id, ck))] = new_dist

        active.remove(ci)
        active.remove(cj)
        active.add(next_id)
        next_id += 1

    return merges


# ── Progressive Alignment ─────────────────────────────────────


def align_profile_to_profile(
    profile_a: list[list[str]],
    profile_b: list[list[str]],
) -> list[list[str]]:
    """Align two profiles (groups of aligned sequences).

    Uses consensus of each profile for the alignment, then maps gaps back.
    """
    consensus_a = _profile_consensus(profile_a)
    consensus_b = _profile_consensus(profile_b)

    aligned_ca, aligned_cb, _ = needleman_wunsch(consensus_a, consensus_b)

    # Map alignment back to all sequences in each profile
    result_a = _apply_alignment_to_profile(profile_a, consensus_a, aligned_ca)
    result_b = _apply_alignment_to_profile(profile_b, consensus_b, aligned_cb)

    return result_a + result_b


def _profile_consensus(profile: list[list[str]]) -> list[str]:
    """Extract consensus sequence from a profile (most common action at each position)."""
    if not profile:
        return []
    width = len(profile[0])
    consensus = []
    for col in range(width):
        counts: Counter[str] = Counter()
        for row in profile:
            if col < len(row) and row[col] != GAP:
                counts[row[col]] += 1
        if counts:
            consensus.append(counts.most_common(1)[0][0])
        else:
            consensus.append(GAP)
    return [c for c in consensus if c != GAP]


def _apply_alignment_to_profile(
    profile: list[list[str]],
    original_consensus: list[str],
    aligned_consensus: list[str],
) -> list[list[str]]:
    """Map gaps from aligned consensus back to all sequences in the profile."""
    result = []
    for seq in profile:
        new_seq: list[str] = []
        seq_idx = 0

        for ac in aligned_consensus:
            if ac == GAP:
                new_seq.append(GAP)
            else:
                # Find corresponding position in original sequence
                while seq_idx < len(seq) and seq[seq_idx] == GAP:
                    new_seq.append(GAP)
                    seq_idx += 1
                if seq_idx < len(seq):
                    new_seq.append(seq[seq_idx])
                    seq_idx += 1
                else:
                    new_seq.append(GAP)

        result.append(new_seq)
    return result


def progressive_align(sequences: list[list[str]]) -> list[list[str]]:
    """ClustalW-style progressive multiple sequence alignment.

    1. Compute pairwise distances
    2. Build UPGMA guide tree
    3. Align progressively following the tree
    """
    n = len(sequences)
    if n == 0:
        return []
    if n == 1:
        return [sequences[0]]
    if n == 2:
        a, b, _ = needleman_wunsch(sequences[0], sequences[1])
        return [a, b]

    distances = pairwise_distances(sequences)
    merges = upgma_tree(distances)

    # Build profiles following merge order
    profiles: dict[int, list[list[str]]] = {i: [seq] for i, seq in enumerate(sequences)}

    next_id = n
    for ci, cj in merges:
        if ci in profiles and cj in profiles:
            merged = align_profile_to_profile(profiles[ci], profiles[cj])
            profiles[next_id] = merged
            del profiles[ci]
            del profiles[cj]
        next_id += 1

    # Return the final alignment, padded to uniform width
    if profiles:
        rows = list(profiles.values())[0]
        if rows:
            max_width = max(len(r) for r in rows)
            rows = [r + [GAP] * (max_width - len(r)) for r in rows]
        return rows
    return []


# ── Consensus Extraction ──────────────────────────────────────


class AlignedStep:
    """A step extracted from multiple sequence alignment."""

    def __init__(self, position: int, actions: dict[str, int], total_sequences: int):
        self.position = position
        self.actions = actions
        self.total = total_sequences
        self.primary_action = max(actions, key=lambda action: actions[action]) if actions else ""
        self.frequency = max(actions.values()) / total_sequences if actions else 0.0

    @property
    def step_type(self) -> str:
        if self.frequency > 0.8:
            return "core"
        elif self.frequency >= 0.4:
            return "optional"
        else:
            return "rare"

    @property
    def has_variants(self) -> bool:
        return len(self.actions) > 1

    @property
    def variants(self) -> list[str]:
        return [a for a in self.actions if a != self.primary_action]


def extract_consensus(
    alignment: list[list[str]],
    min_frequency: float = 0.3,
) -> list[AlignedStep]:
    """Extract consensus steps from a multiple alignment.

    Returns steps classified as CORE (>80%), OPTIONAL (40-80%), or RARE (<40%).
    Steps below min_frequency are dropped entirely.
    """
    if not alignment:
        return []

    n_sequences = len(alignment)
    width = max(len(seq) for seq in alignment) if alignment else 0
    steps: list[AlignedStep] = []

    for col in range(width):
        action_counts: Counter[str] = Counter()
        for seq in alignment:
            if col < len(seq) and seq[col] != GAP:
                action_counts[seq[col]] += 1

        if not action_counts:
            continue

        total_non_gap = sum(action_counts.values())
        frequency = total_non_gap / n_sequences

        if frequency >= min_frequency:
            step = AlignedStep(
                position=len(steps),
                actions=dict(action_counts),
                total_sequences=n_sequences,
            )
            steps.append(step)

    return steps
