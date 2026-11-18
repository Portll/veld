#!/usr/bin/env python3
"""Paired comparison of two recall-eval runs over the same cases, with conversation clustering.

recall_ci.py puts an interval on one run. A change is a difference between two runs over the same
questions, so the question is whether that difference is larger than the questions alone would
produce. Two runs scored on the same cases are paired: each case's difference is the unit, and the
case-to-case spread that both runs share cancels.

LoCoMo's questions are not independent. Every question about one conversation is asked over the
same memories, so a change that helps one conversation's speakers moves many of its questions
together. Treating 1,000 questions over nine conversations as 1,000 independent draws makes an
interval too narrow. Each test is therefore reported twice:

- per case: the case is the resampling unit (paired bootstrap, sign-flip randomisation);
- per conversation: the conversation is the unit, every case in it moves together. With as few
  conversations as LoCoMo has, the randomisation test is exact (every sign pattern is enumerated).

The paired randomisation test and the bootstrap are the tests recommended for IR comparisons
(Smucker, Allan & Carterette, CIKM 2007); the sign and Wilcoxon tests are not used.

Usage:
    python3 scripts/recall_compare.py <baseline.per-case.json> <candidate.per-case.json>
        [--layer full] [--resamples 10000] [--seed 7] [--alpha 0.05]

Prints a markdown table. Stdlib only, deterministic for a seed. Exits 2 when the runs cannot be
paired (different layers or different case sets), 0 otherwise: this script reports, it never gates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

METRICS = (
    ("p_at_1", "p@1"),
    ("mrr", "MRR"),
    ("recall_at_k", "recall@10"),
    ("ndcg_at_k", "nDCG@10"),
)

# Above this many clusters the exact sign-flip enumeration (2^n patterns) gives way to sampling.
EXACT_CLUSTER_LIMIT = 16

# Below this many conversations a cluster bootstrap has too few distinct resamples to mean much.
FEW_CLUSTERS = 5


class PairingError(ValueError):
    """The two runs do not score the same cases, so no per-case difference exists."""


def conversation_of(case_id: str) -> str:
    """The conversation a case belongs to: `conv-42_q17` → `conv-42`.

    A case id without the `_q` question suffix is its own cluster, so a suite that is not
    LoCoMo-shaped degrades to the per-case analysis instead of lumping everything together.
    """
    head, sep, tail = case_id.rpartition("_q")
    if sep and head and tail.isdigit():
        return head
    return case_id


def load_cases(path: Path, layer: str) -> dict[str, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if layer not in report:
        raise PairingError(f"{path}: no layer {layer!r}; have {sorted(report)}")
    cases: dict[str, dict] = {}
    for case in report[layer]:
        case_id = case["case_id"]
        if case_id in cases:
            raise PairingError(f"{path}: case {case_id!r} appears twice")
        cases[case_id] = case
    return cases


def pair(baseline: dict[str, dict], candidate: dict[str, dict]) -> list[tuple[str, dict, dict]]:
    only_base = sorted(set(baseline) - set(candidate))
    only_cand = sorted(set(candidate) - set(baseline))
    if only_base or only_cand:
        raise PairingError(
            f"case sets differ: {len(only_base)} only in the baseline (first: {only_base[:3]}), "
            f"{len(only_cand)} only in the candidate (first: {only_cand[:3]})"
        )
    return [(case_id, baseline[case_id], candidate[case_id]) for case_id in sorted(baseline)]


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted list."""
    if not sorted_values:
        raise ValueError("quantile of nothing")
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


@dataclass(frozen=True)
class Comparison:
    mean_baseline: float
    mean_candidate: float
    delta: float
    ci_case: tuple[float, float]
    p_case: float
    ci_conversation: tuple[float, float] | None
    p_conversation: float | None
    exact_conversation: bool
    better: int
    worse: int


def bootstrap_case(diffs: list[float], resamples: int, rng: random.Random, alpha: float) -> tuple[float, float]:
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    return quantile(means, alpha / 2), quantile(means, 1 - alpha / 2)


def bootstrap_cluster(
    clusters: list[tuple[float, int]], resamples: int, rng: random.Random, alpha: float
) -> tuple[float, float]:
    """Resample whole conversations; the statistic is total difference over total cases.

    Conversations hold different numbers of questions, so the mean of a resample is the ratio of
    sums, not the mean of per-conversation means: a conversation with 200 questions keeps the
    weight it has in the full-suite mean.
    """
    k = len(clusters)
    means = []
    for _ in range(resamples):
        draw = rng.choices(clusters, k=k)
        total = sum(c for _, c in draw)
        means.append(sum(s for s, _ in draw) / total)
    means.sort()
    return quantile(means, alpha / 2), quantile(means, 1 - alpha / 2)


def randomisation_case(diffs: list[float], resamples: int, rng: random.Random) -> float:
    """Two-sided paired sign-flip test on the mean difference, with the +1 correction."""
    observed = abs(sum(diffs))
    if observed == 0:
        return 1.0
    hits = 0
    for _ in range(resamples):
        flipped = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(flipped) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (resamples + 1)


def randomisation_cluster(
    cluster_sums: list[float], resamples: int, rng: random.Random
) -> tuple[float, bool]:
    """Sign-flip test with each conversation's summed difference flipped as one unit.

    Exact over every sign pattern when there are few conversations, which is LoCoMo's case. The
    exact test's smallest attainable p-value is 2 / 2^k: with nine conversations, 0.0039.
    """
    observed = abs(sum(cluster_sums))
    if observed == 0:
        return 1.0, len(cluster_sums) <= EXACT_CLUSTER_LIMIT
    if len(cluster_sums) <= EXACT_CLUSTER_LIMIT:
        hits = 0
        total = 0
        for signs in itertools.product((1.0, -1.0), repeat=len(cluster_sums)):
            total += 1
            if abs(sum(s * v for s, v in zip(signs, cluster_sums))) >= observed - 1e-12:
                hits += 1
        return hits / total, True
    hits = 0
    for _ in range(resamples):
        flipped = sum(v if rng.random() < 0.5 else -v for v in cluster_sums)
        if abs(flipped) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (resamples + 1), False


def compare(
    pairs: list[tuple[str, dict, dict]], key: str, resamples: int, seed: str, alpha: float
) -> Comparison:
    base = [float(b[key]) for _, b, _ in pairs]
    cand = [float(c[key]) for _, _, c in pairs]
    diffs = [c - b for b, c in zip(base, cand)]
    n = len(diffs)

    by_conversation: dict[str, list[float]] = defaultdict(list)
    for (case_id, _, _), d in zip(pairs, diffs):
        by_conversation[conversation_of(case_id)].append(d)
    clusters = [(sum(v), len(v)) for _, v in sorted(by_conversation.items())]

    if len(clusters) >= 2:
        p_conv, exact = randomisation_cluster(
            [s for s, _ in clusters], resamples, random.Random(f"{seed}:{key}:rand-conv")
        )
        ci_conv = bootstrap_cluster(clusters, resamples, random.Random(f"{seed}:{key}:boot-conv"), alpha)
    else:
        # One conversation has no between-conversation spread to estimate; a "[+x, +x]" interval
        # would read as certainty.
        p_conv, exact, ci_conv = None, True, None
    return Comparison(
        mean_baseline=sum(base) / n,
        mean_candidate=sum(cand) / n,
        delta=sum(diffs) / n,
        ci_case=bootstrap_case(diffs, resamples, random.Random(f"{seed}:{key}:boot-case"), alpha),
        p_case=randomisation_case(diffs, resamples, random.Random(f"{seed}:{key}:rand-case")),
        ci_conversation=ci_conv,
        p_conversation=p_conv,
        exact_conversation=exact,
        better=sum(1 for d in diffs if d > 0),
        worse=sum(1 for d in diffs if d < 0),
    )


def render(pairs: list[tuple[str, dict, dict]], resamples: int, seed: str, alpha: float) -> str:
    conversations = sorted({conversation_of(case_id) for case_id, _, _ in pairs})
    level = round(100 * (1 - alpha))
    lines = [
        f"Paired over **{len(pairs)}** cases in **{len(conversations)}** conversation(s). "
        f"Intervals are {level}% percentile bootstrap of candidate − baseline; p-values are two-sided "
        "paired sign-flip randomisation.",
        "",
        f"| metric | baseline | candidate | Δ | {level}% CI per case | p per case | "
        f"{level}% CI per conversation | p per conversation | cases better / worse |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    exact_flags = set()
    for key, label in METRICS:
        if not all(key in b and key in c for _, b, c in pairs):
            continue
        r = compare(pairs, key, resamples, seed, alpha)
        exact_flags.add(r.exact_conversation)
        if r.ci_conversation is None or r.p_conversation is None:
            conv_ci, conv_p = "n/a", "n/a"
        else:
            conv_ci = f"[{r.ci_conversation[0]:+.4f}, {r.ci_conversation[1]:+.4f}]"
            conv_p = f"{r.p_conversation:.4f}"
        lines.append(
            f"| {label} | {r.mean_baseline:.4f} | {r.mean_candidate:.4f} | {r.delta:+.4f} | "
            f"[{r.ci_case[0]:+.4f}, {r.ci_case[1]:+.4f}] | {r.p_case:.4f} | "
            f"{conv_ci} | {conv_p} | {r.better} / {r.worse} |"
        )
    lines.append("")
    if len(conversations) < 2:
        lines.append(
            "One conversation: there is no between-conversation spread to estimate, so the "
            "per-case columns are all there is, and they speak for this conversation only."
        )
    elif exact_flags == {True}:
        lines.append(
            f"The per-conversation test is exact over all 2^{len(conversations)} sign patterns; "
            f"its smallest attainable p-value is {2 / 2 ** len(conversations):.4f}."
        )
    elif exact_flags == {False}:
        lines.append(f"The per-conversation test is sampled ({resamples} sign patterns).")
    if 2 <= len(conversations) < FEW_CLUSTERS:
        lines.append(
            f"Only {len(conversations)} conversations: the per-conversation bootstrap has few "
            "distinct resamples and its interval is coarse."
        )
    if len(conversations) >= 2:
        lines.append(
            "Read the per-conversation columns: questions about one conversation move together, so "
            "the per-case interval is too narrow for LoCoMo."
        )
    lines.append(
        f"Four metrics are tested at once; a p-value near {alpha} on one of them is weak evidence."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--layer", default="full")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", default="7")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie strictly between 0 and 1")
    if args.resamples < 1:
        parser.error("--resamples must be at least 1")

    try:
        pairs = pair(load_cases(args.baseline, args.layer), load_cases(args.candidate, args.layer))
    except PairingError as e:
        print(f"cannot pair the runs: {e}", file=sys.stderr)
        return 2
    if not pairs:
        print("cannot pair the runs: no cases", file=sys.stderr)
        return 2
    print(render(pairs, args.resamples, args.seed, args.alpha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
