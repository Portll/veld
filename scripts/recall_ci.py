#!/usr/bin/env python3
"""95% percentile-bootstrap intervals for one recall-eval run, overall and per category.

recall-eval reports means. A mean over 100 or 1,332 cases moves by whole cases, and two runs that
differ by less than the interval width do not differ in a way the suite can show. Report a held-out
number with its interval.

Usage:
    python3 scripts/recall_ci.py <run.per-case.json> [--layer full] [--resamples 10000] [--seed 7]

Prints a markdown table. Stdlib only, deterministic for a seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

METRICS = (
    ("p_at_1", "p@1"),
    ("mrr", "MRR"),
    ("recall_at_k", "recall@10"),
    ("ndcg_at_k", "nDCG@10"),
)


def interval(values: list[float], resamples: int, rng: random.Random) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return mean, means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("per_case", type=Path)
    parser.add_argument("--layer", default="full")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    report = json.loads(args.per_case.read_text(encoding="utf-8"))
    if args.layer not in report:
        print(f"no layer {args.layer!r}; have {sorted(report)}", file=sys.stderr)
        return 2
    cases = report[args.layer]
    groups: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        groups["overall"].append(case)
        groups[case["category"]].append(case)

    header = "| group | n | " + " | ".join(label for _, label in METRICS) + " |"
    print(header)
    print("|" + "---|" * (2 + len(METRICS)))
    for name in ["overall"] + sorted(k for k in groups if k != "overall"):
        rows = groups[name]
        cells = []
        for key, _ in METRICS:
            rng = random.Random(f"{args.seed}:{name}:{key}")
            mean, lo, hi = interval([float(c[key]) for c in rows], args.resamples, rng)
            cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"| {name} | {len(rows)} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
