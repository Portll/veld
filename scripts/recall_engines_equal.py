#!/usr/bin/env python3
"""Exit 0 when two recall-eval reports, one per storage engine, carry the same scores.

  recall_engines_equal.py <rocksdb report.json> <redb report.json>

The storage engine holds the memories and the graph; it must not move a ranking. Every metric in
`layers` and `by_category` is compared, and any difference is printed and fails the run.
"""
import json
import sys
from pathlib import Path

EPSILON = 1e-12


def metrics(report):
    out = {}
    for layer, values in (report.get("layers") or {}).items():
        for name, value in (values or {}).items():
            if isinstance(value, (int, float)):
                out[f"layers.{layer}.{name}"] = value
    for category, values in (report.get("by_category") or {}).items():
        for name, value in (values or {}).items():
            if isinstance(value, (int, float)):
                out[f"by_category.{category}.{name}"] = value
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    left_path, right_path = map(Path, sys.argv[1:])
    left, right = (json.loads(p.read_text()) for p in (left_path, right_path))
    if left.get("embedder") != right.get("embedder"):
        print(f"embedders differ: {left.get('embedder')} vs {right.get('embedder')}")
        return 2
    a, b = metrics(left), metrics(right)
    if not a:
        print(f"{left_path} carries no metrics")
        return 2
    differing = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            differing.append(f"{key}: present in only one report")
        elif abs(a[key] - b[key]) > EPSILON:
            differing.append(f"{key}: {a[key]!r} vs {b[key]!r}")
    for line in differing:
        print(line)
    print(f"{len(a)} metrics compared, {len(differing)} differ")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
