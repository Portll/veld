#!/usr/bin/env python3
"""Exit 0 when two recall-eval reports, one per storage engine, carry the same scores.

  recall_engines_equal.py <rocksdb report.json> <redb report.json>

The storage engine holds the memories and the graph; it must not move a ranking. Every score in
`layers` and `by_category` is compared, and any difference is printed and fails the run. Latencies
are wall-clock times from two different runners, so they are left out.

The two reports come from two jobs, and GitHub assigns each its own runner. When the models ran live
on two CPU kernel classes, identical code ranks differently (`KernelClass` in
src/recall_harness/report.rs), so a difference between such a pair says nothing about the engines:
it exits 2, as a pair from two embedders or two sets of model outputs does, and a re-run settles it.
Equal scores on two classes still pass.
"""
import json
import sys
from pathlib import Path

EPSILON = 1e-12


def metrics(report):
    out = {}
    for layer, values in (report.get("layers") or {}).items():
        for name, value in (values or {}).items():
            if isinstance(value, (int, float)) and not name.startswith("latency_"):
                out[f"layers.{layer}.{name}"] = value
    for category, values in (report.get("by_category") or {}).items():
        for name, value in (values or {}).items():
            if isinstance(value, (int, float)) and not name.startswith("latency_"):
                out[f"by_category.{category}.{name}"] = value
    return out


def kernel_class(report):
    kernel = report.get("kernel")
    return kernel.get("class") or "" if isinstance(kernel, dict) else ""


def verdict(left, right, left_name="left", right_name="right"):
    """(exit code, lines to print) for two reports."""
    if left.get("embedder") != right.get("embedder"):
        return 2, [f"embedders differ: {left.get('embedder')} vs {right.get('embedder')}"]
    if left.get("model_tape") != right.get("model_tape"):
        return 2, [
            f"model outputs differ: {left.get('model_tape') or 'live'} vs "
            f"{right.get('model_tape') or 'live'}"
        ]
    a, b = metrics(left), metrics(right)
    if not a:
        return 2, [f"{left_name} carries no metrics"]
    lines = []
    left_class, right_class = kernel_class(left), kernel_class(right)
    lines.append(
        f"CPU kernel class: {left_name} {left_class or 'not recorded'}, "
        f"{right_name} {right_class or 'not recorded'}"
    )
    differing = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            differing.append(f"{key}: present in only one report")
        elif abs(a[key] - b[key]) > EPSILON:
            differing.append(f"{key}: {a[key]!r} vs {b[key]!r}")
    lines.extend(differing)
    lines.append(f"{len(a)} metrics compared, {len(differing)} differ")
    if not differing:
        return 0, lines
    replayed = left.get("model_tape") is not None
    if not replayed and (not left_class or left_class != right_class):
        lines.append(
            "The models ran live on two different, or unrecorded, CPU kernel classes, so this "
            "difference cannot be attributed to the storage engine. Re-run the workflow."
        )
        return 2, lines
    return 1, lines


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    left_path, right_path = map(Path, sys.argv[1:])
    left, right = (json.loads(p.read_text()) for p in (left_path, right_path))
    code, lines = verdict(left, right, str(left_path), str(right_path))
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
