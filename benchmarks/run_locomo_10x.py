#!/usr/bin/env python3
"""Run the LOCOMO-style API benchmark N times against a running veld instance
and aggregate mean/std of the headline metrics across runs.

Each invocation of locomo_api_benchmark.py uses a fresh timestamped user_id,
so the runs are independent retrieval measurements against the same instance.
"""

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCRIPT = HERE / "locomo_api_benchmark.py"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

SAVED_RE = re.compile(r"Raw results saved to:\s*(.+\.json)\s*$", re.MULTILINE)


def run_once(i: int) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"--- run {i} FAILED (exit {proc.returncode}) ---")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(1)
    m = SAVED_RE.search(proc.stdout)
    if not m:
        print(proc.stdout[-2000:])
        raise SystemExit(f"run {i}: could not find saved-results path")
    path = (REPO / m.group(1).strip()).resolve()
    data = json.loads(path.read_text())
    mt = data["metrics"]
    print(
        f"  run {i:2d}: MRR={mt['overall_mrr']:.4f} "
        f"Hit@5={mt['overall_hit_at_5']:.4f} Hit@10={mt['overall_hit_at_10']:.4f} "
        f"R@5={mt['overall_recall_at_5']:.4f} R@10={mt['overall_recall_at_10']:.4f} "
        f"[{path.name}]"
    )
    return data


def main() -> None:
    print("=" * 72)
    print(f"LOCOMO API benchmark — {N} runs")
    print("=" * 72)
    runs = [run_once(i + 1) for i in range(N)]

    keys = [
        ("overall_mrr", "MRR"),
        ("overall_hit_at_5", "Hit@5"),
        ("overall_hit_at_10", "Hit@10"),
        ("overall_recall_at_5", "R@5"),
        ("overall_recall_at_10", "R@10"),
    ]

    print("\n" + "=" * 72)
    print(f"AGGREGATE over {N} runs (mean ± stdev, [min, max])")
    print("-" * 72)
    for key, label in keys:
        vals = [r["metrics"][key] for r in runs]
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label:<7s} {mean:.4f} ± {sd:.4f}   [{min(vals):.4f}, {max(vals):.4f}]")

    # Per-category mean MRR across runs
    cats = ["single-hop", "temporal", "multi-hop", "open-domain"]
    print("-" * 72)
    print("  Per-category mean MRR across runs:")
    for cat in cats:
        vals = [r["per_category"][cat]["mrr"] for r in runs]
        print(f"    {cat:<12s} {statistics.mean(vals):.4f}")
    print("=" * 72)

    summary = {
        "n_runs": N,
        "server_version": runs[0].get("server_version"),
        "aggregate": {
            key: {
                "mean": statistics.mean([r["metrics"][key] for r in runs]),
                "stdev": statistics.pstdev([r["metrics"][key] for r in runs]),
                "min": min(r["metrics"][key] for r in runs),
                "max": max(r["metrics"][key] for r in runs),
            }
            for key, _ in keys
        },
        "per_run": [r["metrics"] for r in runs],
        "user_ids": [r["user_id"] for r in runs],
    }
    out = HERE / "locomo_10run_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    main()
