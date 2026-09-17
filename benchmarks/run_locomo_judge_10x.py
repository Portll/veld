#!/usr/bin/env python3
"""Run the LOCOMO LLM-judged evaluation N times against the clean veld instance
using the Together.ai judge configured in .env, and aggregate across runs.

Each run of locomo_llm_judge.py writes evaluations/locomo_judge_<gitbuild>.json
(same name every run, since build == git short HEAD), so we copy each run's
output to a per-run file before the next run overwrites it.

Judge creds come from .env (VELD_JUDGE_URL / VELD_JUDGE_MODEL / VELD_JUDGE_API_KEY).
The veld target is forced to the clean instance; pass URL/key via env to override.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCRIPT = HERE / "locomo_llm_judge.py"
EVALS = REPO / "evaluations"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

VELD_TARGET_URL = os.environ.get("VELD_TARGET_URL", "http://127.0.0.1:3031")
VELD_TARGET_KEY = os.environ.get("VELD_TARGET_KEY", "dev-key-local")


def load_dotenv(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def git_build() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "unknown"


def run_once(i: int, child_env: dict, build: str) -> dict:
    out_path = EVALS / f"locomo_judge_{build}.json"
    if out_path.exists():
        out_path.unlink()  # ensure we read THIS run's output
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO),
        env=child_env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        print(f"--- run {i} FAILED (exit {proc.returncode}) ---")
        print(proc.stdout[-3000:])
        print(proc.stderr[-2000:])
        raise SystemExit(1)
    data = json.loads(out_path.read_text())
    # Preserve this run's output before the next overwrites it.
    run_copy = EVALS / f"locomo_judge_{build}_run{i:02d}.json"
    shutil.copy(out_path, run_copy)
    o = data["overall"]
    print(
        f"  run {i:2d}: MRR={o['mrr']:.4f} R@5={o['r5']:.4f} R@10={o['r10']:.4f} "
        f"evid={o['evidence_sufficiency_avg']:.3f}/3 corr={o['answer_correctness_avg']:.3f}/3 "
        f"gap={o['generation_gap_count']} mis/q={o['avg_misleading_per_query']:.2f} "
        f"[{run_copy.name}]"
    )
    return data


def main() -> None:
    dotenv = load_dotenv(REPO / ".env")
    child_env = os.environ.copy()
    # Judge creds from .env (do not echo the API key).
    for k in ("VELD_JUDGE_URL", "VELD_JUDGE_MODEL", "VELD_JUDGE_API_KEY"):
        if dotenv.get(k):
            child_env[k] = dotenv[k]
    # Force the veld target to the clean instance.
    child_env["VELD_URL"] = VELD_TARGET_URL
    child_env["VELD_API_KEY"] = VELD_TARGET_KEY
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    build = git_build()

    print("=" * 72)
    print(f"LOCOMO LLM-JUDGE benchmark — {N} runs")
    print(f"  veld target : {VELD_TARGET_URL}")
    print(f"  judge model : {child_env.get('VELD_JUDGE_MODEL')}")
    print(f"  judge url   : {child_env.get('VELD_JUDGE_URL')}")
    print(f"  judge key   : {'set' if child_env.get('VELD_JUDGE_API_KEY') else 'MISSING'}")
    print(f"  git build   : {build}")
    print("=" * 72)

    if not child_env.get("VELD_JUDGE_API_KEY"):
        raise SystemExit("VELD_JUDGE_API_KEY not found in .env — cannot reach Together.ai")

    runs = [run_once(i + 1, child_env, build) for i in range(N)]

    keys = [
        ("mrr", "MRR"),
        ("r5", "R@5"),
        ("r10", "R@10"),
        ("evidence_sufficiency_avg", "Evidence/3"),
        ("answer_correctness_avg", "Correct/3"),
        ("avg_misleading_per_query", "Mislead/q"),
        ("avg_contradictions_per_query", "Contra/q"),
    ]

    print("\n" + "=" * 72)
    print(f"AGGREGATE over {N} runs (mean ± stdev, [min, max])")
    print("-" * 72)
    for key, label in keys:
        vals = [r["overall"][key] for r in runs]
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label:<11s} {mean:.4f} ± {sd:.4f}   [{min(vals):.4f}, {max(vals):.4f}]")

    cats = ["single-hop", "temporal", "multi-hop", "open-domain"]
    print("-" * 72)
    print("  Per-category mean answer-correctness across runs:")
    for cat in cats:
        vals = []
        for r in runs:
            row = next((b for b in r["by_type"] if b["type"] == cat), None)
            if row:
                vals.append(row["answer_correctness_avg"])
        if vals:
            print(f"    {cat:<12s} corr={statistics.mean(vals):.3f}/3")
    print("=" * 72)

    summary = {
        "n_runs": N,
        "judge_model": runs[0].get("judge_model"),
        "veld_target": VELD_TARGET_URL,
        "server_version": runs[0].get("version"),
        "build": build,
        "aggregate": {
            key: {
                "mean": statistics.mean([r["overall"][key] for r in runs]),
                "stdev": statistics.pstdev([r["overall"][key] for r in runs]),
                "min": min(r["overall"][key] for r in runs),
                "max": max(r["overall"][key] for r in runs),
            }
            for key, _ in keys
        },
        "per_run": [r["overall"] for r in runs],
    }
    out = EVALS / f"locomo_judge_{build}_10run_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    main()
