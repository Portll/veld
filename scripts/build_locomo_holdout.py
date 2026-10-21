#!/usr/bin/env python3
"""Build the held-out LoCoMo recall suite: the full LoCoMo suite minus every conversation the gate uses.

The `locomo` suite (1,531 questions over 10 conversations) contains all 100 gate questions, and
the gate is what recall changes are tuned against, so a number measured on `locomo` is partly a
measurement of fit. `locomo-holdout` drops the gate's conversations from both the corpus and the
cases. It also drops from the cases, not the corpus, every case in locomo-holdout-spent.txt: cases
whose individual result someone has examined. Nothing may be tuned against it.

Usage:
    python3 scripts/build_locomo_holdout.py [--check]

Without --check it rewrites the two hold-out files. With --check it exits 1 when they differ from
what it would write. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECALL = ROOT / "tests" / "recall"
SOURCE_CORPUS = RECALL / "corpora" / "locomo.jsonl"
SOURCE_CASES = RECALL / "locomo_cases.jsonl"
GATE_CASES = RECALL / "locomo-gate-cases.jsonl"
HOLDOUT_CORPUS = RECALL / "corpora" / "locomo-holdout.jsonl"
HOLDOUT_CASES = RECALL / "locomo-holdout-cases.jsonl"
SPENT_CASES = RECALL / "locomo-holdout-spent.txt"


def conversation_of_case(case_id: str) -> str:
    return case_id.split("_", 1)[0]


def conversation_of_turn(turn_id: str) -> str:
    return turn_id.split(":", 1)[0]


def read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_spent() -> set[str]:
    return {
        line.strip()
        for line in read_lines(SPENT_CASES)
        if not line.lstrip().startswith("#")
    }


def build() -> tuple[str, str, set[str]]:
    gate_conversations = {
        conversation_of_case(json.loads(line)["id"]) for line in read_lines(GATE_CASES)
    }
    spent = read_spent()
    unknown = sorted(spent - {json.loads(line)["id"] for line in read_lines(SOURCE_CASES)})
    if unknown:
        raise SystemExit(f"{SPENT_CASES.name} names cases that do not exist: {unknown}")
    corpus = [
        line
        for line in read_lines(SOURCE_CORPUS)
        if conversation_of_turn(json.loads(line)["id"]) not in gate_conversations
    ]
    cases = []
    for line in read_lines(SOURCE_CASES):
        case = json.loads(line)
        if conversation_of_case(case["id"]) in gate_conversations or case["id"] in spent:
            continue
        leaked = [
            item["corpus_item_id"]
            for item in case["relevant"]
            if conversation_of_turn(item["corpus_item_id"]) in gate_conversations
        ]
        if leaked:
            raise SystemExit(f"{case['id']} cites gate turns {leaked}")
        cases.append(line)
    return "\n".join(corpus) + "\n", "\n".join(cases) + "\n", gate_conversations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the files are stale")
    args = parser.parse_args()

    corpus, cases, gate_conversations = build()
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in ((HOLDOUT_CORPUS, corpus), (HOLDOUT_CASES, cases))
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            print(f"stale: {', '.join(stale)}; run scripts/build_locomo_holdout.py", file=sys.stderr)
            return 1
        return 0

    HOLDOUT_CORPUS.write_text(corpus, encoding="utf-8")
    HOLDOUT_CASES.write_text(cases, encoding="utf-8")
    print(
        f"excluded {sorted(gate_conversations)}: "
        f"{corpus.count(chr(10))} turns, {cases.count(chr(10))} cases"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
