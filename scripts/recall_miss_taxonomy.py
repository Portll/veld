#!/usr/bin/env python3
"""Classify every zero-recall@k case of a recall-eval per-case report.

Two questions per miss, answered from the report and the fixture texts alone:

1. Reach: was any gold document in the 100-deep list (`rank_low`, so a rescorer
   could promote it) or in none of it (`never_retrieved`, so only a new
   candidate leg can help)? Needs `recall_at_100` from a run with
   RECALL_DIAG_K=100; without it every miss reads as never retrieved.
2. Failure type, first rule that matches wins:
   `date_in_metadata`  the query names a day or month-year and no gold text
                       carries that date; it exists only in `created_at`.
   `multi_hop_pair`    the case is multi_hop with two or more gold documents.
   `anaphora`          no gold text shares a content word with the query, so
                       the reply only makes sense with the turn before it.
   `other`             everything else.

Stdlib only, like recall_diff.py, so CI needs no pip step.

Usage:
    python3 scripts/recall_miss_taxonomy.py --per-case per-case-rocksdb.json
        --cases tests/recall/locomo-gate-cases.jsonl
        --corpus tests/recall/corpora/locomo-gate.jsonl
        --baseline tests/recall/locomo-gate-baseline.json
        --output evaluations/recall/locomo-gate-miss-taxonomy.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MONTHS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)
MONTH_ALT = "|".join(MONTHS)
DATE_PATTERNS = (
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTH_ALT})\b,?\s*\d{{4}}", re.I),
    re.compile(rf"\b(?:{MONTH_ALT})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}", re.I),
    re.compile(rf"\b(?:{MONTH_ALT})\s+\d{{4}}\b", re.I),
)
YEAR = re.compile(r"\b(19|20)\d{2}\b")
WORD = re.compile(r"[a-z0-9']+")
STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being
    do does did done doing have has had having what which who whom whose when where why how
    it its this that these those there here he she they them his her their him i me my we us
    our you your yes no not very so too just about into over after before during than then
    any some all each both such own same other more most can could would should will shall
    may might must also like get got make made recently mention mentioned did say said think
    thought kind sort one ones thing things something anything""".split()
)
PRONOUNS = frozenset("it its them they that this these those he she his her him there one".split())


def stem(word: str) -> str:
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def content_words(text: str, exclude: set[str]) -> set[str]:
    words = {stem(w) for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}
    return {w for w in words if w not in exclude}


def shares_word(a: set[str], b: set[str]) -> bool:
    # A five-letter common prefix catches allergies/allergic and recipe/recipes without a real stemmer.
    return any(x == y or (len(x) >= 5 and len(y) >= 5 and x[:5] == y[:5]) for x in a for y in b)


def strip_speaker(content: str) -> tuple[str, str]:
    speaker, sep, rest = content.partition(":")
    if sep and " " not in speaker.strip():
        return speaker.strip().lower(), rest.strip()
    return "", content


def query_dates(query: str) -> list[str]:
    return [m.group(0) for p in DATE_PATTERNS for m in p.finditer(query)]


def gold_carries_date(gold_text: str, dates: list[str]) -> bool:
    lowered = gold_text.lower()
    for date in dates:
        month = next((m for m in MONTHS if m in date.lower()), None)
        year = YEAR.search(date)
        if (month and month in lowered) or (year and year.group(0) in lowered):
            return True
    return False


def classify(case: dict[str, Any], gold: list[dict[str, Any]]) -> tuple[str, list[str]]:
    query = case["query"]
    speakers = {strip_speaker(g["content"])[0] for g in gold} - {""}
    gold_texts = [strip_speaker(g["content"])[1] for g in gold]
    hits: list[str] = []

    dates = query_dates(query)
    if dates and not any(gold_carries_date(t, dates) for t in gold_texts):
        hits.append("date_in_metadata")
    if case["category"] == "multi_hop" and case["relevant_total"] >= 2:
        hits.append("multi_hop_pair")
    q_words = content_words(query, speakers)
    overlaps = [shares_word(q_words, content_words(t, speakers)) for t in gold_texts]
    pronominal = [bool(set(WORD.findall(t.lower())) & PRONOUNS) for t in gold_texts]
    if q_words and not any(overlaps) and any(pronominal):
        hits.append("anaphora")
    if not hits:
        hits.append("other")
    return hits[0], hits[1:]


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {row["id"]: row for row in rows}


def build(per_case_path: Path, cases_path: Path, corpus_path: Path, baseline_path: Path | None) -> dict[str, Any]:
    report = json.loads(per_case_path.read_text())
    per_case = report["full"] if isinstance(report, dict) else report
    cases = load_jsonl(cases_path)
    corpus = load_jsonl(corpus_path)

    deep_available = any(c.get("recall_at_100", c["recall_at_k"]) != c["recall_at_k"] for c in per_case)
    for c in per_case:
        if c.get("recall_at_100", 0.0) + 1e-9 < c["recall_at_k"]:
            raise SystemExit(f"{c['case_id']}: recall_at_100 {c['recall_at_100']} < recall_at_k {c['recall_at_k']}")

    rows = []
    for c in per_case:
        if c["recall_at_k"] > 0:
            continue
        gold_ids = [r["corpus_item_id"] for r in cases[c["case_id"]]["relevant"]]
        gold = [corpus[g] for g in gold_ids]
        reach = "rank_low" if c.get("recall_at_100", 0.0) > 0 else "never_retrieved"
        primary, secondary = classify(c, gold)
        rows.append({
            "case_id": c["case_id"],
            "category": c["category"],
            "query": c["query"],
            "reach": reach,
            "recall_at_100": c.get("recall_at_100", 0.0),
            "failure_type": primary,
            "also": secondary,
            "gold": [
                {"id": g["id"], "created_at": g["created_at"], "content": g["content"]}
                for g in gold
            ],
        })

    by_type = Counter(r["failure_type"] for r in rows)
    by_reach = Counter(r["reach"] for r in rows)
    by_category = Counter(r["category"] for r in rows)
    by_category_reach = {
        cat: dict(Counter(r["reach"] for r in rows if r["category"] == cat)) for cat in sorted(by_category)
    }
    by_type_reach = {
        t: dict(Counter(r["reach"] for r in rows if r["failure_type"] == t)) for t in sorted(by_type)
    }

    baseline_sha = ""
    rerank = None
    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        baseline_sha = baseline.get("git_sha", "")
        rerank = baseline.get("rerank")

    return {
        "suite": "locomo-gate",
        "baselineSha": baseline_sha,
        "rerank": rerank,
        "perCaseFile": str(per_case_path),
        "deepListAvailable": deep_available,
        "caseCount": len(per_case),
        "zeroRecall": len(rows),
        "rankLow": by_reach.get("rank_low", 0),
        "neverRetrieved": by_reach.get("never_retrieved", 0),
        "byType": {t: by_type.get(t, 0) for t in ("anaphora", "date_in_metadata", "multi_hop_pair", "other")},
        "byTypeReach": by_type_reach,
        "byCategory": dict(sorted(by_category.items())),
        "byCategoryReach": by_category_reach,
        "rule": {
            "reach": "rank_low when recall_at_100 > 0 for the case, else never_retrieved; "
                     "recall_at_100 is only meaningful from a run with RECALL_DIAG_K=100",
            "failureType": [
                "date_in_metadata: the query names a day-month-year, month-day-year or month-year and no gold "
                "text contains that month name or year, so the date lives only in created_at",
                "multi_hop_pair: category multi_hop with relevant_total >= 2",
                "anaphora: no gold text shares a stemmed content word with the query (speaker names and "
                "stopwords removed) and at least one gold text contains a pronoun or demonstrative",
                "other: no rule matched",
            ],
            "precedence": "first matching rule is failure_type; later matches are listed under also",
        },
        "cases": rows,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-case", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    taxonomy = build(args.per_case, args.cases, args.corpus, args.baseline)
    text = json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    summary = {k: v for k, v in taxonomy.items() if k != "cases"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
