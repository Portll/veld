#!/usr/bin/env python3
"""The upstream ledger: one row per upstream shodh-memory first-parent commit after the fork base,
plus outbound rows for veld changes offered upstream.

  ledger.py sync  [--upstream <repo>] [--head <rev>]   append new upstream commits as pending
  ledger.py check [--veld <repo>]                      validate every row

Both default to this checkout; sync reads `upstream/main`, so the checkout needs the remote:
  git remote add upstream https://github.com/varun29ankuS/shodh-memory.git && git fetch upstream
"""
import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "docs" / "upstream" / "LEDGER.tsv"
FORK_BASE = "2737b4f"
COLUMNS = ["direction", "upstream_commit", "pr", "date", "theme", "wave", "status", "veld_commit", "reason"]
INBOUND_STATUSES = {"pending", "ported", "adapted", "partial", "already-in-veld", "veld-origin",
                    "declined", "superseded", "veld-layer"}
OUTBOUND_STATUSES = {"open", "merged", "closed"}
NEEDS_REASON = {"adapted", "partial", "declined", "superseded", "veld-layer", "closed"}
NEEDS_VELD_COMMIT = {"ported", "adapted", "partial", "already-in-veld"}

# First match wins, so the narrower subsystems come before the broad ones.
THEMES = [
    ("anomaly", "W12", r"anomal|surprise|freegad"),
    ("robotics-geo", "W12", r"zenoh|robot|ros2|mission|/geo|geo_|gdelt"),
    ("mcp-hooks", "W9", r"^mcp-server/|^hooks/|/mcp"),
    ("web-ui", "W11", r"^front/|^web/|workbench|seat|demo|\.tsx?$|\.css$"),
    ("ner-entities", "W6", r"gliner|spacy|gazetteer|toponym|wikidata|/ner|entity_alias|entities?\.rs"),
    ("lineage", "W4", r"lineage|causal|precedes"),
    ("facts-todos", "W5", r"facts?\.rs|todos?\.rs|consolidat|compression\.rs|narrative"),
    ("storage-integrity", "W7", r"oplog|integrity|sources?\.rs|source_cursor|memory/storage\.rs|backup|serialization|migration"),
    ("embeddings-vectors", "W8", r"embedding|vector_db|vamana|spann|chunk"),
    ("graph", "W3", r"graph_memory|graph_retrieval|relation|pmi|curvature|hebbian"),
    ("retrieval", "W2", r"retrieval|hybrid_search|query_parser|rerank|cross_encoder|fusion|recall\.rs|ppr|pagerank"),
    ("mcp-hooks", "W9", r"mcp|hooks?/|\.claude"),
    ("http-api", "W10", r"handlers/|openapi|router"),
    ("eval", "W0", r"recall_harness|tests/recall|recall_eval|eval|bench|gate"),
]
# Subjects and merged branch names name the work more reliably than which files a merge touched.
SUBJECT_THEMES = [
    ("anomaly", "W12", r"anomal|surprise|freegad"),
    ("robotics-geo", "W12", r"zenoh|robot|ros2|mission|\bgeo|gdelt|spatial"),
    ("web-ui", "W11", r"web ?ui|frontend|front-end|react|dashboard|workbench|\bseat|demo|\bui\b"),
    ("eval", "W0", r"harness|\beval|gate|benchmark|locomo|longmemeval|\bRH-?\d|ablation"),
    ("ner-entities", "W6", r"\bner\b|gliner|spacy|gazetteer|toponym|wikidata|entit"),
    ("lineage", "W4", r"lineage|causal|root.cause|precedes"),
    ("facts-todos", "W5", r"\bfacts?\b|todo|consolidat|narrative|compression"),
    ("storage-integrity", "W7", r"oplog|integrity|scrub|\bsources?\b|storage|backup|postcard|serializ|migrat"),
    ("embeddings-vectors", "W8", r"embed|vector|vamana|spann|chunk"),
    ("graph", "W3", r"graph|curvature|ricci|\bedges?\b|relation|pmi|hebbian"),
    ("retrieval", "W2", r"recall|retriev|rank|fusion|\bppr\b|pagerank|bm25|search"),
    ("mcp-hooks", "W9", r"\bmcp\b|hooks?\b"),
    ("http-api", "W10", r"\bapi\b|route|endpoint|openapi|handler"),
]
INFRA = re.compile(r"^(chore|ci|docs?|style|build|deps)\b|rustfmt|clippy|cargo\.lock|\bbump|release/|dependabot|version", re.I)
SECURITY = re.compile(r"\b(security|csp|auth\w*|tls|https|rustsec|advisor\w*|secrets?|confine)\b", re.I)
PR = re.compile(r"\(#(\d+)\)|Merge pull request #(\d+)|Merge #(\d+)")


def read(path=LEDGER):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            sys.exit(f"{path}: header is {reader.fieldnames}, expected {COLUMNS}")
        return list(reader)


def write(rows, path=LEDGER):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def classify(subject, files):
    if SECURITY.search(subject):
        return "security", "W1"
    if INFRA.search(subject):
        return "infra", "-"
    for theme, wave, pattern in SUBJECT_THEMES:
        if re.search(pattern, subject, re.I):
            return theme, wave
    votes = Counter()
    for path in files:
        for theme, wave, pattern in THEMES:
            if re.search(pattern, path, re.I):
                votes[(theme, wave)] += 1
                break
    if not votes:
        return "infra", "-"
    return votes.most_common(1)[0][0]


def first_parent_commits(upstream, head):
    log = git(upstream, "log", "--reverse", "--first-parent", "-m", "--name-only",
              "--format=@@%h%x09%ad%x09%s", "--date=short", f"{FORK_BASE}..{head}")
    commits, current = [], None
    for line in log.splitlines():
        if line.startswith("@@"):
            sha, date, subject = line[2:].split("\t", 2)
            current = {"sha": sha, "date": date, "subject": subject, "files": []}
            commits.append(current)
        elif line.strip() and current is not None:
            current["files"].append(line.strip())
    return commits


def sync(args):
    rows = read()
    known = {r["upstream_commit"] for r in rows if r["direction"] == "in"}
    added = 0
    for c in first_parent_commits(Path(args.upstream), args.head):
        if c["sha"] in known:
            continue
        m = PR.search(c["subject"])
        theme, wave = classify(c["subject"], c["files"])
        rows.append({
            "direction": "in", "upstream_commit": c["sha"],
            "pr": next((g for g in m.groups() if g), "") if m else "",
            "date": c["date"], "theme": theme, "wave": wave, "status": "pending",
            "veld_commit": "", "reason": "",
        })
        added += 1
    write(rows)
    print(f"{added} upstream commit(s) appended; {len(rows)} rows")


def check(args):
    rows = read()
    errors = []
    seen = set()
    commit_ok = {}

    def veld_commit_exists(ref):
        if ref not in commit_ok:
            r = subprocess.run(["git", "-C", args.veld, "cat-file", "-e", f"{ref}^{{commit}}"],
                               capture_output=True)
            commit_ok[ref] = r.returncode == 0
        return commit_ok[ref]

    for n, r in enumerate(rows, start=2):
        where = f"LEDGER.tsv:{n}"
        allowed = INBOUND_STATUSES if r["direction"] == "in" else OUTBOUND_STATUSES
        if r["direction"] not in {"in", "out"}:
            errors.append(f"{where}: direction '{r['direction']}' is neither in nor out")
            continue
        if r["status"] not in allowed:
            errors.append(f"{where}: status '{r['status']}' is not one of {sorted(allowed)}")
        if r["direction"] == "in":
            if not r["upstream_commit"]:
                errors.append(f"{where}: an inbound row names no upstream commit")
            elif r["upstream_commit"] in seen:
                errors.append(f"{where}: upstream commit {r['upstream_commit']} has a second row")
            seen.add(r["upstream_commit"])
        if r["status"] in NEEDS_REASON and not r["reason"]:
            errors.append(f"{where}: status '{r['status']}' needs a reason")
        if r["status"] in NEEDS_VELD_COMMIT and not r["veld_commit"]:
            errors.append(f"{where}: status '{r['status']}' needs a veld commit")
        for ref in filter(None, r["veld_commit"].split(",")):
            if not veld_commit_exists(ref.strip()):
                errors.append(f"{where}: veld commit {ref.strip()} does not exist")
    for e in errors:
        print(e, file=sys.stderr)
    counts = Counter((r["direction"], r["status"]) for r in rows)
    print(", ".join(f"{d}/{s} {c}" for (d, s), c in sorted(counts.items())))
    sys.exit(1 if errors else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--upstream", default=str(LEDGER.parents[2]))
    s.add_argument("--head", default="upstream/main")
    s.set_defaults(func=sync)
    c = sub.add_parser("check")
    c.add_argument("--veld", default=str(LEDGER.parents[2]))
    c.set_defaults(func=check)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
