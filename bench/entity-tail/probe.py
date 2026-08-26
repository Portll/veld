#!/usr/bin/env python3
"""Does recall ever match on a TAIL entity?

`GRAPH-ENTRY-REDACTION.md` measured that 67.2% of records sit at exactly the 50-entity cap, that
about 7 of those 50 are authored by the writer and about 43 are derived, and that `truncate(50)`
keeps the first 50 in ARRIVAL order rather than by relevance. It left one question open and said it
needed instrumentation rather than argument:

    if graph traversal never surfaces a tail entity, the tail is pure cost — storage and
    disclosure with no retrieval benefit — and the answer is to cut hard rather than filter gently.

The recall API returns a `score` and no attribution: nothing in the response says WHICH tag or
entity caused a match, so the question cannot be answered by reading recall output. This probes it
from outside instead, by position.

METHOD. Sample memories that carry enough entities to have a tail. For each, pick one entity from
each positional bucket and ask whether querying that entity alone retrieves the memory it came from.

  head   index 0-6     the authored tags (scope:, intent:, the writer's own)
  mid    index 7-24    early derived
  tail   index 25+     late derived, the part the cap is about

Two endpoints, because the difference between them is the finding:

  /api/recall/tags   exact-ish tag lookup — is the entity INDEXED at all?
  /api/recall        semantic — does the entity contribute to RETRIEVAL?

An entity can be indexed and still never help. That gap is the cost the question is asking about.

SELECTIVITY IS THE REAL METRIC, AND hit@N ALONE MISLEADS. The first run of this probe reported
head 26.7% / mid 45% / tail 28.3% on tag lookup and looked like the head was hardest to retrieve.
It is not. The head misses were `hook` and `scope:claude-code` — tags carried by thousands of
memories. Asking for 20 results from a tag that 4,000 memories share will almost never return the
one you started from, and that is a property of the DENOMINATOR, not of reachability. Hit@N
conflates "not indexed" with "indexed but carries no information", which are opposite problems.

So the probe also reports `mention_count` per tag by bucket, which measures the thing that actually
decides whether an entity is worth its storage: how many memories it fails to distinguish between.
Run `--selectivity <entity-dump.json>` with a dump from `/api/graph/entities/all`.

WHAT THIS DOES NOT MEASURE. Whether real queries ever match a tail entity — that needs match
attribution inside recall, which does not exist. This measures REACHABILITY: whether the tail
*could* be matched. A tail that is not even reachable is definitively pure cost; a reachable one is
not thereby useful. Reachability is the upper bound on usefulness, so a low number here is
conclusive and a high number is not.

Read-only. Plain `/api/recall` does not reinforce (reinforcement is on the feedback path), so this
does not perturb what it measures — but it is not free, and the sample is kept small deliberately.

    VELD_API_KEY=... python3 bench/entity-tail/probe.py --sample 60 \
        --selectivity entity-dump.json > bench/entity-tail/results-<date>.json

Findings written up in docs/ENTITY-TAIL-FINDINGS-2026-08-27.md.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HEAD = (0, 7)
MID = (7, 25)
TAIL = (25, 10**6)
BUCKETS = [("head", HEAD), ("mid", MID), ("tail", TAIL)]


def post(base, path, key, body, timeout=60):
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # Fail closed: a refused query is unknown, never "no match".
        raise SystemExit(f"{path} -> HTTP {e.code}: {e.read()[:200]!r}")


def ids(resp):
    return [m["id"] for m in (resp.get("memories") or resp.get("results") or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("VELD_URL", "http://127.0.0.1:3030"))
    ap.add_argument("--user", default="portll")
    ap.add_argument("--sample", type=int, default=60, help="memories probed per bucket")
    ap.add_argument("--pool", type=int, default=400, help="memories listed to sample from")
    ap.add_argument("--limit", type=int, default=20, help="hit@N")
    ap.add_argument(
        "--selectivity",
        help="a /api/graph/entities/all dump; adds mention_count per bucket, the metric that matters",
    )
    args = ap.parse_args()

    key = os.environ.get("VELD_API_KEY")
    if not key:
        raise SystemExit("VELD_API_KEY not set")

    listing = post(args.base, "/api/memories", key, {"user_id": args.user, "limit": args.pool})
    pool = [m for m in listing["memories"] if len(m.get("tags") or []) >= 26]
    if not pool:
        raise SystemExit(f"no memory in the {args.pool} listed has 26+ tags — nothing has a tail")

    print(
        f"pool: {len(pool)} of {len(listing['memories'])} listed have a tail (26+ tags)",
        file=sys.stderr,
    )

    out = {
        "corpus_total": listing.get("total"),
        "pool": len(pool),
        "listed": len(listing["memories"]),
        "hit_at": args.limit,
        "buckets": {},
    }

    for name, (lo, hi) in BUCKETS:
        probed = tags_hits = sem_hits = 0
        misses = []
        for m in pool[: args.sample]:
            tags = m["tags"]
            window = tags[lo : min(hi, len(tags))]
            if not window:
                continue
            # Middle of the bucket, deterministically — no sampling noise between runs.
            tag = window[len(window) // 2]
            probed += 1

            by_tag = ids(
                post(args.base, "/api/recall/tags", key,
                     {"user_id": args.user, "tags": [tag], "limit": args.limit})
            )
            by_sem = ids(
                post(args.base, "/api/recall", key,
                     {"user_id": args.user, "query": tag, "limit": args.limit})
            )
            t_hit, s_hit = m["id"] in by_tag, m["id"] in by_sem
            tags_hits += t_hit
            sem_hits += s_hit
            if not t_hit and not s_hit and len(misses) < 8:
                misses.append(tag)

        out["buckets"][name] = {
            "index_range": [lo, hi if hi < 10**6 else None],
            "probed": probed,
            "tag_lookup_hits": tags_hits,
            "semantic_hits": sem_hits,
            "tag_lookup_rate": round(tags_hits / probed, 4) if probed else None,
            "semantic_rate": round(sem_hits / probed, 4) if probed else None,
            "example_misses": misses,
        }
        b = out["buckets"][name]
        print(
            f"  {name:<5} n={probed:<4} tag-lookup {b['tag_lookup_rate']}  semantic {b['semantic_rate']}",
            file=sys.stderr,
        )

    if args.selectivity:
        import statistics as stats

        dump = json.load(open(args.selectivity))
        mentions = {e["name"]: e["mention_count"] for e in dump["entities"]}
        print(f"\nselectivity, from {len(mentions)} entities:", file=sys.stderr)
        for name, (lo, hi) in BUCKETS:
            seen, absent = [], 0
            for m in pool:
                for t in m["tags"][lo : min(hi, len(m["tags"]))]:
                    if t in mentions:
                        seen.append(mentions[t])
                    else:
                        absent += 1
            if not seen:
                continue
            b = out["buckets"].setdefault(name, {})
            b["selectivity"] = {
                "tags": len(seen) + absent,
                # A tag with no entity record: same list on paper, not in the store.
                "no_entity_record": absent,
                "median_mentions": stats.median(seen),
                "mean_mentions": round(stats.mean(seen), 1),
                # A tag on 100+ memories cannot pick one out of 4,929.
                "share_on_100_plus": round(sum(v >= 100 for v in seen) / len(seen), 4),
                "share_near_unique": round(sum(v <= 2 for v in seen) / len(seen), 4),
            }
            s = b["selectivity"]
            print(
                f"  {name:<5} median {s['median_mentions']:>6.0f}  "
                f">=100 on {100*s['share_on_100_plus']:>5.1f}%  "
                f"near-unique {100*s['share_near_unique']:>4.1f}%  "
                f"no entity record {absent}",
                file=sys.stderr,
            )

    json.dump(out, sys.stdout, indent=2)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()
