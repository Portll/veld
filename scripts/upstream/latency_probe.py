#!/usr/bin/env python3
"""Time /api/recall, /api/relevant and /api/remember against a running veld and check budgets.

  latency_probe.py --port 3996 --runs 10 [--user portll] [--budgets docs/upstream/latency-budgets.json]

Run it against an APFS clone of a live data directory, never the live server: remember writes.
The API key comes from VELD_API_KEY. One untimed warm-up pass runs first, because the first
request for a tenant can trigger its startup repair. With --budgets, exits 1 when any
endpoint's p95 exceeds its budget.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

QUERIES = [
    "storage engine cutover and rollback",
    "release procedure and tags",
    "recall gate determinism",
    "what changed in the gap analysis",
    "embedder configuration and llama-server",
]


def post(port, key, route, body, timeout):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    except (urllib.error.URLError, TimeoutError):
        status = 0
    return status, (time.perf_counter() - t0) * 1000


def calls(user, i):
    q = QUERIES[i % len(QUERIES)]
    return [
        ("recall", "/api/recall", {"user_id": user, "query": q, "limit": 5}),
        ("relevant", "/api/relevant", {"user_id": user, "context": q, "max_results": 5,
                                       "config": {"enable_entity_matching": False}}),
        ("relevant_entities", "/api/relevant", {"user_id": user, "context": q, "max_results": 5}),
        ("remember", "/api/remember", {"user_id": "latency-probe", "content": f"latency probe {i} {time.time()}",
                                       "tags": ["latency-probe"]}),
    ]


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--user", default="portll")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--budgets")
    args = ap.parse_args()
    key = os.environ.get("VELD_API_KEY") or sys.exit("set VELD_API_KEY")

    for _, route, body in calls(args.user, 0):
        post(args.port, key, route, body, args.timeout)

    timings, failures = {}, []
    for i in range(args.runs):
        for name, route, body in calls(args.user, i):
            status, ms = post(args.port, key, route, body, args.timeout)
            if status != 200:
                failures.append(f"{name} run {i}: status {status} after {ms:.0f} ms")
            timings.setdefault(name, []).append(ms)

    report = {name: {"p50_ms": round(percentile(v, 0.5)), "p95_ms": round(percentile(v, 0.95)),
                     "max_ms": round(max(v)), "runs": len(v)} for name, v in timings.items()}
    print(json.dumps({"failures": failures, "endpoints": report}, indent=2))

    over = []
    if args.budgets:
        budgets = json.load(open(args.budgets))["p95_ms"]
        over = [f"{n}: p95 {report[n]['p95_ms']} ms > budget {b} ms"
                for n, b in budgets.items() if n in report and report[n]["p95_ms"] > b]
        for line in over:
            print(line, file=sys.stderr)
    sys.exit(1 if failures or over else 0)


if __name__ == "__main__":
    main()
