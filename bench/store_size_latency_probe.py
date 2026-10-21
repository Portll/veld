#!/usr/bin/env python3
"""
store_size_latency_probe.py -- store-size latency lane for veld.

bench/recall_latency_guard.py runs against a 20-memory corpus, so a path that
costs O(N) per request costs the same before and after a fix and the guard
cannot see it.  This probe seeds one tenant to N memories (default 10 000),
creates a fresh tenant in the same run, and measures per-route latency on both:

  remember        POST /api/remember
  recall          POST /api/recall
  memory_get      GET  /api/memory/{id}            (full UUID)
  relevant        POST /api/relevant
  stats           GET  /api/users/{id}/stats
  health_index    GET  /api/health/index?user_id=  (per-tenant index health)
  health_public   GET  /health/index               (public aggregate probe)

It exits non-zero when any large-tenant p95 exceeds its threshold, when the
large/fresh p95 ratio exceeds --max-ratio, or when any call times out or errors.
A timeout or error is recorded as a failed sample charged at the timeout value;
it is never dropped from the percentiles.

Environment:
  VELD_API_URL   server base URL  (default http://127.0.0.1:3030)
  VELD_API_KEY   API key          (required)
  GITHUB_SHA     commit sha recorded in the JSON (falls back to `git rev-parse HEAD`)

Standard library only.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

CALL_TIMEOUT_S = 60.0
BATCH_SIZE = 100  # server cap is 1000 per batch; 100 keeps each call well under the timeout

DEFAULT_THRESHOLDS = {
    "remember": 2.0,
    "recall": 2.0,
    "memory_get": 0.2,
    "relevant": 1.0,
    "stats": 0.2,
    "health_index": 0.2,
    "health_public": 0.2,
}
ROUTES = list(DEFAULT_THRESHOLDS)

TOPICS = ["payment gateway", "cache invalidation", "onboarding wizard", "kubernetes rollout",
          "analytics pipeline", "checkout latency", "retention policy", "incident review",
          "schema migration", "vector index", "release train", "customer interview"]
PEOPLE = ["Sarah", "Raj", "Marcus", "Elena", "Priya", "Tomas", "Aiko", "Lena"]
VERBS = ["decided on", "investigated", "rolled back", "benchmarked", "documented", "escalated",
         "prototyped", "reviewed"]


class CallFailure(Exception):
    pass


def varied_content(i: int, salt: str) -> str:
    """Distinct, topically varied text so embeddings, entities and dedup see real spread."""
    p = PEOPLE[i % len(PEOPLE)]
    q = PEOPLE[(i * 7 + 3) % len(PEOPLE)]
    t = TOPICS[(i * 5 + 1) % len(TOPICS)]
    v = VERBS[(i * 3 + 2) % len(VERBS)]
    return (f"[{salt}-{i}] {p} {v} the {t} work with {q} in week {i % 52}; "
            f"ticket {1000 + i} touched module m{i % 97} and took {i % 13 + 1} days.")


class Client:
    def __init__(self, base: str, key: str):
        self.base = base.rstrip("/")
        self.key = key

    def call(self, method: str, path: str, body=None):
        """Return (seconds, parsed_json). Raise CallFailure on timeout, transport error or non-2xx."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"X-API-Key": self.key, "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raise CallFailure(f"HTTP {e.code} {method} {path}: {e.read()[:200]!r}")
        except (TimeoutError, OSError) as e:  # socket.timeout is TimeoutError; URLError is an OSError
            raise CallFailure(f"{type(e).__name__} {method} {path}: {e}")
        dt = time.perf_counter() - t0
        try:
            return dt, json.loads(raw) if raw else None
        except ValueError:
            raise CallFailure(f"non-JSON body from {method} {path}")


def seed(client: Client, user: str, n: int, salt: str, log=print):
    """Seed n memories; returns the list of created memory ids."""
    ids = []
    done = 0
    t0 = time.perf_counter()
    while done < n:
        k = min(BATCH_SIZE, n - done)
        items = [{"content": varied_content(done + j, salt),
                  "tags": [TOPICS[(done + j) % len(TOPICS)].replace(" ", "-"), "store-size-probe"]}
                 for j in range(k)]
        _, resp = client.call("POST", "/api/remember/batch", {"user_id": user, "memories": items})
        if resp.get("failed"):
            raise CallFailure(f"batch seed had {resp['failed']} failures: {resp.get('errors')}")
        ids.extend(resp.get("memory_ids", []))
        done += k
        if done % 1000 == 0 or done == n:
            log(f"  seeded {done}/{n} for {user} ({time.perf_counter() - t0:.0f}s)")
    return ids


def percentile(sorted_vals, q):
    """Nearest-rank percentile on a non-empty sorted list."""
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def measure(client: Client, user: str, ids, calls: int, tag: str):
    """Return {route: {p50, p95, max, samples, failures, errors}} for one tenant."""
    uq = urllib.parse.quote(user, safe="")
    out = {}

    def run(route, fn):
        lat, errors = [], []
        for i in range(calls):
            try:
                dt, _ = fn(i)
                lat.append(dt)
            except CallFailure as e:
                lat.append(CALL_TIMEOUT_S)  # a failed call is charged at the full timeout
                errors.append(str(e))
        s = sorted(lat)
        out[route] = {"p50": percentile(s, 0.50), "p95": percentile(s, 0.95), "max": s[-1],
                      "samples": len(s), "failures": len(errors), "errors": errors[:3]}

    run("remember", lambda i: client.call("POST", "/api/remember", {
        "user_id": user, "content": varied_content(900000 + i, f"probe-{tag}-{uuid.uuid4().hex[:8]}"),
        "tags": ["store-size-probe"]}))
    run("recall", lambda i: client.call("POST", "/api/recall", {
        "user_id": user, "query": f"{TOPICS[i % len(TOPICS)]} {PEOPLE[i % len(PEOPLE)]}", "limit": 10}))
    run("memory_get", lambda i: client.call("GET", f"/api/memory/{ids[i % len(ids)]}?user_id={uq}"))
    run("relevant", lambda i: client.call("POST", "/api/relevant", {
        "user_id": user, "context": f"working on the {TOPICS[i % len(TOPICS)]} with {PEOPLE[i % len(PEOPLE)]}"}))
    run("stats", lambda i: client.call("GET", f"/api/users/{uq}/stats"))
    run("health_index", lambda i: client.call("GET", f"/api/health/index?user_id={uq}"))
    run("health_public", lambda i: client.call("GET", "/health/index"))
    return out


def commit_sha():
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate(large, fresh, thresholds, max_ratio, ratio_floor):
    """Return (rows, violations)."""
    rows, violations = [], []
    for r in ROUTES:
        L, F = large[r], fresh[r]
        ratio = L["p95"] / max(F["p95"], ratio_floor)
        thr = thresholds[r]
        row = {"route": r, "large_p50": L["p50"], "large_p95": L["p95"], "fresh_p50": F["p50"],
               "fresh_p95": F["p95"], "ratio": ratio, "threshold": thr, "status": "ok"}
        if L["failures"] or F["failures"]:
            violations.append(f"{r}: {L['failures']} large / {F['failures']} fresh failed calls "
                              f"(timeout {CALL_TIMEOUT_S:.0f}s); first: {(L['errors'] or F['errors'])[0]}")
            row["status"] = "FAIL"
        if L["p95"] > thr:
            violations.append(f"{r}: large-tenant p95 {L['p95']:.3f}s exceeds threshold {thr:.3f}s")
            row["status"] = "FAIL"
        if ratio > max_ratio:
            violations.append(f"{r}: large/fresh p95 ratio {ratio:.1f}x exceeds {max_ratio:.1f}x")
            row["status"] = "FAIL"
        rows.append(row)
    return rows, violations


def print_table(rows):
    hdr = f"{'route':<14}{'large p50':>11}{'large p95':>11}{'fresh p50':>11}{'fresh p95':>11}{'ratio':>8}{'limit':>8}  status"
    print(hdr)
    print("-" * len(hdr))
    for w in rows:
        print(f"{w['route']:<14}{w['large_p50']:>10.3f}s{w['large_p95']:>10.3f}s{w['fresh_p50']:>10.3f}s"
              f"{w['fresh_p95']:>10.3f}s{w['ratio']:>7.1f}x{w['threshold']:>7.2f}s  {w['status']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Store-size latency probe: large tenant vs fresh tenant, per route.")
    ap.add_argument("--size", type=int, default=10000, help="memories to seed on the large tenant (default 10000)")
    ap.add_argument("--calls", "-m", type=int, default=20, help="measured calls per route per tenant (default 20)")
    ap.add_argument("--out", required=True, help="path to write the JSON result")
    ap.add_argument("--thresholds", help="JSON file of per-route p95 limits in seconds; "
                                         f"keys: {', '.join(ROUTES)}; unlisted routes keep the defaults")
    ap.add_argument("--max-ratio", type=float, default=5.0, help="max large/fresh p95 ratio (default 5)")
    ap.add_argument("--ratio-floor", type=float, default=0.05,
                    help="fresh p95 is floored at this many seconds in the ratio, so sub-50ms noise "
                         "on a fresh tenant cannot manufacture a ratio failure (default 0.05)")
    ap.add_argument("--fresh-size", type=int, default=20, help="memories seeded on the fresh tenant (default 20)")
    args = ap.parse_args(argv)

    if args.size < 1 or args.calls < 1 or args.fresh_size < 1:
        ap.error("--size, --calls and --fresh-size must be >= 1")
    key = os.environ.get("VELD_API_KEY")
    if not key:
        print("VELD_API_KEY is not set", file=sys.stderr)
        return 2
    base = os.environ.get("VELD_API_URL", "http://127.0.0.1:3030")

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds:
        with open(args.thresholds) as f:
            override = json.load(f)
        unknown = set(override) - set(ROUTES)
        if unknown:
            ap.error(f"unknown route(s) in thresholds: {sorted(unknown)}")
        thresholds.update({k: float(v) for k, v in override.items()})

    client = Client(base, key)
    run_id = uuid.uuid4().hex[:8]
    large_user, fresh_user = f"sizeprobe-large-{run_id}", f"sizeprobe-fresh-{run_id}"
    started = datetime.now(timezone.utc)

    try:
        print(f"seeding large tenant {large_user} to {args.size} memories at {base}")
        large_ids = seed(client, large_user, args.size, "L")
        print(f"seeding fresh tenant {fresh_user} to {args.fresh_size} memories")
        fresh_ids = seed(client, fresh_user, args.fresh_size, "F")
        if not large_ids or not fresh_ids:
            raise CallFailure("seeding returned no memory ids")
        print(f"measuring {args.calls} calls per route per tenant")
        large = measure(client, large_user, large_ids, args.calls, "L")
        fresh = measure(client, fresh_user, fresh_ids, args.calls, "F")
    except CallFailure as e:
        print(f"probe could not complete: {e}", file=sys.stderr)
        return 2

    rows, violations = evaluate(large, fresh, thresholds, args.max_ratio, args.ratio_floor)
    print()
    print_table(rows)

    result = {
        "commit": commit_sha(),
        "server": base,
        "store_size": args.size,
        "fresh_size": args.fresh_size,
        "calls_per_route": args.calls,
        "call_timeout_s": CALL_TIMEOUT_S,
        "max_ratio": args.max_ratio,
        "ratio_floor_s": args.ratio_floor,
        "thresholds": thresholds,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "large_tenant": large_user,
        "fresh_tenant": fresh_user,
        "large": large,
        "fresh": fresh,
        "table": rows,
        "violations": violations,
        "passed": not violations,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")

    if violations:
        print("\nFAILED:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
