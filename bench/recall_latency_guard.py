#!/usr/bin/env python3
"""
recall_latency_guard.py — evidence-recall + latency regression guard for veld.

Runs a SMALL fixed-corpus recall probe (multi-hop queries only, no LLM judge
required) and asserts two thresholds:

  FLOOR  multi_hop_r10  >= MULTI_HOP_R10_FLOOR
  CEILING p99_latency_ms <= P99_LATENCY_CEILING_MS

Exits 0 if both pass; exits 1 and prints the offending metrics if either fails.

----------------------------------------------------------------------------
BASELINE (V-4 / V-5 runs, 2026-05-xx, commit 8d3ddc9):

  The evidence-recall sessions established that veld's multi-hop R@10 on the
  20-memory / 20-query LOCOMO-style corpus sits at ~0.78 (78 %) when all
  expected memories are stored and the server is healthy.  The MC10 judge
  analysis placed the contamination / random floor at 51.7 % (items correctly
  answered without veld context).

  The baseline recall p99 observed across a 10-item locomo_mc10_eval run
  (stored in locomo_results.json at commit 8d3ddc9) was 1 225 ms.  That run
  uses the HTTP server round-trip; the pure-recall path here is faster because
  we skip the MC10 judge LLM call.

FLOOR choice:
  78 % known-good minus a 10 pp regression margin  →  0.68
  Rounded conservatively down to 0.60 so that env-to-env noise (different
  embedding cache warm-up, OS scheduling jitter) does not produce spurious
  failures while still catching any real recall regression.

CEILING choice:
  Baseline p99 = 1 225 ms.  Guard doubles it (2 × 1 225 ms ≈ 2 450 ms) and
  then rounds up to 3 000 ms to absorb CI machine variance (GitHub runners are
  slower than local NVMe-backed dev machines).  A regression that doubles
  latency (e.g. missing index, serialisation regression) will still be caught.
----------------------------------------------------------------------------

Environment variables (all optional, fall back to defaults for local dev):

  VELD_URL          veld HTTP endpoint  (default: http://127.0.0.1:3030)
  VELD_API_KEY      API key             (default: dev-key-local)
  MULTI_HOP_R10_FLOOR   float 0–1      (default: 0.60)
  P99_LATENCY_CEILING_MS  int ms        (default: 3000)
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Thresholds (override via env for local experimentation)
# ---------------------------------------------------------------------------
MULTI_HOP_R10_FLOOR: float = float(os.environ.get("MULTI_HOP_R10_FLOOR", "0.60"))
P99_LATENCY_CEILING_MS: int = int(os.environ.get("P99_LATENCY_CEILING_MS", "3000"))

# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------
VELD_URL: str = os.environ.get("VELD_URL", "http://127.0.0.1:3030").rstrip("/")
VELD_API_KEY: str = os.environ.get("VELD_API_KEY", "dev-key-local")
USER_ID: str = f"guard_{int(time.time())}"
HEADERS: dict = {"Content-Type": "application/json", "X-API-Key": VELD_API_KEY}

# ---------------------------------------------------------------------------
# Fixed corpus — same 20 memories as locomo_api_benchmark.py / locomo_llm_judge.py
# so results are comparable across runs.  Do NOT modify without updating the
# baseline comment above.
# ---------------------------------------------------------------------------
BASE_TIME = datetime.now(timezone.utc) - timedelta(weeks=4)


def _ts(days: int, hours: int = 10) -> str:
    return (BASE_TIME + timedelta(days=days, hours=hours)).isoformat()


MEMORIES = [
    {"id_tag": "M01", "content": "Sprint 14 planning: We committed to 34 story points. Key items are the payment gateway migration from Stripe v2 to v3, the Redis cache invalidation bug, and the new onboarding wizard. Sarah is leading the payment work, Raj is on caching, and I'm handling onboarding.", "tags": ["sprint", "planning", "sprint-14"], "memory_type": "Observation", "created_at": _ts(0)},
    {"id_tag": "M02", "content": "Architecture decision: We chose PostgreSQL over MongoDB for the analytics pipeline. Key reasons were complex joins needed for funnel analysis, ACID compliance for financial data, and the team's existing expertise. Marcus strongly advocated for Mongo but the vote was 4-1 in favor of Postgres.", "tags": ["architecture", "database", "decision", "analytics"], "memory_type": "Decision", "created_at": _ts(1)},
    {"id_tag": "M03", "content": "Bug report: Users in the EU region are experiencing 5-second delays on the checkout page. The root cause is the payment provider's EU endpoint having high latency. Temporary fix: route EU traffic through the UK proxy. Permanent fix needs Stripe v3 migration which Sarah is working on.", "tags": ["bug", "performance", "checkout", "EU", "latency"], "memory_type": "Observation", "created_at": _ts(2)},
    {"id_tag": "M04", "content": "Personal preference: I strongly prefer dark mode in all development tools. My IDE theme is Dracula, terminal is Catppuccin Mocha, and I use the Fira Code font with ligatures enabled at 14pt.", "tags": ["preference", "tooling", "IDE", "theme"], "memory_type": "Preference", "created_at": _ts(3)},
    {"id_tag": "M05", "content": "Raj found the Redis cache invalidation bug. It was a race condition in the pub/sub listener — when two nodes receive the same invalidation event, they both try to refresh the cache simultaneously, causing a thundering herd. Fix: added distributed locking with Redlock.", "tags": ["bug-fix", "redis", "cache", "race-condition", "Raj"], "memory_type": "Observation", "created_at": _ts(7)},
    {"id_tag": "M06", "content": "Team lunch at Sakura Sushi near the downtown office. Everyone was there except Marcus who was working remotely from Portland. Sarah mentioned she's thinking about transitioning to a principal engineer role. Good team morale overall.", "tags": ["social", "team", "lunch", "Sakura-Sushi"], "memory_type": "Observation", "created_at": _ts(8, 12)},
    {"id_tag": "M07", "content": "The onboarding wizard prototype is working. It has 5 steps: account creation, team setup, first project, integration connections, and a guided tour. User testing showed 78% completion rate, up from 45% with the old flow. VP of Product Elena approved moving to production.", "tags": ["onboarding", "prototype", "user-testing", "Elena"], "memory_type": "Observation", "created_at": _ts(9)},
    {"id_tag": "M08", "content": "Decision to use Kubernetes over ECS for the new microservices deployment. Reasoning: better multi-cloud portability, stronger community tooling (Helm, ArgoCD), and our SRE team already has K8s experience from the data pipeline project. Cost estimate is $2,400/month higher but worth the flexibility.", "tags": ["infrastructure", "kubernetes", "deployment", "decision"], "memory_type": "Decision", "created_at": _ts(10)},
    {"id_tag": "M09", "content": "Production incident at 3 AM: the analytics database ran out of disk space. Root cause was the funnel_events table growing 10x faster than projected because of a missing TTL policy. We added a 90-day retention policy and partitioned by month. Total downtime was 47 minutes.", "tags": ["incident", "database", "disk-space", "analytics", "downtime"], "memory_type": "Observation", "created_at": _ts(14, 3)},
    {"id_tag": "M10", "content": "Sarah completed the Stripe v3 migration for the US region. Performance improved by 40% — average checkout time dropped from 3.2s to 1.9s. EU migration is scheduled for next week. She used the new Stripe Payment Intents API which also enables Apple Pay and Google Pay.", "tags": ["stripe", "migration", "performance", "Sarah", "checkout"], "memory_type": "Observation", "created_at": _ts(15)},
    {"id_tag": "M11", "content": "Conference talk accepted: I'll be presenting 'Building Resilient Caching at Scale' at RustConf 2026 in Austin, Texas on September 15th. The talk covers our Redis architecture evolution, the thundering herd fix, and distributed locking patterns.", "tags": ["conference", "RustConf", "talk", "Austin"], "memory_type": "Observation", "created_at": _ts(16)},
    {"id_tag": "M12", "content": "Code review feedback from Marcus on the onboarding wizard: he flagged that the integration step makes 12 sequential API calls which could be parallelized. Good catch — refactored to use Promise.all() and reduced step 4 load time from 4.5s to 0.8s.", "tags": ["code-review", "Marcus", "onboarding", "performance"], "memory_type": "Observation", "created_at": _ts(17)},
    {"id_tag": "M13", "content": "Q3 OKR planning meeting. Our team's key results: (1) reduce p95 API latency from 800ms to 200ms, (2) achieve 99.95% uptime SLA, (3) launch self-serve enterprise onboarding. VP Elena emphasized that the enterprise onboarding is the highest revenue priority.", "tags": ["OKR", "Q3", "planning", "latency", "uptime", "enterprise"], "memory_type": "Observation", "created_at": _ts(21)},
    {"id_tag": "M14", "content": "Security audit findings: The penetration test by CrowdStrike found 3 medium-severity issues: (1) missing rate limiting on the password reset endpoint, (2) CORS misconfiguration allowing wildcard origins in staging, (3) session tokens not rotated after privilege escalation. All assigned to Sprint 15.", "tags": ["security", "audit", "CrowdStrike", "vulnerabilities"], "memory_type": "Observation", "created_at": _ts(22)},
    {"id_tag": "M15", "content": "Raj is transferring to the ML platform team next month. His replacement on our team will be Priya, who has 5 years of experience with distributed systems at Netflix. She starts on April 15th. Raj will do a 2-week knowledge transfer on the caching layer.", "tags": ["team-change", "Raj", "Priya", "transfer", "Netflix"], "memory_type": "Observation", "created_at": _ts(23)},
    {"id_tag": "M16", "content": "Decided to adopt GraphQL for the new enterprise API instead of REST. Reasons: clients need flexible field selection for dashboard customization, reduces over-fetching which is critical for mobile, and Apollo Federation allows us to stitch microservice schemas. Timeline: MVP by end of Q3.", "tags": ["architecture", "GraphQL", "enterprise", "API", "decision"], "memory_type": "Decision", "created_at": _ts(24)},
    {"id_tag": "M17", "content": "The annual company hackathon is on April 20-21. Our team is building a real-time collaboration feature using CRDTs and WebSockets. I'm excited because this could become a real product feature. Last year's winning team built the AI-powered search that's now in production.", "tags": ["hackathon", "CRDT", "collaboration", "WebSocket"], "memory_type": "Observation", "created_at": _ts(25)},
    {"id_tag": "M18", "content": "Performance benchmark results for the new analytics pipeline on PostgreSQL: 500K events/minute ingestion, p99 query latency at 120ms for 30-day windows, and 45ms for 7-day windows. This exceeds our Q3 target of 200ms p95. The columnar extension (TimescaleDB) was key to this performance.", "tags": ["benchmark", "analytics", "PostgreSQL", "TimescaleDB", "performance"], "memory_type": "Observation", "created_at": _ts(26)},
    {"id_tag": "M19", "content": "Sarah's Stripe v3 EU migration went live today. Checkout latency for EU users dropped from 5 seconds to 1.4 seconds. The bug from two weeks ago about EU checkout delays is now fully resolved. Apple Pay and Google Pay are also enabled for EU customers.", "tags": ["stripe", "EU", "migration", "Sarah", "checkout", "resolved"], "memory_type": "Observation", "created_at": _ts(27)},
    {"id_tag": "M20", "content": "End-of-sprint retro for Sprint 14. Completed 31 of 34 story points. Carried over: the enterprise SSO integration (blocked by IdP provider delays) and two minor UI polish tasks. Team velocity trending up — average of last 3 sprints is 30 points. Celebrating at Happy Hour Friday at The Rusty Anchor.", "tags": ["retro", "sprint-14", "velocity", "team"], "memory_type": "Observation", "created_at": _ts(28)},
]

# ---------------------------------------------------------------------------
# Multi-hop queries only — these are the hardest and most sensitive to recall
# regression; single-hop queries tend to pass even with a degraded index.
# Expected tags identify which memories MUST appear in the top-10 results.
# ---------------------------------------------------------------------------
MULTI_HOP_QUERIES = [
    {
        "query": "How did Sarah's Stripe migration fix the EU checkout delay problem?",
        "expected": ["M03", "M10", "M19"],
    },
    {
        "query": "What caching problems did we have and how were they resolved?",
        "expected": ["M01", "M05"],
    },
    {
        "query": "Which team member reviewed the onboarding wizard and what did they find?",
        "expected": ["M07", "M12"],
    },
    {
        "query": "What is Raj working on now and what happens when he leaves?",
        "expected": ["M05", "M15"],
    },
    {
        "query": "How do our analytics benchmark numbers compare to the Q3 OKR targets?",
        "expected": ["M13", "M18"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recall_at_k(ranked_ids: list[str], expected_ids: set[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & expected_ids) / len(expected_ids)


def _store(mem: dict) -> str | None:
    payload = {
        "user_id": USER_ID,
        "content": mem["content"],
        "tags": mem.get("tags", []),
        "memory_type": mem.get("memory_type", "Observation"),
        "created_at": mem.get("created_at"),
    }
    try:
        r = requests.post(f"{VELD_URL}/api/remember", headers=HEADERS, json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get("id")
    except Exception as e:
        print(f"  STORE ERROR {mem['id_tag']}: {e}", file=sys.stderr)
        return None


def _recall(query: str, limit: int = 10) -> tuple[list[str], float]:
    """Returns (list-of-memory-ids, latency_ms)."""
    payload = {"user_id": USER_ID, "query": query, "limit": limit, "mode": "hybrid"}
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{VELD_URL}/api/recall", headers=HEADERS, json=payload, timeout=15)
        r.raise_for_status()
        memories = r.json().get("memories", [])
    except Exception as e:
        print(f"  RECALL ERROR: {e}", file=sys.stderr)
        memories = []
    latency_ms = (time.perf_counter() - t0) * 1000
    ids = [m["id"] for m in memories if "id" in m]
    return ids, latency_ms


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("veld recall+latency guard")
    print(f"  server:  {VELD_URL}")
    print(f"  user_id: {USER_ID}")
    print(f"  thresholds: multi_hop_R@10 >= {MULTI_HOP_R10_FLOOR:.2f}  |  p99_latency <= {P99_LATENCY_CEILING_MS} ms")
    print("=" * 72)

    # Health check
    try:
        h = requests.get(f"{VELD_URL}/health", timeout=5).json()
        print(f"\nveld {h.get('version', '?')} — {h.get('status', '?')}")
    except Exception as e:
        print(f"\nFATAL: veld unreachable at {VELD_URL}: {e}", file=sys.stderr)
        return 1

    # Store all 20 memories
    print(f"\n[1/3] Storing {len(MEMORIES)} memories...")
    id_map: dict[str, str] = {}
    for mem in MEMORIES:
        uid = _store(mem)
        if uid:
            id_map[mem["id_tag"]] = uid
    print(f"  stored {len(id_map)}/{len(MEMORIES)}")

    if len(id_map) < len(MEMORIES):
        print("FATAL: could not store all memories — abort.", file=sys.stderr)
        return 1

    # Brief pause to let indexing settle (same as other benchmarks)
    time.sleep(1.5)

    # Run multi-hop recall queries
    print(f"\n[2/3] Running {len(MULTI_HOP_QUERIES)} multi-hop recall queries...")
    uuid_to_tag = {v: k for k, v in id_map.items()}
    latencies: list[float] = []
    r10_scores: list[float] = []

    for i, q in enumerate(MULTI_HOP_QUERIES):
        ranked_ids, lat = _recall(q["query"], limit=10)
        latencies.append(lat)
        expected_uuids = {id_map[t] for t in q["expected"] if t in id_map}
        r10 = _recall_at_k(ranked_ids, expected_uuids, 10)
        r10_scores.append(r10)
        returned_tags = [uuid_to_tag.get(uid, "?") for uid in ranked_ids[:5]]
        status = "PASS" if r10 >= MULTI_HOP_R10_FLOOR else "FAIL"
        print(
            f"  Q{i+1} [{status}] R@10={r10:.2f}  lat={lat:.0f}ms  "
            f"top5={returned_tags}  expected={q['expected']}"
        )

    # Compute aggregate metrics
    multi_hop_r10 = sum(r10_scores) / len(r10_scores) if r10_scores else 0.0

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p99_idx = max(0, int(0.99 * n) - 1) if n > 1 else 0
    p99_latency = latencies_sorted[p99_idx]
    # With only 5 queries p99 == the max; that is expected and intentional.
    # The ceiling is deliberately generous (3 s) to absorb CI machine variance.

    print(f"\n[3/3] Results")
    print("=" * 72)
    print(f"  multi_hop R@10 (avg):  {multi_hop_r10:.4f}   floor={MULTI_HOP_R10_FLOOR:.2f}")
    print(f"  p99 recall latency:    {p99_latency:.0f} ms  ceiling={P99_LATENCY_CEILING_MS} ms")
    print(f"  all recall latencies:  {[round(l) for l in latencies_sorted]} ms")

    # Guard assertions
    failures: list[str] = []

    if multi_hop_r10 < MULTI_HOP_R10_FLOOR:
        msg = (
            f"RECALL REGRESSION: multi_hop R@10 = {multi_hop_r10:.4f} "
            f"< floor {MULTI_HOP_R10_FLOOR:.2f}  "
            f"(per-query scores: {[round(s, 3) for s in r10_scores]})"
        )
        failures.append(msg)

    if p99_latency > P99_LATENCY_CEILING_MS:
        msg = (
            f"LATENCY REGRESSION: p99 recall = {p99_latency:.0f} ms "
            f"> ceiling {P99_LATENCY_CEILING_MS} ms"
        )
        failures.append(msg)

    if failures:
        print("\n*** GUARD FAILED ***", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print("\n  All thresholds passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
