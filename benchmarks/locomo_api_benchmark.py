#!/usr/bin/env python3
"""
LOCOMO-style benchmark against the Veld API.

Stores 20 diverse conversational memories simulating multi-session project work,
then runs 20 retrieval queries across 4 categories:
  - Single-hop (direct factual recall)
  - Temporal (time-based reasoning)
  - Multi-hop (requires connecting multiple memories)
  - Open-domain (broader reasoning, preference, opinion)

Reports per-category MRR, R@5, R@10, and overall MRR.
"""

import json
import sys
import time
import requests
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:3030"
API_KEY = "dev-key-antidote"
USER_ID = f"locomo_bench_{int(time.time())}"
HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}

# ---------------------------------------------------------------------------
# Memories — 20 diverse entries spread across weeks
# ---------------------------------------------------------------------------
# base_time is ~4 weeks ago
BASE_TIME = datetime.now(timezone.utc) - timedelta(weeks=4)


def ts(days_offset: int, hours: int = 10) -> str:
    """Create ISO timestamp offset from BASE_TIME."""
    return (BASE_TIME + timedelta(days=days_offset, hours=hours)).isoformat()


MEMORIES = [
    # --- Sprint planning & project management ---
    {
        "id_tag": "M01",
        "content": "Sprint 14 planning: We committed to 34 story points. Key items are the payment gateway migration from Stripe v2 to v3, the Redis cache invalidation bug, and the new onboarding wizard. Sarah is leading the payment work, Raj is on caching, and I'm handling onboarding.",
        "tags": ["sprint", "planning", "sprint-14"],
        "memory_type": "Observation",
        "created_at": ts(0),
        "source_type": "user",
        "emotional_valence": 0.3,
        "emotional_arousal": 0.4,
    },
    {
        "id_tag": "M02",
        "content": "Architecture decision: We chose PostgreSQL over MongoDB for the analytics pipeline. Key reasons were complex joins needed for funnel analysis, ACID compliance for financial data, and the team's existing expertise. Marcus strongly advocated for Mongo but the vote was 4-1 in favor of Postgres.",
        "tags": ["architecture", "database", "decision", "analytics"],
        "memory_type": "Decision",
        "created_at": ts(1),
        "source_type": "user",
        "emotional_valence": 0.2,
        "emotional_arousal": 0.3,
    },
    {
        "id_tag": "M03",
        "content": "Bug report: Users in the EU region are experiencing 5-second delays on the checkout page. The root cause is the payment provider's EU endpoint having high latency. Temporary fix: route EU traffic through the UK proxy. Permanent fix needs Stripe v3 migration which Sarah is working on.",
        "tags": ["bug", "performance", "checkout", "EU", "latency"],
        "memory_type": "Observation",
        "created_at": ts(2),
        "source_type": "system",
        "emotional_valence": -0.5,
        "emotional_arousal": 0.7,
    },
    {
        "id_tag": "M04",
        "content": "Personal preference: I strongly prefer dark mode in all development tools. My IDE theme is Dracula, terminal is Catppuccin Mocha, and I use the Fira Code font with ligatures enabled at 14pt.",
        "tags": ["preference", "tooling", "IDE", "theme"],
        "memory_type": "Preference",
        "created_at": ts(3),
        "source_type": "user",
        "emotional_valence": 0.6,
        "emotional_arousal": 0.2,
    },
    # --- Week 2: Development progress ---
    {
        "id_tag": "M05",
        "content": "Raj found the Redis cache invalidation bug. It was a race condition in the pub/sub listener — when two nodes receive the same invalidation event, they both try to refresh the cache simultaneously, causing a thundering herd. Fix: added distributed locking with Redlock.",
        "tags": ["bug-fix", "redis", "cache", "race-condition", "Raj"],
        "memory_type": "Observation",
        "created_at": ts(7),
        "source_type": "user",
        "emotional_valence": 0.5,
        "emotional_arousal": 0.5,
    },
    {
        "id_tag": "M06",
        "content": "Team lunch at Sakura Sushi near the downtown office. Everyone was there except Marcus who was working remotely from Portland. Sarah mentioned she's thinking about transitioning to a principal engineer role. Good team morale overall.",
        "tags": ["social", "team", "lunch", "Sakura-Sushi"],
        "memory_type": "Observation",
        "created_at": ts(8, 12),
        "source_type": "user",
        "emotional_valence": 0.7,
        "emotional_arousal": 0.3,
    },
    {
        "id_tag": "M07",
        "content": "The onboarding wizard prototype is working. It has 5 steps: account creation, team setup, first project, integration connections, and a guided tour. User testing showed 78% completion rate, up from 45% with the old flow. VP of Product Elena approved moving to production.",
        "tags": ["onboarding", "prototype", "user-testing", "Elena"],
        "memory_type": "Observation",
        "created_at": ts(9),
        "source_type": "user",
        "emotional_valence": 0.6,
        "emotional_arousal": 0.5,
    },
    {
        "id_tag": "M08",
        "content": "Decision to use Kubernetes over ECS for the new microservices deployment. Reasoning: better multi-cloud portability, stronger community tooling (Helm, ArgoCD), and our SRE team already has K8s experience from the data pipeline project. Cost estimate is $2,400/month higher but worth the flexibility.",
        "tags": ["infrastructure", "kubernetes", "deployment", "decision"],
        "memory_type": "Decision",
        "created_at": ts(10),
        "source_type": "user",
        "emotional_valence": 0.1,
        "emotional_arousal": 0.3,
    },
    # --- Week 3: Incidents & resolutions ---
    {
        "id_tag": "M09",
        "content": "Production incident at 3 AM: the analytics database ran out of disk space. Root cause was the funnel_events table growing 10x faster than projected because of a missing TTL policy. We added a 90-day retention policy and partitioned by month. Total downtime was 47 minutes.",
        "tags": ["incident", "database", "disk-space", "analytics", "downtime"],
        "memory_type": "Observation",
        "created_at": ts(14, 3),
        "source_type": "system",
        "emotional_valence": -0.7,
        "emotional_arousal": 0.9,
    },
    {
        "id_tag": "M10",
        "content": "Sarah completed the Stripe v3 migration for the US region. Performance improved by 40% — average checkout time dropped from 3.2s to 1.9s. EU migration is scheduled for next week. She used the new Stripe Payment Intents API which also enables Apple Pay and Google Pay.",
        "tags": ["stripe", "migration", "performance", "Sarah", "checkout"],
        "memory_type": "Observation",
        "created_at": ts(15),
        "source_type": "user",
        "emotional_valence": 0.7,
        "emotional_arousal": 0.5,
    },
    {
        "id_tag": "M11",
        "content": "Conference talk accepted: I'll be presenting 'Building Resilient Caching at Scale' at RustConf 2026 in Austin, Texas on September 15th. The talk covers our Redis architecture evolution, the thundering herd fix, and distributed locking patterns.",
        "tags": ["conference", "RustConf", "talk", "Austin"],
        "memory_type": "Observation",
        "created_at": ts(16),
        "source_type": "user",
        "emotional_valence": 0.8,
        "emotional_arousal": 0.7,
    },
    {
        "id_tag": "M12",
        "content": "Code review feedback from Marcus on the onboarding wizard: he flagged that the integration step makes 12 sequential API calls which could be parallelized. Good catch — refactored to use Promise.all() and reduced step 4 load time from 4.5s to 0.8s.",
        "tags": ["code-review", "Marcus", "onboarding", "performance"],
        "memory_type": "Observation",
        "created_at": ts(17),
        "source_type": "user",
        "emotional_valence": 0.4,
        "emotional_arousal": 0.3,
    },
    # --- Week 4: Strategy & planning ---
    {
        "id_tag": "M13",
        "content": "Q3 OKR planning meeting. Our team's key results: (1) reduce p95 API latency from 800ms to 200ms, (2) achieve 99.95% uptime SLA, (3) launch self-serve enterprise onboarding. VP Elena emphasized that the enterprise onboarding is the highest revenue priority.",
        "tags": ["OKR", "Q3", "planning", "latency", "uptime", "enterprise"],
        "memory_type": "Observation",
        "created_at": ts(21),
        "source_type": "user",
        "emotional_valence": 0.2,
        "emotional_arousal": 0.4,
    },
    {
        "id_tag": "M14",
        "content": "Security audit findings: The penetration test by CrowdStrike found 3 medium-severity issues: (1) missing rate limiting on the password reset endpoint, (2) CORS misconfiguration allowing wildcard origins in staging, (3) session tokens not rotated after privilege escalation. All assigned to Sprint 15.",
        "tags": ["security", "audit", "CrowdStrike", "vulnerabilities"],
        "memory_type": "Observation",
        "created_at": ts(22),
        "source_type": "system",
        "emotional_valence": -0.3,
        "emotional_arousal": 0.6,
    },
    {
        "id_tag": "M15",
        "content": "Raj is transferring to the ML platform team next month. His replacement on our team will be Priya, who has 5 years of experience with distributed systems at Netflix. She starts on April 15th. Raj will do a 2-week knowledge transfer on the caching layer.",
        "tags": ["team-change", "Raj", "Priya", "transfer", "Netflix"],
        "memory_type": "Observation",
        "created_at": ts(23),
        "source_type": "user",
        "emotional_valence": 0.0,
        "emotional_arousal": 0.4,
    },
    {
        "id_tag": "M16",
        "content": "Decided to adopt GraphQL for the new enterprise API instead of REST. Reasons: clients need flexible field selection for dashboard customization, reduces over-fetching which is critical for mobile, and Apollo Federation allows us to stitch microservice schemas. Timeline: MVP by end of Q3.",
        "tags": ["architecture", "GraphQL", "enterprise", "API", "decision"],
        "memory_type": "Decision",
        "created_at": ts(24),
        "source_type": "user",
        "emotional_valence": 0.3,
        "emotional_arousal": 0.3,
    },
    {
        "id_tag": "M17",
        "content": "The annual company hackathon is on April 20-21. Our team is building a real-time collaboration feature using CRDTs and WebSockets. I'm excited because this could become a real product feature. Last year's winning team built the AI-powered search that's now in production.",
        "tags": ["hackathon", "CRDT", "collaboration", "WebSocket"],
        "memory_type": "Observation",
        "created_at": ts(25),
        "source_type": "user",
        "emotional_valence": 0.7,
        "emotional_arousal": 0.6,
    },
    {
        "id_tag": "M18",
        "content": "Performance benchmark results for the new analytics pipeline on PostgreSQL: 500K events/minute ingestion, p99 query latency at 120ms for 30-day windows, and 45ms for 7-day windows. This exceeds our Q3 target of 200ms p95. The columnar extension (TimescaleDB) was key to this performance.",
        "tags": ["benchmark", "analytics", "PostgreSQL", "TimescaleDB", "performance"],
        "memory_type": "Observation",
        "created_at": ts(26),
        "source_type": "system",
        "emotional_valence": 0.6,
        "emotional_arousal": 0.4,
    },
    {
        "id_tag": "M19",
        "content": "Sarah's Stripe v3 EU migration went live today. Checkout latency for EU users dropped from 5 seconds to 1.4 seconds. The bug from two weeks ago about EU checkout delays is now fully resolved. Apple Pay and Google Pay are also enabled for EU customers.",
        "tags": ["stripe", "EU", "migration", "Sarah", "checkout", "resolved"],
        "memory_type": "Observation",
        "created_at": ts(27),
        "source_type": "user",
        "emotional_valence": 0.8,
        "emotional_arousal": 0.5,
    },
    {
        "id_tag": "M20",
        "content": "End-of-sprint retro for Sprint 14. Completed 31 of 34 story points. Carried over: the enterprise SSO integration (blocked by IdP provider delays) and two minor UI polish tasks. Team velocity trending up — average of last 3 sprints is 30 points. Celebrating at Happy Hour Friday at The Rusty Anchor.",
        "tags": ["retro", "sprint-14", "velocity", "team"],
        "memory_type": "Observation",
        "created_at": ts(28),
        "source_type": "user",
        "emotional_valence": 0.5,
        "emotional_arousal": 0.3,
    },
]

# ---------------------------------------------------------------------------
# Queries — 20 queries across 4 categories (5 each)
# Each maps to one or more expected memory id_tags
# ---------------------------------------------------------------------------
QUERIES = [
    # ========= SINGLE-HOP (direct factual recall) =========
    {
        "category": "single-hop",
        "query": "What database did we choose for the analytics pipeline?",
        "expected": ["M02"],
        "rationale": "Direct recall of the PostgreSQL decision",
    },
    {
        "category": "single-hop",
        "query": "What was the root cause of the Redis cache invalidation bug?",
        "expected": ["M05"],
        "rationale": "Direct recall of the race condition finding",
    },
    {
        "category": "single-hop",
        "query": "What is my preferred IDE theme and font?",
        "expected": ["M04"],
        "rationale": "Direct recall of personal preferences",
    },
    {
        "category": "single-hop",
        "query": "How many story points did we commit to in Sprint 14?",
        "expected": ["M01"],
        "rationale": "Direct factual recall from sprint planning",
    },
    {
        "category": "single-hop",
        "query": "What were the security vulnerabilities found in the penetration test?",
        "expected": ["M14"],
        "rationale": "Direct recall of CrowdStrike audit findings",
    },
    # ========= TEMPORAL (time-based reasoning) =========
    {
        "category": "temporal",
        "query": "What happened during the production incident that caused downtime?",
        "expected": ["M09"],
        "rationale": "Recall of the 3AM disk space incident",
    },
    {
        "category": "temporal",
        "query": "When is my conference talk and what is it about?",
        "expected": ["M11"],
        "rationale": "Temporal event recall — RustConf September 15th",
    },
    {
        "category": "temporal",
        "query": "When does Priya start and who is she replacing?",
        "expected": ["M15"],
        "rationale": "Temporal recall of team change timing",
    },
    {
        "category": "temporal",
        "query": "What are the dates for the company hackathon?",
        "expected": ["M17"],
        "rationale": "Temporal event recall — April 20-21 hackathon",
    },
    {
        "category": "temporal",
        "query": "When was the EU checkout latency issue finally resolved?",
        "expected": ["M19"],
        "rationale": "Temporal resolution — Sarah's EU migration going live",
    },
    # ========= MULTI-HOP (connecting multiple memories) =========
    {
        "category": "multi-hop",
        "query": "How did Sarah's Stripe migration fix the EU checkout delay problem?",
        "expected": ["M03", "M10", "M19"],
        "rationale": "Connects: EU delay bug (M03) -> US migration (M10) -> EU migration (M19)",
    },
    {
        "category": "multi-hop",
        "query": "What caching problems did we have and how were they resolved?",
        "expected": ["M01", "M05"],
        "rationale": "Connects: sprint planning mentioning cache bug (M01) -> Raj's fix (M05)",
    },
    {
        "category": "multi-hop",
        "query": "Which team member reviewed the onboarding wizard and what did they find?",
        "expected": ["M07", "M12"],
        "rationale": "Connects: onboarding prototype (M07) -> Marcus's code review (M12)",
    },
    {
        "category": "multi-hop",
        "query": "What is Raj working on now and what happens when he leaves?",
        "expected": ["M05", "M15"],
        "rationale": "Connects: Raj's current work on caching (M05) -> his transfer to ML team (M15)",
    },
    {
        "category": "multi-hop",
        "query": "How do our analytics benchmark numbers compare to the Q3 OKR targets?",
        "expected": ["M13", "M18"],
        "rationale": "Connects: Q3 OKR latency target 200ms (M13) -> benchmark showing 120ms (M18)",
    },
    # ========= OPEN-DOMAIN (broader reasoning) =========
    {
        "category": "open-domain",
        "query": "What is the team's overall morale and social dynamics like?",
        "expected": ["M06", "M20"],
        "rationale": "Broader inference from social events and retro celebrations",
    },
    {
        "category": "open-domain",
        "query": "What are the most important strategic priorities for the team?",
        "expected": ["M13", "M16"],
        "rationale": "Strategic reasoning from OKRs and enterprise API decisions",
    },
    {
        "category": "open-domain",
        "query": "What infrastructure and deployment decisions have we made recently?",
        "expected": ["M02", "M08", "M16"],
        "rationale": "Aggregate infrastructure decisions: Postgres, K8s, GraphQL",
    },
    {
        "category": "open-domain",
        "query": "What performance improvements have we achieved across the product?",
        "expected": ["M05", "M07", "M10", "M12", "M18", "M19"],
        "rationale": "Aggregate performance wins across multiple workstreams",
    },
    {
        "category": "open-domain",
        "query": "Who are the key people on the team and what are they responsible for?",
        "expected": ["M01", "M05", "M06", "M10", "M15"],
        "rationale": "People-centric reasoning: Sarah (payments), Raj (caching), Marcus, Priya",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def store_memory(mem: dict) -> str | None:
    """Store a single memory and return its ID."""
    payload = {
        "user_id": USER_ID,
        "content": mem["content"],
        "tags": mem.get("tags", []),
        "memory_type": mem.get("memory_type", "Observation"),
        "source_type": mem.get("source_type", "user"),
    }
    if mem.get("created_at"):
        payload["created_at"] = mem["created_at"]
    if mem.get("emotional_valence") is not None:
        payload["emotional_valence"] = mem["emotional_valence"]
    if mem.get("emotional_arousal") is not None:
        payload["emotional_arousal"] = mem["emotional_arousal"]

    resp = requests.post(f"{BASE_URL}/api/remember", headers=HEADERS, json=payload)
    if resp.status_code != 200:
        print(f"  ERROR storing {mem['id_tag']}: {resp.status_code} {resp.text}")
        return None
    data = resp.json()
    return data.get("id")


def recall_memories(query: str, limit: int = 10) -> list[dict]:
    """Recall memories for a query."""
    payload = {
        "user_id": USER_ID,
        "query": query,
        "limit": limit,
        "mode": "hybrid",
    }
    resp = requests.post(f"{BASE_URL}/api/recall", headers=HEADERS, json=payload)
    if resp.status_code != 200:
        print(f"  ERROR recalling: {resp.status_code} {resp.text}")
        return []
    data = resp.json()
    return data.get("memories", [])


def compute_mrr(ranked_ids: list[str], expected_ids: set[str]) -> float:
    """Compute Mean Reciprocal Rank — rank of first correct result."""
    for i, rid in enumerate(ranked_ids):
        if rid in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(ranked_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Compute Recall@K — fraction of expected items found in top K."""
    top_k = set(ranked_ids[:k])
    if not expected_ids:
        return 0.0
    return len(top_k & expected_ids) / len(expected_ids)


def compute_hit_at_k(ranked_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Compute Hit@K — 1 if any expected item in top K, else 0."""
    top_k = set(ranked_ids[:k])
    return 1.0 if top_k & expected_ids else 0.0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("LOCOMO-STYLE BENCHMARK — Veld API")
    print(f"Server: {BASE_URL}")
    print(f"User:   {USER_ID}")
    print(f"Time:   {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)

    # ── Step 1: Health check ──
    print("\n[1/4] Health check...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        health = resp.json()
        print(f"  Status: {health.get('status')} | Version: {health.get('version')}")
    except Exception as e:
        print(f"  FATAL: Cannot reach server: {e}")
        sys.exit(1)

    # ── Step 2: Store memories ──
    print(f"\n[2/4] Storing {len(MEMORIES)} memories...")
    id_map: dict[str, str] = {}  # id_tag -> server UUID
    for mem in MEMORIES:
        uuid = store_memory(mem)
        if uuid:
            id_map[mem["id_tag"]] = uuid
            print(f"  {mem['id_tag']} -> {uuid[:12]}... ({mem['content'][:50]}...)")
        else:
            print(f"  {mem['id_tag']} -> FAILED")

    if len(id_map) < len(MEMORIES):
        print(f"\n  WARNING: Only {len(id_map)}/{len(MEMORIES)} memories stored successfully.")

    # Small delay to allow indexing
    time.sleep(1.5)

    # ── Step 3: Run queries ──
    print(f"\n[3/4] Running {len(QUERIES)} queries...")

    # Reverse map: server UUID -> id_tag
    uuid_to_tag = {v: k for k, v in id_map.items()}

    results = []
    for i, q in enumerate(QUERIES):
        memories = recall_memories(q["query"], limit=10)
        ranked_uuids = [m["id"] for m in memories]
        ranked_tags = [uuid_to_tag.get(uid, "?") for uid in ranked_uuids]

        expected_uuids = {id_map[t] for t in q["expected"] if t in id_map}
        expected_tags = set(q["expected"])

        mrr = compute_mrr(ranked_uuids, expected_uuids)
        hit5 = compute_hit_at_k(ranked_uuids, expected_uuids, 5)
        hit10 = compute_hit_at_k(ranked_uuids, expected_uuids, 10)
        recall5 = compute_recall_at_k(ranked_uuids, expected_uuids, 5)
        recall10 = compute_recall_at_k(ranked_uuids, expected_uuids, 10)

        result = {
            "index": i + 1,
            "category": q["category"],
            "query": q["query"],
            "expected_tags": list(expected_tags),
            "returned_tags": ranked_tags,
            "returned_scores": [m.get("score", 0) for m in memories],
            "mrr": mrr,
            "hit_at_5": hit5,
            "hit_at_10": hit10,
            "recall_at_5": recall5,
            "recall_at_10": recall10,
        }
        results.append(result)

        # Print inline
        status = "HIT" if mrr > 0 else "MISS"
        found_tags = [t for t in ranked_tags if t in expected_tags]
        print(
            f"  Q{i+1:02d} [{q['category']:11s}] MRR={mrr:.3f} R@5={recall5:.2f} R@10={recall10:.2f} "
            f"{status:4s} | expected={q['expected']} found={found_tags} "
            f"| top3={ranked_tags[:3]}"
        )

    # ── Step 4: Compute and report metrics ──
    print(f"\n[4/4] Results")
    print("=" * 72)

    categories = ["single-hop", "temporal", "multi-hop", "open-domain"]
    all_mrr = []
    all_hit5 = []
    all_hit10 = []
    all_recall5 = []
    all_recall10 = []

    print(f"\n{'Category':<14s} {'MRR':>7s} {'Hit@5':>7s} {'Hit@10':>7s} {'R@5':>7s} {'R@10':>7s}  n")
    print("-" * 72)

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        cat_mrr = sum(r["mrr"] for r in cat_results) / n if n else 0
        cat_hit5 = sum(r["hit_at_5"] for r in cat_results) / n if n else 0
        cat_hit10 = sum(r["hit_at_10"] for r in cat_results) / n if n else 0
        cat_r5 = sum(r["recall_at_5"] for r in cat_results) / n if n else 0
        cat_r10 = sum(r["recall_at_10"] for r in cat_results) / n if n else 0

        print(f"  {cat:<12s} {cat_mrr:>7.4f} {cat_hit5:>7.4f} {cat_hit10:>7.4f} {cat_r5:>7.4f} {cat_r10:>7.4f}  {n}")

        all_mrr.extend(r["mrr"] for r in cat_results)
        all_hit5.extend(r["hit_at_5"] for r in cat_results)
        all_hit10.extend(r["hit_at_10"] for r in cat_results)
        all_recall5.extend(r["recall_at_5"] for r in cat_results)
        all_recall10.extend(r["recall_at_10"] for r in cat_results)

    n_total = len(results)
    overall_mrr = sum(all_mrr) / n_total if n_total else 0
    overall_hit5 = sum(all_hit5) / n_total if n_total else 0
    overall_hit10 = sum(all_hit10) / n_total if n_total else 0
    overall_r5 = sum(all_recall5) / n_total if n_total else 0
    overall_r10 = sum(all_recall10) / n_total if n_total else 0

    print("-" * 72)
    print(f"  {'OVERALL':<12s} {overall_mrr:>7.4f} {overall_hit5:>7.4f} {overall_hit10:>7.4f} {overall_r5:>7.4f} {overall_r10:>7.4f}  {n_total}")
    print()

    # Detailed failure analysis
    misses = [r for r in results if r["mrr"] == 0.0]
    if misses:
        print(f"MISSES ({len(misses)}):")
        for r in misses:
            print(f"  Q{r['index']:02d} [{r['category']}] {r['query'][:60]}")
            print(f"       expected={r['expected_tags']} returned={r['returned_tags'][:5]}")
    else:
        print("No misses — all queries found at least one expected memory in top 10.")

    # Partial hits for multi-expected queries
    partials = [
        r for r in results
        if 0 < r["recall_at_10"] < 1.0 and len(r["expected_tags"]) > 1
    ]
    if partials:
        print(f"\nPARTIAL HITS ({len(partials)} multi-hop/open-domain queries with incomplete recall):")
        for r in partials:
            found_in_10 = [t for t in r["returned_tags"][:10] if t in r["expected_tags"]]
            missed = [t for t in r["expected_tags"] if t not in r["returned_tags"][:10]]
            print(f"  Q{r['index']:02d} [{r['category']}] R@10={r['recall_at_10']:.2f}")
            print(f"       found={found_in_10} missed={missed}")

    print("\n" + "=" * 72)
    print(f"Benchmark complete. User ID: {USER_ID}")
    print(f"Overall MRR: {overall_mrr:.4f} | Hit@5: {overall_hit5:.4f} | Hit@10: {overall_hit10:.4f}")
    print("=" * 72)

    # Save raw results as JSON
    output = {
        "benchmark": "locomo-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_version": health.get("version"),
        "user_id": USER_ID,
        "n_memories": len(id_map),
        "n_queries": len(QUERIES),
        "metrics": {
            "overall_mrr": overall_mrr,
            "overall_hit_at_5": overall_hit5,
            "overall_hit_at_10": overall_hit10,
            "overall_recall_at_5": overall_r5,
            "overall_recall_at_10": overall_r10,
        },
        "per_category": {},
        "results": results,
    }
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        output["per_category"][cat] = {
            "mrr": sum(r["mrr"] for r in cat_results) / n if n else 0,
            "hit_at_5": sum(r["hit_at_5"] for r in cat_results) / n if n else 0,
            "hit_at_10": sum(r["hit_at_10"] for r in cat_results) / n if n else 0,
            "recall_at_5": sum(r["recall_at_5"] for r in cat_results) / n if n else 0,
            "recall_at_10": sum(r["recall_at_10"] for r in cat_results) / n if n else 0,
            "n": n,
        }

    output_path = f"benchmarks/locomo_api_{USER_ID}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")


if __name__ == "__main__":
    main()
