#!/usr/bin/env python3
"""
LOCOMO LLM-judged evaluation for veld.

Stores 20 conversational memories, runs 20 retrieval queries, then uses
a local LLM (QwQ on LM Studio) to judge whether the retrieved context
is sufficient to correctly answer each question.

Scoring rubric (per query):
  3 = fully answerable from retrieved context
  2 = partially answerable (key info present but incomplete)
  1 = tangentially relevant (related but can't answer)
  0 = not answerable (irrelevant or missing context)

Reports per-category and overall scores alongside retrieval metrics.
"""

import json
import re
import sys
import time
import hashlib
import subprocess
import requests
from datetime import datetime, timedelta, timezone

from eval_metrics import (
    compute_mrr,
    compute_recall_at_k,
    compute_memory_age_days,
    classify_freshness_band,
    FRESHNESS_BANDS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
import os

VELD_URL = os.environ.get("VELD_URL", "http://127.0.0.1:3030")
VELD_API_KEY = os.environ.get("VELD_API_KEY", "dev-key-local")
# Judge LLM — defaults to local LM Studio (qwen/qwq-32b); env vars switch to
# a hosted OpenAI-compatible endpoint such as Together AI.
JUDGE_URL = os.environ.get("VELD_JUDGE_URL", "http://localhost:1234/v1")
JUDGE_MODEL = os.environ.get("VELD_JUDGE_MODEL", "qwen/qwq-32b")
JUDGE_API_KEY = os.environ.get("VELD_JUDGE_API_KEY")  # None for LM Studio
# Keep the old name for any downstream references that still read it.
LM_STUDIO_URL = JUDGE_URL
USER_ID = f"locomo_judge_{int(time.time())}"
HEADERS = {"Content-Type": "application/json", "X-API-Key": VELD_API_KEY}
JUDGE_HEADERS = {"Content-Type": "application/json"}
if JUDGE_API_KEY:
    JUDGE_HEADERS["Authorization"] = f"Bearer {JUDGE_API_KEY}"

# ---------------------------------------------------------------------------
# Memories — 20 entries across 4 weeks
# ---------------------------------------------------------------------------
BASE_TIME = datetime.now(timezone.utc) - timedelta(weeks=4)


def ts(days_offset: int, hours: int = 10) -> str:
    return (BASE_TIME + timedelta(days=days_offset, hours=hours)).isoformat()


MEMORIES = [
    {
        "id_tag": "M01",
        "content": "Sprint 14 planning: We committed to 34 story points. Key items are the payment gateway migration from Stripe v2 to v3, the Redis cache invalidation bug, and the new onboarding wizard. Sarah is leading the payment work, Raj is on caching, and I'm handling onboarding.",
        "tags": ["sprint", "planning", "sprint-14"],
        "memory_type": "Observation",
        "created_at": ts(0),
    },
    {
        "id_tag": "M02",
        "content": "Architecture decision: We chose PostgreSQL over MongoDB for the analytics pipeline. Key reasons were complex joins needed for funnel analysis, ACID compliance for financial data, and the team's existing expertise. Marcus strongly advocated for Mongo but the vote was 4-1 in favor of Postgres.",
        "tags": ["architecture", "database", "decision", "analytics"],
        "memory_type": "Decision",
        "created_at": ts(1),
    },
    {
        "id_tag": "M03",
        "content": "Bug report: Users in the EU region are experiencing 5-second delays on the checkout page. The root cause is the payment provider's EU endpoint having high latency. Temporary fix: route EU traffic through the UK proxy. Permanent fix needs Stripe v3 migration which Sarah is working on.",
        "tags": ["bug", "performance", "checkout", "EU", "latency"],
        "memory_type": "Observation",
        "created_at": ts(2),
    },
    {
        "id_tag": "M04",
        "content": "Personal preference: I strongly prefer dark mode in all development tools. My IDE theme is Dracula, terminal is Catppuccin Mocha, and I use the Fira Code font with ligatures enabled at 14pt.",
        "tags": ["preference", "tooling", "IDE", "theme"],
        "memory_type": "Preference",
        "created_at": ts(3),
    },
    {
        "id_tag": "M05",
        "content": "Raj found the Redis cache invalidation bug. It was a race condition in the pub/sub listener — when two nodes receive the same invalidation event, they both try to refresh the cache simultaneously, causing a thundering herd. Fix: added distributed locking with Redlock.",
        "tags": ["bug-fix", "redis", "cache", "race-condition", "Raj"],
        "memory_type": "Observation",
        "created_at": ts(7),
    },
    {
        "id_tag": "M06",
        "content": "Team lunch at Sakura Sushi near the downtown office. Everyone was there except Marcus who was working remotely from Portland. Sarah mentioned she's thinking about transitioning to a principal engineer role. Good team morale overall.",
        "tags": ["social", "team", "lunch", "Sakura-Sushi"],
        "memory_type": "Observation",
        "created_at": ts(8, 12),
    },
    {
        "id_tag": "M07",
        "content": "The onboarding wizard prototype is working. It has 5 steps: account creation, team setup, first project, integration connections, and a guided tour. User testing showed 78% completion rate, up from 45% with the old flow. VP of Product Elena approved moving to production.",
        "tags": ["onboarding", "prototype", "user-testing", "Elena"],
        "memory_type": "Observation",
        "created_at": ts(9),
    },
    {
        "id_tag": "M08",
        "content": "Decision to use Kubernetes over ECS for the new microservices deployment. Reasoning: better multi-cloud portability, stronger community tooling (Helm, ArgoCD), and our SRE team already has K8s experience from the data pipeline project. Cost estimate is $2,400/month higher but worth the flexibility.",
        "tags": ["infrastructure", "kubernetes", "deployment", "decision"],
        "memory_type": "Decision",
        "created_at": ts(10),
    },
    {
        "id_tag": "M09",
        "content": "Production incident at 3 AM: the analytics database ran out of disk space. Root cause was the funnel_events table growing 10x faster than projected because of a missing TTL policy. We added a 90-day retention policy and partitioned by month. Total downtime was 47 minutes.",
        "tags": ["incident", "database", "disk-space", "analytics", "downtime"],
        "memory_type": "Observation",
        "created_at": ts(14, 3),
    },
    {
        "id_tag": "M10",
        "content": "Sarah completed the Stripe v3 migration for the US region. Performance improved by 40% — average checkout time dropped from 3.2s to 1.9s. EU migration is scheduled for next week. She used the new Stripe Payment Intents API which also enables Apple Pay and Google Pay.",
        "tags": ["stripe", "migration", "performance", "Sarah", "checkout"],
        "memory_type": "Observation",
        "created_at": ts(15),
    },
    {
        "id_tag": "M11",
        "content": "Conference talk accepted: I'll be presenting 'Building Resilient Caching at Scale' at RustConf 2026 in Austin, Texas on September 15th. The talk covers our Redis architecture evolution, the thundering herd fix, and distributed locking patterns.",
        "tags": ["conference", "RustConf", "talk", "Austin"],
        "memory_type": "Observation",
        "created_at": ts(16),
    },
    {
        "id_tag": "M12",
        "content": "Code review feedback from Marcus on the onboarding wizard: he flagged that the integration step makes 12 sequential API calls which could be parallelized. Good catch — refactored to use Promise.all() and reduced step 4 load time from 4.5s to 0.8s.",
        "tags": ["code-review", "Marcus", "onboarding", "performance"],
        "memory_type": "Observation",
        "created_at": ts(17),
    },
    {
        "id_tag": "M13",
        "content": "Q3 OKR planning meeting. Our team's key results: (1) reduce p95 API latency from 800ms to 200ms, (2) achieve 99.95% uptime SLA, (3) launch self-serve enterprise onboarding. VP Elena emphasized that the enterprise onboarding is the highest revenue priority.",
        "tags": ["OKR", "Q3", "planning", "latency", "uptime", "enterprise"],
        "memory_type": "Observation",
        "created_at": ts(21),
    },
    {
        "id_tag": "M14",
        "content": "Security audit findings: The penetration test by CrowdStrike found 3 medium-severity issues: (1) missing rate limiting on the password reset endpoint, (2) CORS misconfiguration allowing wildcard origins in staging, (3) session tokens not rotated after privilege escalation. All assigned to Sprint 15.",
        "tags": ["security", "audit", "CrowdStrike", "vulnerabilities"],
        "memory_type": "Observation",
        "created_at": ts(22),
    },
    {
        "id_tag": "M15",
        "content": "Raj is transferring to the ML platform team next month. His replacement on our team will be Priya, who has 5 years of experience with distributed systems at Netflix. She starts on April 15th. Raj will do a 2-week knowledge transfer on the caching layer.",
        "tags": ["team-change", "Raj", "Priya", "transfer", "Netflix"],
        "memory_type": "Observation",
        "created_at": ts(23),
    },
    {
        "id_tag": "M16",
        "content": "Decided to adopt GraphQL for the new enterprise API instead of REST. Reasons: clients need flexible field selection for dashboard customization, reduces over-fetching which is critical for mobile, and Apollo Federation allows us to stitch microservice schemas. Timeline: MVP by end of Q3.",
        "tags": ["architecture", "GraphQL", "enterprise", "API", "decision"],
        "memory_type": "Decision",
        "created_at": ts(24),
    },
    {
        "id_tag": "M17",
        "content": "The annual company hackathon is on April 20-21. Our team is building a real-time collaboration feature using CRDTs and WebSockets. I'm excited because this could become a real product feature. Last year's winning team built the AI-powered search that's now in production.",
        "tags": ["hackathon", "CRDT", "collaboration", "WebSocket"],
        "memory_type": "Observation",
        "created_at": ts(25),
    },
    {
        "id_tag": "M18",
        "content": "Performance benchmark results for the new analytics pipeline on PostgreSQL: 500K events/minute ingestion, p99 query latency at 120ms for 30-day windows, and 45ms for 7-day windows. This exceeds our Q3 target of 200ms p95. The columnar extension (TimescaleDB) was key to this performance.",
        "tags": ["benchmark", "analytics", "PostgreSQL", "TimescaleDB", "performance"],
        "memory_type": "Observation",
        "created_at": ts(26),
    },
    {
        "id_tag": "M19",
        "content": "Sarah's Stripe v3 EU migration went live today. Checkout latency for EU users dropped from 5 seconds to 1.4 seconds. The bug from two weeks ago about EU checkout delays is now fully resolved. Apple Pay and Google Pay are also enabled for EU customers.",
        "tags": ["stripe", "EU", "migration", "Sarah", "checkout", "resolved"],
        "memory_type": "Observation",
        "created_at": ts(27),
    },
    {
        "id_tag": "M20",
        "content": "End-of-sprint retro for Sprint 14. Completed 31 of 34 story points. Carried over: the enterprise SSO integration (blocked by IdP provider delays) and two minor UI polish tasks. Team velocity trending up — average of last 3 sprints is 30 points. Celebrating at Happy Hour Friday at The Rusty Anchor.",
        "tags": ["retro", "sprint-14", "velocity", "team"],
        "memory_type": "Observation",
        "created_at": ts(28),
    },
]

# ---------------------------------------------------------------------------
# Queries — 20 across 4 categories, each with expected answers for judging
# ---------------------------------------------------------------------------
QUERIES = [
    # ── SINGLE-HOP ──
    {
        "category": "single-hop",
        "query": "What database did we choose for the analytics pipeline?",
        "expected": ["M02"],
        "gold_answer": "PostgreSQL. The decision was made over MongoDB, with reasons including complex joins for funnel analysis, ACID compliance for financial data, and team expertise. The vote was 4-1.",
    },
    {
        "category": "single-hop",
        "query": "What was the root cause of the Redis cache invalidation bug?",
        "expected": ["M05"],
        "gold_answer": "A race condition in the pub/sub listener. Two nodes receiving the same invalidation event both try to refresh the cache simultaneously, causing a thundering herd. Fixed with distributed locking using Redlock.",
    },
    {
        "category": "single-hop",
        "query": "What is my preferred IDE theme and font?",
        "expected": ["M04"],
        "gold_answer": "Dracula theme for IDE, Catppuccin Mocha for terminal, Fira Code font with ligatures at 14pt. Prefers dark mode in all dev tools.",
    },
    {
        "category": "single-hop",
        "query": "How many story points did we commit to in Sprint 14?",
        "expected": ["M01"],
        "gold_answer": "34 story points. Key items were payment gateway migration (Stripe v2 to v3), Redis cache invalidation bug, and new onboarding wizard.",
    },
    {
        "category": "single-hop",
        "query": "What were the security vulnerabilities found in the penetration test?",
        "expected": ["M14"],
        "gold_answer": "CrowdStrike found 3 medium-severity issues: (1) missing rate limiting on password reset endpoint, (2) CORS misconfiguration allowing wildcard origins in staging, (3) session tokens not rotated after privilege escalation.",
    },
    # ── TEMPORAL ──
    {
        "category": "temporal",
        "query": "What happened during the production incident that caused downtime?",
        "expected": ["M09"],
        "gold_answer": "At 3 AM, the analytics database ran out of disk space. The funnel_events table grew 10x faster than projected due to a missing TTL policy. Added 90-day retention and monthly partitioning. 47 minutes total downtime.",
    },
    {
        "category": "temporal",
        "query": "When is my conference talk and what is it about?",
        "expected": ["M11"],
        "gold_answer": "September 15th at RustConf 2026 in Austin, Texas. Topic: 'Building Resilient Caching at Scale' — covering Redis architecture evolution, thundering herd fix, and distributed locking patterns.",
    },
    {
        "category": "temporal",
        "query": "When does Priya start and who is she replacing?",
        "expected": ["M15"],
        "gold_answer": "Priya starts April 15th, replacing Raj who is transferring to the ML platform team. She has 5 years of distributed systems experience from Netflix. Raj will do a 2-week knowledge transfer on the caching layer.",
    },
    {
        "category": "temporal",
        "query": "What are the dates for the company hackathon?",
        "expected": ["M17"],
        "gold_answer": "April 20-21. The team is building a real-time collaboration feature using CRDTs and WebSockets.",
    },
    {
        "category": "temporal",
        "query": "When was the EU checkout latency issue finally resolved?",
        "expected": ["M19"],
        "gold_answer": "Resolved when Sarah's Stripe v3 EU migration went live (about 4 weeks after the bug was first reported). EU checkout latency dropped from 5 seconds to 1.4 seconds.",
    },
    # ── MULTI-HOP ──
    {
        "category": "multi-hop",
        "query": "How did Sarah's Stripe migration fix the EU checkout delay problem?",
        "expected": ["M03", "M10", "M19"],
        "gold_answer": "The EU checkout had 5s delays due to the payment provider's EU endpoint latency (M03). Sarah first migrated US to Stripe v3, improving checkout from 3.2s to 1.9s (M10). Then the EU migration brought EU latency from 5s to 1.4s, fully resolving the issue (M19).",
    },
    {
        "category": "multi-hop",
        "query": "What caching problems did we have and how were they resolved?",
        "expected": ["M01", "M05"],
        "gold_answer": "The Redis cache invalidation bug was identified in Sprint 14 planning (M01). Raj found it was a race condition in the pub/sub listener causing a thundering herd, and fixed it with Redlock distributed locking (M05).",
    },
    {
        "category": "multi-hop",
        "query": "Which team member reviewed the onboarding wizard and what did they find?",
        "expected": ["M07", "M12"],
        "gold_answer": "Marcus reviewed the onboarding wizard (M12). He found that the integration step made 12 sequential API calls that could be parallelized. After refactoring with Promise.all(), step 4 load time dropped from 4.5s to 0.8s. The original prototype (M07) had 78% completion rate.",
    },
    {
        "category": "multi-hop",
        "query": "What is Raj working on now and what happens when he leaves?",
        "expected": ["M05", "M15"],
        "gold_answer": "Raj is working on the caching layer — he fixed the Redis cache invalidation bug using Redlock (M05). He's transferring to the ML platform team next month. Priya from Netflix replaces him starting April 15th, with a 2-week knowledge transfer (M15).",
    },
    {
        "category": "multi-hop",
        "query": "How do our analytics benchmark numbers compare to the Q3 OKR targets?",
        "expected": ["M13", "M18"],
        "gold_answer": "Q3 OKR targets include reducing p95 API latency to 200ms (M13). The analytics pipeline benchmark shows p99 at 120ms for 30-day windows and 45ms for 7-day windows (M18) — already exceeding the target.",
    },
    # ── OPEN-DOMAIN ──
    {
        "category": "open-domain",
        "query": "What is the team's overall morale and social dynamics like?",
        "expected": ["M06", "M20"],
        "gold_answer": "Good morale overall. Team had lunch at Sakura Sushi with positive vibe (M06). Sprint 14 retro showed strong velocity (31/34 points) and the team celebrated at Happy Hour at The Rusty Anchor (M20). Sarah considering principal engineer role.",
    },
    {
        "category": "open-domain",
        "query": "What are the most important strategic priorities for the team?",
        "expected": ["M13", "M16"],
        "gold_answer": "Enterprise onboarding is highest revenue priority per VP Elena (M13). Q3 OKRs: reduce p95 latency to 200ms, achieve 99.95% uptime, launch self-serve enterprise onboarding. GraphQL adopted for enterprise API with MVP by end of Q3 (M16).",
    },
    {
        "category": "open-domain",
        "query": "What infrastructure and deployment decisions have we made recently?",
        "expected": ["M02", "M08", "M16"],
        "gold_answer": "Three key decisions: (1) PostgreSQL over MongoDB for analytics (M02), (2) Kubernetes over ECS for microservices at $2,400/month more (M08), (3) GraphQL over REST for enterprise API using Apollo Federation (M16).",
    },
    {
        "category": "open-domain",
        "query": "What performance improvements have we achieved across the product?",
        "expected": ["M05", "M07", "M10", "M12", "M18", "M19"],
        "gold_answer": "Multiple wins: Redis thundering herd fix (M05), onboarding wizard 45%->78% completion (M07), US checkout 3.2s->1.9s (M10), onboarding step 4 from 4.5s->0.8s (M12), analytics p99 120ms beating 200ms target (M18), EU checkout 5s->1.4s (M19).",
    },
    {
        "category": "open-domain",
        "query": "Who are the key people on the team and what are they responsible for?",
        "expected": ["M01", "M05", "M06", "M10", "M15"],
        "gold_answer": "Sarah: payment/Stripe migration. Raj: caching layer (transferring to ML team). Marcus: code reviews, remote from Portland. Priya: incoming replacement from Netflix. Elena: VP of Product. The user handles onboarding.",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def store_memory(mem: dict) -> str | None:
    payload = {
        "user_id": USER_ID,
        "content": mem["content"],
        "tags": mem.get("tags", []),
        "memory_type": mem.get("memory_type", "Observation"),
    }
    if mem.get("created_at"):
        payload["created_at"] = mem["created_at"]
    resp = requests.post(f"{VELD_URL}/api/remember", headers=HEADERS, json=payload)
    if resp.status_code != 200:
        print(f"  ERROR storing {mem['id_tag']}: {resp.status_code} {resp.text}")
        return None
    return resp.json().get("id")


def recall_memories(query: str, limit: int = 10) -> list[dict]:
    payload = {"user_id": USER_ID, "query": query, "limit": limit, "mode": "semantic"}
    resp = requests.post(f"{VELD_URL}/api/recall", headers=HEADERS, json=payload)
    if resp.status_code != 200:
        print(f"  ERROR recalling: {resp.status_code} {resp.text}")
        return []
    return resp.json().get("memories", [])



def _call_judge(prompt: str) -> dict:
    """Send a prompt to the judge LLM and parse SCORE:/REASON: response."""
    try:
        resp = requests.post(
            f"{JUDGE_URL}/chat/completions",
            headers=JUDGE_HEADERS,
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            timeout=180,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip QwQ thinking tags
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if not clean:
            clean = text

        score = None
        reason = ""

        for line in clean.split("\n"):
            line = line.strip()
            if re.match(r'^[*\s]*SCORE[:\s]', line, re.IGNORECASE):
                m = re.search(r'(\d)', line)
                if m:
                    score = min(int(m.group(1)), 3)
            elif re.match(r'^[*\s]*REASON[:\s]', line, re.IGNORECASE):
                reason = re.sub(r'^[*\s]*REASON[:\s]*', '', line, flags=re.IGNORECASE).strip()

        if score is None:
            all_scores = re.findall(r'SCORE[:\s]*(\d)', clean, re.IGNORECASE)
            if all_scores:
                score = min(int(all_scores[-1]), 3)

        if score is None:
            all_scores = re.findall(r'SCORE[:\s]*(\d)', text, re.IGNORECASE)
            if all_scores:
                score = min(int(all_scores[-1]), 3)
                reason = reason or "(extracted from reasoning trace)"

        if score is None:
            score = -1
            reason = f"Parse failure: {clean[:200] if clean else text[:200]}"

        if not reason and score >= 0:
            reason_match = re.search(r'REASON[:\s]*(.+)', clean, re.IGNORECASE)
            if reason_match:
                reason = reason_match.group(1).strip()

        return {"score": score, "reason": reason, "raw": text}

    except Exception as e:
        return {"score": -1, "reason": f"LLM error: {e}", "raw": ""}


def _build_context_block(retrieved_memories: list[dict], limit: int = 5) -> str:
    return "\n\n".join(
        f"[Memory {i+1}] (score: {m.get('score', 0):.3f})\n{m.get('experience', {}).get('content', '') or m.get('content', '')}"
        for i, m in enumerate(retrieved_memories[:limit])
    )


# ---------------------------------------------------------------------------
# E5: 2-Pass Judge (evidence sufficiency + answer correctness)
# Overlook fix: split the existing single rubric into two independent passes
# so we can distinguish "retrieval found enough" from "answer was correct".
# ---------------------------------------------------------------------------

def judge_evidence_sufficiency(query: str, retrieved_memories: list[dict]) -> dict:
    """Pass 1: Can the query be answered from ONLY the retrieved memories?"""
    context_block = _build_context_block(retrieved_memories)

    prompt = f"""You are evaluating whether retrieved memories contain sufficient evidence to answer a question.
Do NOT answer the question. Only judge whether the evidence is there.

QUESTION: {query}

RETRIEVED MEMORIES:
{context_block}

Score the evidence sufficiency:
  3 = all information needed is present in the memories
  2 = some key information present but incomplete
  1 = tangentially related but cannot answer
  0 = no relevant information

Reply ONLY: SCORE: N REASON: one sentence"""

    return _call_judge(prompt)


def judge_answer_correctness(query: str, gold_answer: str, retrieved_memories: list[dict]) -> dict:
    """Pass 2: Given the memories, is the answer correct? Compares to gold."""
    context_block = _build_context_block(retrieved_memories)

    prompt = f"""Score how well a system could answer this question using ONLY the retrieved memories.
Compare what the memories support against the gold answer.

Q: {query}
GOLD ANSWER: {gold_answer}

RETRIEVED MEMORIES:
{context_block}

Rubric: 3=fully correct answer derivable, 2=partially correct, 1=tangential, 0=wrong/irrelevant.
Reply ONLY: SCORE: N REASON: one sentence"""

    return _call_judge(prompt)


# ---------------------------------------------------------------------------
# E6: Misleading-Context Detection (LLM-judge only, no cross-reference)
# Overlook fix: dropped Hebbian feedback cross-reference (endpoint doesn't exist)
# ---------------------------------------------------------------------------

def judge_misleading_context(query: str, gold_answer: str, retrieved_memories: list[dict]) -> dict:
    """Detect memories in top-K that could mislead from the correct answer."""
    context_block = _build_context_block(retrieved_memories)

    prompt = f"""You are checking retrieved memories for misleading content.
A memory is MISLEADING if it contains information that could cause an incorrect answer
to the question — e.g. outdated facts, contradictory claims, or plausible-but-wrong details.

QUESTION: {query}
CORRECT ANSWER: {gold_answer}

RETRIEVED MEMORIES:
{context_block}

For each memory, state if it is SAFE or MISLEADING.
Then give a count.

Reply format:
MEMORY_1: SAFE or MISLEADING (reason)
MEMORY_2: SAFE or MISLEADING (reason)
...
SCORE: N (number of misleading memories, 0-5)
REASON: one sentence summary"""

    result = _call_judge(prompt)

    # Parse which positions are misleading
    misleading_positions = []
    text = result.get("raw", "")
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    for i in range(1, 6):
        pattern = rf'MEMORY_{i}\s*:\s*MISLEADING'
        if re.search(pattern, clean, re.IGNORECASE):
            misleading_positions.append(i)

    result["misleading_positions"] = misleading_positions
    result["misleading_count"] = len(misleading_positions)
    return result


# ---------------------------------------------------------------------------
# E1: Contradiction Detection (content-based, not metadata-based)
# Overlook fix: benchmark memories bypass hook pipeline so temporal facts
# don't exist. Instead, detect contradictions from the CONTENT of retrieved
# memories — two memories stating conflicting facts about the same entity.
# ---------------------------------------------------------------------------

def detect_contradictions_in_topk(query: str, retrieved_memories: list[dict]) -> dict:
    """Detect contradicting facts between memories in the top-K results."""
    if len(retrieved_memories) < 2:
        return {"contradiction_count": 0, "contradictions": [], "raw": ""}

    context_block = _build_context_block(retrieved_memories)

    prompt = f"""Analyze these retrieved memories for CONTRADICTIONS.
A contradiction is when two memories state conflicting facts about the same entity, event, or topic.
Examples: different dates for the same event, different people assigned to the same task,
conflicting decisions, or superseded information where the old version is also present.

QUERY: {query}

MEMORIES:
{context_block}

List each contradiction found (if any). For each, cite the two memory numbers.
If no contradictions, say NONE.

Reply format:
CONTRADICTION_1: Memory X vs Memory Y — description
CONTRADICTION_2: Memory X vs Memory Y — description
SCORE: N (number of contradictions found, 0 if none)
REASON: one sentence"""

    result = _call_judge(prompt)

    # Parse contradiction pairs
    clean = re.sub(r'<think>.*?</think>', '', result.get("raw", ""), flags=re.DOTALL).strip()
    contradictions = re.findall(
        r'CONTRADICTION_\d+\s*:\s*Memory\s*(\d+)\s*vs\s*Memory\s*(\d+)\s*[—-]\s*(.+)',
        clean, re.IGNORECASE
    )
    result["contradictions"] = [
        {"memory_a": int(a), "memory_b": int(b), "description": desc.strip()}
        for a, b, desc in contradictions
    ]
    result["contradiction_count"] = len(contradictions)
    return result


# ---------------------------------------------------------------------------
# Legacy wrapper for backward compat
# ---------------------------------------------------------------------------

def judge_with_llm(query: str, gold_answer: str, retrieved_memories: list[dict]) -> dict:
    """Combined judge — runs both passes and merges results."""
    sufficiency = judge_evidence_sufficiency(query, retrieved_memories)
    correctness = judge_answer_correctness(query, gold_answer, retrieved_memories)
    return {
        "score": correctness["score"],
        "reason": correctness["reason"],
        "raw": correctness["raw"],
        "evidence_sufficiency": sufficiency["score"],
        "evidence_reason": sufficiency["reason"],
    }


# ---------------------------------------------------------------------------
# Single-call combined judge — folds E5 evidence + E5 correctness + E1
# contradictions + E6 misleading into one LLM round-trip per query. Drops
# total judge calls from 4N to N (80→20 for the standard 20-query suite).
# ---------------------------------------------------------------------------

def judge_query_all_in_one(query: str, gold_answer: str, retrieved_memories: list[dict]) -> dict:
    """One LLM call that scores all four judge axes for a single query."""
    context_block = _build_context_block(retrieved_memories)

    prompt = f"""You are evaluating retrieved memories against a question. Score FOUR dimensions in a single response. Be precise and concise.

QUESTION: {query}
GOLD ANSWER: {gold_answer}

RETRIEVED MEMORIES:
{context_block}

Dimensions:
1. EVIDENCE — does the evidence support answering at all? (0=none, 1=tangential, 2=partial, 3=full)
2. CORRECTNESS — could the gold answer be derived from these memories alone? (0=miss, 1=tangential, 2=partial, 3=full)
3. CONTRADICTIONS — number of memory PAIRS that state conflicting facts about the same entity, event, or topic
4. MISLEADING — number of memories that could mislead from the correct answer (outdated, contradictory, plausible-but-wrong)

Reply EXACTLY this format, one field per line, nothing else:

EVIDENCE_SCORE: <0-3>
EVIDENCE_REASON: <one sentence>
CORRECTNESS_SCORE: <0-3>
CORRECTNESS_REASON: <one sentence>
CONTRADICTION_COUNT: <integer>
CONTRADICTION_REASON: <one sentence or "none">
MISLEADING_COUNT: <integer 0-5>
MISLEADING_POSITIONS: <comma-separated memory indices 1-5, or "none">
MISLEADING_REASON: <one sentence>"""

    try:
        resp = requests.post(
            f"{JUDGE_URL}/chat/completions",
            headers=JUDGE_HEADERS,
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=240,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip() or text
    except Exception as e:
        return _empty_combined(f"LLM error: {e}")

    def field(name, default=""):
        m = re.search(rf'^[*\s]*{name}\s*:\s*(.+)$', clean, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else default

    def int_field(name, lo=0, hi=99):
        v = field(name, "")
        m = re.search(r'-?\d+', v)
        if not m:
            return -1
        try:
            return max(lo, min(hi, int(m.group(0))))
        except ValueError:
            return -1

    misleading_positions_text = field("MISLEADING_POSITIONS", "")
    if "none" in misleading_positions_text.lower():
        misleading_positions = []
    else:
        misleading_positions = [
            int(m.group(0)) for m in re.finditer(r'\d+', misleading_positions_text)
            if 1 <= int(m.group(0)) <= 5
        ]

    return {
        "evidence_sufficiency": int_field("EVIDENCE_SCORE", 0, 3),
        "evidence_reason": field("EVIDENCE_REASON"),
        "judge_score": int_field("CORRECTNESS_SCORE", 0, 3),
        "judge_reason": field("CORRECTNESS_REASON"),
        "contradiction_count": max(int_field("CONTRADICTION_COUNT", 0, 25), 0),
        "contradiction_reason": field("CONTRADICTION_REASON"),
        "misleading_count": max(int_field("MISLEADING_COUNT", 0, 5), 0),
        "misleading_positions": misleading_positions,
        "misleading_reason": field("MISLEADING_REASON"),
        "raw": text,
    }


def _empty_combined(reason: str) -> dict:
    return {
        "evidence_sufficiency": -1,
        "evidence_reason": reason,
        "judge_score": -1,
        "judge_reason": reason,
        "contradiction_count": 0,
        "contradiction_reason": reason,
        "misleading_count": 0,
        "misleading_positions": [],
        "misleading_reason": reason,
        "raw": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("LOCOMO LLM-JUDGED EVALUATION — veld")
    print(f"Server:  {VELD_URL}")
    print(f"Judge:   {JUDGE_MODEL} @ {JUDGE_URL}")
    print(f"User:    {USER_ID}")
    print(f"Time:    {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)

    # Health check
    print("\n[1/3] Health checks...")
    try:
        h = requests.get(f"{VELD_URL}/health", timeout=5).json()
        print(f"  veld: {h.get('status')} v{h.get('version')}")
    except Exception as e:
        print(f"  FATAL: veld unreachable: {e}")
        sys.exit(1)
    try:
        m = requests.get(f"{JUDGE_URL}/models", headers=JUDGE_HEADERS, timeout=15).json()
        # Tolerate Together's larger model list — only warn if model is missing.
        data_field = m.get("data", m) if isinstance(m, dict) else m
        model_ids = [x.get("id", "") for x in data_field] if isinstance(data_field, list) else []
        if model_ids and JUDGE_MODEL not in model_ids:
            print(f"  WARNING: {JUDGE_MODEL} not in models listing ({len(model_ids)} models seen)")
        else:
            print(f"  judge: {JUDGE_MODEL} ready")
    except Exception as e:
        print(f"  FATAL: judge endpoint unreachable at {JUDGE_URL}: {e}")
        sys.exit(1)

    # Get git commit
    try:
        build = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True
        ).strip()
    except Exception:
        build = "unknown"

    # Store memories
    print(f"\n[2/3] Storing {len(MEMORIES)} memories...")
    id_map = {}
    for mem in MEMORIES:
        uuid = store_memory(mem)
        if uuid:
            id_map[mem["id_tag"]] = uuid
            print(f"  {mem['id_tag']} -> {uuid[:12]}...")
        else:
            print(f"  {mem['id_tag']} -> FAILED")

    print(f"  Stored: {len(id_map)}/{len(MEMORIES)}")
    time.sleep(1.5)  # indexing

    # [3/3] Interleaved retrieve + judge.
    # Recall and judge each query back-to-back so Together's dedicated endpoint
    # stays warm (its inactive_timeout kicks in if too much time passes without
    # a call).
    print(f"\n[3/3] Interleaved recall + combined judge ({len(QUERIES)} queries) with {JUDGE_MODEL}...")
    uuid_to_tag = {v: k for k, v in id_map.items()}
    results = []

    for i, q in enumerate(QUERIES):
        # ---- recall ----
        t0 = time.time()
        memories = recall_memories(q["query"], limit=10)
        recall_ms = (time.time() - t0) * 1000

        ranked_uuids = [m["id"] for m in memories]
        ranked_tags = [uuid_to_tag.get(uid, "?") for uid in ranked_uuids]
        expected_uuids = {id_map[t] for t in q["expected"] if t in id_map}

        mrr = compute_mrr(ranked_uuids, expected_uuids)
        r5 = compute_recall_at_k(ranked_uuids, expected_uuids, 5)
        r10 = compute_recall_at_k(ranked_uuids, expected_uuids, 10)

        # E4: Freshness — compute per-memory age bands for top-K
        freshness_bands = []
        for m in memories[:5]:
            age = compute_memory_age_days(m.get("created_at", ""))
            freshness_bands.append(classify_freshness_band(age))

        result = {
            "index": i + 1,
            "category": q["category"],
            "query": q["query"],
            "expected_tags": q["expected"],
            "returned_tags": ranked_tags,
            "returned_scores": [m.get("score", 0) for m in memories],
            "mrr": mrr,
            "recall_at_5": r5,
            "recall_at_10": r10,
            "latency_ms": round(recall_ms),
            "memories_for_judge": memories[:5],
            "gold_answer": q["gold_answer"],
            "freshness_bands": freshness_bands,
        }

        status = "HIT " if mrr > 0 else "MISS"
        found = [t for t in ranked_tags[:5] if t in q["expected"]]

        # ---- judge ----
        t1 = time.time()
        j = judge_query_all_in_one(q["query"], q["gold_answer"], memories[:5])
        judge_ms = (time.time() - t1) * 1000

        result["evidence_sufficiency"] = j["evidence_sufficiency"]
        result["evidence_reason"] = j["evidence_reason"]
        result["judge_score"] = j["judge_score"]
        result["judge_reason"] = j["judge_reason"]
        result["contradiction_count"] = j["contradiction_count"]
        result["contradictions"] = []
        result["misleading_count"] = j["misleading_count"]
        result["misleading_positions"] = j["misleading_positions"]
        result["misleading_rate"] = j["misleading_count"] / max(len(memories[:5]), 1)

        ev_tag = {3: "FULL", 2: "PART", 1: "TANG", 0: "NONE", -1: "ERR "}.get(j["evidence_sufficiency"], "????")
        co_tag = {3: "FULL", 2: "PART", 1: "TANG", 0: "MISS", -1: "ERR "}.get(j["judge_score"], "????")
        print(
            f"  Q{i+1:02d} [{q['category']:11s}] {status} MRR={mrr:.3f} R@5={r5:.2f} top3={ranked_tags[:3]}"
        )
        print(
            f"      ev={j['evidence_sufficiency']}/3 {ev_tag} cor={j['judge_score']}/3 {co_tag} "
            f"contra={j['contradiction_count']} mis={j['misleading_count']} "
            f"(recall {recall_ms:.0f}ms + judge {judge_ms:.0f}ms)"
        )
        gap = result["evidence_sufficiency"] - result["judge_score"]
        if gap > 0 and result["evidence_sufficiency"] >= 0 and result["judge_score"] >= 0:
            print(f"      ^ GENERATION GAP: evidence={result['evidence_sufficiency']} but correct={result['judge_score']}")

        results.append(result)

    # Report
    print(f"\nResults")
    print("=" * 72)

    categories = ["single-hop", "temporal", "multi-hop", "open-domain"]

    print(f"\n{'Category':<13s} {'MRR':>6s} {'R@5':>6s} {'R@10':>6s} {'Evid':>5s} {'Corr':>5s} {'Mis':>4s} {'Con':>4s}  n")
    print("-" * 72)

    all_results = []
    for cat in categories:
        cr = [r for r in results if r["category"] == cat]
        n = len(cr)
        cat_mrr = sum(r["mrr"] for r in cr) / n
        cat_r5 = sum(r["recall_at_5"] for r in cr) / n
        cat_r10 = sum(r["recall_at_10"] for r in cr) / n
        valid_evid = [r["evidence_sufficiency"] for r in cr if r["evidence_sufficiency"] >= 0]
        cat_evid = sum(valid_evid) / len(valid_evid) if valid_evid else 0
        valid_corr = [r["judge_score"] for r in cr if r["judge_score"] >= 0]
        cat_corr = sum(valid_corr) / len(valid_corr) if valid_corr else 0
        cat_mislead = sum(r.get("misleading_count", 0) for r in cr) / n
        cat_contra = sum(r.get("contradiction_count", 0) for r in cr) / n

        print(
            f"  {cat:<11s} {cat_mrr:>6.3f} {cat_r5:>6.3f} {cat_r10:>6.3f} "
            f"{cat_evid:>4.1f}/3 {cat_corr:>4.1f}/3 {cat_mislead:>4.2f} {cat_contra:>4.2f}  {n}"
        )
        all_results.extend(cr)

    n = len(all_results)
    overall_mrr = sum(r["mrr"] for r in all_results) / n
    overall_r5 = sum(r["recall_at_5"] for r in all_results) / n
    overall_r10 = sum(r["recall_at_10"] for r in all_results) / n
    valid_evid = [r["evidence_sufficiency"] for r in all_results if r["evidence_sufficiency"] >= 0]
    overall_evid = sum(valid_evid) / len(valid_evid) if valid_evid else 0
    valid_corr = [r["judge_score"] for r in all_results if r["judge_score"] >= 0]
    overall_corr = sum(valid_corr) / len(valid_corr) if valid_corr else 0
    overall_mislead = sum(r.get("misleading_count", 0) for r in all_results) / n
    overall_contra = sum(r.get("contradiction_count", 0) for r in all_results) / n
    avg_latency = sum(r["latency_ms"] for r in all_results) / n

    print("-" * 72)
    print(
        f"  {'OVERALL':<11s} {overall_mrr:>6.3f} {overall_r5:>6.3f} {overall_r10:>6.3f} "
        f"{overall_evid:>4.1f}/3 {overall_corr:>4.1f}/3 {overall_mislead:>4.2f} {overall_contra:>4.2f}  {n}"
    )

    # E5 diagnostic: generation gap
    generation_gap_queries = [
        r for r in all_results
        if r.get("evidence_sufficiency", -1) >= 2 and r.get("judge_score", -1) <= 1
    ]
    if generation_gap_queries:
        print(f"\n  GENERATION GAP: {len(generation_gap_queries)} queries have sufficient evidence but poor answers:")
        for r in generation_gap_queries:
            print(f"    Q{r['index']:02d} [{r['category']}] evidence={r['evidence_sufficiency']} correct={r['judge_score']}")

    print(f"\n  Avg latency: {avg_latency:.0f}ms")
    print(f"  Correctness: {sum(1 for v in valid_corr if v==3)} full, "
          f"{sum(1 for v in valid_corr if v==2)} partial, "
          f"{sum(1 for v in valid_corr if v==1)} tangential, "
          f"{sum(1 for v in valid_corr if v==0)} miss"
          f"{', ' + str(sum(1 for r in all_results if r['judge_score']<0)) + ' errors' if any(r['judge_score']<0 for r in all_results) else ''}")

    # E4: Freshness distribution summary
    all_bands = [b for r in all_results for b in r.get("freshness_bands", [])]
    if all_bands:
        band_counts = {band: all_bands.count(band) for band in sorted(set(all_bands))}
        print(f"  Freshness distribution: {band_counts}")
        if len(band_counts) <= 1:
            print(f"  WARNING: All memories in single band — freshness stratification uninformative for this dataset")

    # E1+E6: Contamination summary
    total_misleading = sum(r.get("misleading_count", 0) for r in all_results)
    total_contradictions = sum(r.get("contradiction_count", 0) for r in all_results)
    print(f"  Contamination: {total_misleading} misleading memories, {total_contradictions} contradictions across {n} queries")

    # Save JSON
    content_hash = hashlib.md5(
        json.dumps([m["content"] for m in MEMORIES], sort_keys=True).encode()
    ).hexdigest()[:15]

    output = {
        "benchmark": "LOCOMO_LLM_JUDGE_v2",
        "version": h.get("version", "?"),
        "build": build,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "judge_model": JUDGE_MODEL,
        "corpus_size": len(MEMORIES),
        "query_count": len(QUERIES),
        "content_hash": content_hash,
        "metrics_version": "intugest-2026-03-30",
        "overall": {
            "mrr": round(overall_mrr, 4),
            "r5": round(overall_r5, 4),
            "r10": round(overall_r10, 4),
            "evidence_sufficiency_avg": round(overall_evid, 3),
            "answer_correctness_avg": round(overall_corr, 3),
            "generation_gap_count": len(generation_gap_queries),
            "avg_misleading_per_query": round(overall_mislead, 3),
            "avg_contradictions_per_query": round(overall_contra, 3),
            "avg_latency_ms": round(avg_latency),
        },
        "contamination": {
            "total_misleading_memories": total_misleading,
            "total_contradictions": total_contradictions,
            "queries_with_misleading": sum(1 for r in all_results if r.get("misleading_count", 0) > 0),
            "queries_with_contradictions": sum(1 for r in all_results if r.get("contradiction_count", 0) > 0),
        },
        "freshness": {
            "band_distribution": {band: all_bands.count(band) for band in sorted(set(all_bands))} if all_bands else {},
            "note": "LOCOMO dataset has 28-day window — all memories fall in Band B. Use LongMemEval for meaningful freshness stratification.",
        },
        "by_type": [],
        "per_query": [],
    }

    for cat in categories:
        cr = [r for r in results if r["category"] == cat]
        n = len(cr)
        vs_evid = [r["evidence_sufficiency"] for r in cr if r["evidence_sufficiency"] >= 0]
        vs_corr = [r["judge_score"] for r in cr if r["judge_score"] >= 0]
        evid_avg = sum(vs_evid) / len(vs_evid) if vs_evid else 0
        corr_avg = sum(vs_corr) / len(vs_corr) if vs_corr else 0
        output["by_type"].append({
            "type": cat,
            "n": n,
            "mrr": round(sum(r["mrr"] for r in cr) / n, 4),
            "r5": round(sum(r["recall_at_5"] for r in cr) / n, 4),
            "r10": round(sum(r["recall_at_10"] for r in cr) / n, 4),
            "evidence_sufficiency_avg": round(evid_avg, 3),
            "answer_correctness_avg": round(corr_avg, 3),
            "avg_misleading": round(sum(r.get("misleading_count", 0) for r in cr) / n, 3),
            "avg_contradictions": round(sum(r.get("contradiction_count", 0) for r in cr) / n, 3),
        })

    for r in results:
        output["per_query"].append({
            "query": r["query"],
            "type": r["category"],
            "expected": r["expected_tags"],
            "returned": r["returned_tags"][:5],
            "mrr": r["mrr"],
            "r5": r["recall_at_5"],
            "r10": r["recall_at_10"],
            "evidence_sufficiency": r.get("evidence_sufficiency", -1),
            "evidence_reason": r.get("evidence_reason", ""),
            "answer_correctness": r["judge_score"],
            "answer_reason": r["judge_reason"],
            "misleading_count": r.get("misleading_count", 0),
            "misleading_positions": r.get("misleading_positions", []),
            "contradiction_count": r.get("contradiction_count", 0),
            "contradictions": r.get("contradictions", []),
            "freshness_bands": r.get("freshness_bands", []),
            "ms": r["latency_ms"],
        })

    out_path = f"evaluations/locomo_judge_{build}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 72)
    print(f"MRR: {overall_mrr:.3f} | R@5: {overall_r5:.3f} | R@10: {overall_r10:.3f}")
    print(f"Evidence: {overall_evid:.2f}/3 | Correctness: {overall_corr:.2f}/3 | Gap: {len(generation_gap_queries)}")
    print(f"Contamination: {overall_mislead:.2f} misleading/q, {overall_contra:.2f} contradictions/q")
    print("=" * 72)


if __name__ == "__main__":
    main()
