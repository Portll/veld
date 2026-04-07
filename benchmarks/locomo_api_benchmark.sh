#!/bin/bash
# LOCOMO-style benchmark against veld API
# Stores 20 memories, runs 20 queries, computes MRR/R@5/R@10
set -euo pipefail

BASE_URL="http://127.0.0.1:3030"
API_KEY="dev-key-local"
USER_ID="locomo_bench_$(date +%s)"
RESULTS_FILE="/tmp/locomo_bench_results_$$.json"

echo "========================================================================"
echo "LOCOMO-STYLE BENCHMARK — veld API"
echo "Server: $BASE_URL"
echo "User:   $USER_ID"
echo "Time:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================================"

# Helper: store a memory and capture ID
store() {
    local id_tag="$1" content="$2" tags="$3" mtype="${4:-Observation}" src="${5:-user}" created="${6:-}" valence="${7:-}" arousal="${8:-}"
    local payload
    payload=$(cat <<EOJSON
{
  "user_id": "$USER_ID",
  "content": "$content",
  "tags": [$tags],
  "memory_type": "$mtype",
  "source_type": "$src"
  $([ -n "$created" ] && echo ", \"created_at\": \"$created\"" || true)
  $([ -n "$valence" ] && echo ", \"emotional_valence\": $valence" || true)
  $([ -n "$arousal" ] && echo ", \"emotional_arousal\": $arousal" || true)
}
EOJSON
)
    local resp
    resp=$(curl -s -X POST "$BASE_URL/api/remember" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d "$payload")
    local uuid
    uuid=$(echo "$resp" | /usr/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('id','FAIL'))" 2>/dev/null || echo "FAIL")
    echo "$uuid"
}

# Helper: recall and get ranked IDs as newline-separated list
recall_ids() {
    local query="$1" limit="${2:-10}"
    local payload
    payload=$(cat <<EOJSON
{
  "user_id": "$USER_ID",
  "query": "$query",
  "limit": $limit,
  "mode": "hybrid"
}
EOJSON
)
    curl -s -X POST "$BASE_URL/api/recall" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d "$payload" | /usr/bin/python3 -c "
import sys,json
data=json.load(sys.stdin)
for m in data.get('memories',[]):
    print(m['id'])
" 2>/dev/null
}

echo ""
echo "[1/4] Health check..."
HEALTH=$(curl -s "$BASE_URL/health")
echo "  $(echo "$HEALTH" | /usr/bin/python3 -c "import sys,json;h=json.load(sys.stdin);print(f\"Status: {h['status']} | Version: {h['version']}\")")"

echo ""
echo "[2/4] Storing 20 memories..."

# Timestamps spread over 4 weeks
BASE_TS=$(date -v-4w -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "4 weeks ago" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "2026-03-01T10:00:00Z")

# We'll compute offsets from a base epoch
# macOS date arithmetic
mk_ts() {
    local days=$1 hours=${2:-10}
    if date -v+1d +%s >/dev/null 2>&1; then
        # macOS
        date -v-4w -v+"${days}d" -v+"${hours}H" -u +%Y-%m-%dT%H:%M:%SZ
    else
        # GNU date
        date -u -d "4 weeks ago + $days days + $hours hours" +%Y-%m-%dT%H:%M:%SZ
    fi
}

declare -A ID_MAP

# Store all 20 memories
ID_MAP[M01]=$(store "M01" \
    "Sprint 14 planning: We committed to 34 story points. Key items are the payment gateway migration from Stripe v2 to v3, the Redis cache invalidation bug, and the new onboarding wizard. Sarah is leading the payment work, Raj is on caching, and I am handling onboarding." \
    '"sprint","planning","sprint-14"' \
    "Observation" "user" "$(mk_ts 0)" "0.3" "0.4")
echo "  M01 -> ${ID_MAP[M01]:0:12}..."

ID_MAP[M02]=$(store "M02" \
    "Architecture decision: We chose PostgreSQL over MongoDB for the analytics pipeline. Key reasons were complex joins needed for funnel analysis, ACID compliance for financial data, and the team existing expertise. Marcus strongly advocated for Mongo but the vote was 4-1 in favor of Postgres." \
    '"architecture","database","decision","analytics"' \
    "Decision" "user" "$(mk_ts 1)" "0.2" "0.3")
echo "  M02 -> ${ID_MAP[M02]:0:12}..."

ID_MAP[M03]=$(store "M03" \
    "Bug report: Users in the EU region are experiencing 5-second delays on the checkout page. The root cause is the payment provider EU endpoint having high latency. Temporary fix: route EU traffic through the UK proxy. Permanent fix needs Stripe v3 migration which Sarah is working on." \
    '"bug","performance","checkout","EU","latency"' \
    "Observation" "system" "$(mk_ts 2)" "-0.5" "0.7")
echo "  M03 -> ${ID_MAP[M03]:0:12}..."

ID_MAP[M04]=$(store "M04" \
    "Personal preference: I strongly prefer dark mode in all development tools. My IDE theme is Dracula, terminal is Catppuccin Mocha, and I use the Fira Code font with ligatures enabled at 14pt." \
    '"preference","tooling","IDE","theme"' \
    "Preference" "user" "$(mk_ts 3)" "0.6" "0.2")
echo "  M04 -> ${ID_MAP[M04]:0:12}..."

ID_MAP[M05]=$(store "M05" \
    "Raj found the Redis cache invalidation bug. It was a race condition in the pub/sub listener. When two nodes receive the same invalidation event, they both try to refresh the cache simultaneously, causing a thundering herd. Fix: added distributed locking with Redlock." \
    '"bug-fix","redis","cache","race-condition","Raj"' \
    "Observation" "user" "$(mk_ts 7)" "0.5" "0.5")
echo "  M05 -> ${ID_MAP[M05]:0:12}..."

ID_MAP[M06]=$(store "M06" \
    "Team lunch at Sakura Sushi near the downtown office. Everyone was there except Marcus who was working remotely from Portland. Sarah mentioned she is thinking about transitioning to a principal engineer role. Good team morale overall." \
    '"social","team","lunch","Sakura-Sushi"' \
    "Observation" "user" "$(mk_ts 8 12)" "0.7" "0.3")
echo "  M06 -> ${ID_MAP[M06]:0:12}..."

ID_MAP[M07]=$(store "M07" \
    "The onboarding wizard prototype is working. It has 5 steps: account creation, team setup, first project, integration connections, and a guided tour. User testing showed 78 percent completion rate, up from 45 percent with the old flow. VP of Product Elena approved moving to production." \
    '"onboarding","prototype","user-testing","Elena"' \
    "Observation" "user" "$(mk_ts 9)" "0.6" "0.5")
echo "  M07 -> ${ID_MAP[M07]:0:12}..."

ID_MAP[M08]=$(store "M08" \
    "Decision to use Kubernetes over ECS for the new microservices deployment. Reasoning: better multi-cloud portability, stronger community tooling like Helm and ArgoCD, and our SRE team already has K8s experience from the data pipeline project. Cost estimate is 2400 dollars per month higher but worth the flexibility." \
    '"infrastructure","kubernetes","deployment","decision"' \
    "Decision" "user" "$(mk_ts 10)" "0.1" "0.3")
echo "  M08 -> ${ID_MAP[M08]:0:12}..."

ID_MAP[M09]=$(store "M09" \
    "Production incident at 3 AM: the analytics database ran out of disk space. Root cause was the funnel_events table growing 10x faster than projected because of a missing TTL policy. We added a 90-day retention policy and partitioned by month. Total downtime was 47 minutes." \
    '"incident","database","disk-space","analytics","downtime"' \
    "Observation" "system" "$(mk_ts 14 3)" "-0.7" "0.9")
echo "  M09 -> ${ID_MAP[M09]:0:12}..."

ID_MAP[M10]=$(store "M10" \
    "Sarah completed the Stripe v3 migration for the US region. Performance improved by 40 percent. Average checkout time dropped from 3.2s to 1.9s. EU migration is scheduled for next week. She used the new Stripe Payment Intents API which also enables Apple Pay and Google Pay." \
    '"stripe","migration","performance","Sarah","checkout"' \
    "Observation" "user" "$(mk_ts 15)" "0.7" "0.5")
echo "  M10 -> ${ID_MAP[M10]:0:12}..."

ID_MAP[M11]=$(store "M11" \
    "Conference talk accepted: I will be presenting Building Resilient Caching at Scale at RustConf 2026 in Austin, Texas on September 15th. The talk covers our Redis architecture evolution, the thundering herd fix, and distributed locking patterns." \
    '"conference","RustConf","talk","Austin"' \
    "Observation" "user" "$(mk_ts 16)" "0.8" "0.7")
echo "  M11 -> ${ID_MAP[M11]:0:12}..."

ID_MAP[M12]=$(store "M12" \
    "Code review feedback from Marcus on the onboarding wizard: he flagged that the integration step makes 12 sequential API calls which could be parallelized. Good catch. Refactored to use Promise.all and reduced step 4 load time from 4.5s to 0.8s." \
    '"code-review","Marcus","onboarding","performance"' \
    "Observation" "user" "$(mk_ts 17)" "0.4" "0.3")
echo "  M12 -> ${ID_MAP[M12]:0:12}..."

ID_MAP[M13]=$(store "M13" \
    "Q3 OKR planning meeting. Our team key results: reduce p95 API latency from 800ms to 200ms, achieve 99.95 percent uptime SLA, launch self-serve enterprise onboarding. VP Elena emphasized that the enterprise onboarding is the highest revenue priority." \
    '"OKR","Q3","planning","latency","uptime","enterprise"' \
    "Observation" "user" "$(mk_ts 21)" "0.2" "0.4")
echo "  M13 -> ${ID_MAP[M13]:0:12}..."

ID_MAP[M14]=$(store "M14" \
    "Security audit findings: The penetration test by CrowdStrike found 3 medium-severity issues: missing rate limiting on the password reset endpoint, CORS misconfiguration allowing wildcard origins in staging, session tokens not rotated after privilege escalation. All assigned to Sprint 15." \
    '"security","audit","CrowdStrike","vulnerabilities"' \
    "Observation" "system" "$(mk_ts 22)" "-0.3" "0.6")
echo "  M14 -> ${ID_MAP[M14]:0:12}..."

ID_MAP[M15]=$(store "M15" \
    "Raj is transferring to the ML platform team next month. His replacement on our team will be Priya, who has 5 years of experience with distributed systems at Netflix. She starts on April 15th. Raj will do a 2-week knowledge transfer on the caching layer." \
    '"team-change","Raj","Priya","transfer","Netflix"' \
    "Observation" "user" "$(mk_ts 23)" "0.0" "0.4")
echo "  M15 -> ${ID_MAP[M15]:0:12}..."

ID_MAP[M16]=$(store "M16" \
    "Decided to adopt GraphQL for the new enterprise API instead of REST. Reasons: clients need flexible field selection for dashboard customization, reduces over-fetching which is critical for mobile, and Apollo Federation allows us to stitch microservice schemas. Timeline: MVP by end of Q3." \
    '"architecture","GraphQL","enterprise","API","decision"' \
    "Decision" "user" "$(mk_ts 24)" "0.3" "0.3")
echo "  M16 -> ${ID_MAP[M16]:0:12}..."

ID_MAP[M17]=$(store "M17" \
    "The annual company hackathon is on April 20-21. Our team is building a real-time collaboration feature using CRDTs and WebSockets. I am excited because this could become a real product feature. Last year winning team built the AI-powered search that is now in production." \
    '"hackathon","CRDT","collaboration","WebSocket"' \
    "Observation" "user" "$(mk_ts 25)" "0.7" "0.6")
echo "  M17 -> ${ID_MAP[M17]:0:12}..."

ID_MAP[M18]=$(store "M18" \
    "Performance benchmark results for the new analytics pipeline on PostgreSQL: 500K events per minute ingestion, p99 query latency at 120ms for 30-day windows, and 45ms for 7-day windows. This exceeds our Q3 target of 200ms p95. The columnar extension TimescaleDB was key to this performance." \
    '"benchmark","analytics","PostgreSQL","TimescaleDB","performance"' \
    "Observation" "system" "$(mk_ts 26)" "0.6" "0.4")
echo "  M18 -> ${ID_MAP[M18]:0:12}..."

ID_MAP[M19]=$(store "M19" \
    "Sarah Stripe v3 EU migration went live today. Checkout latency for EU users dropped from 5 seconds to 1.4 seconds. The bug from two weeks ago about EU checkout delays is now fully resolved. Apple Pay and Google Pay are also enabled for EU customers." \
    '"stripe","EU","migration","Sarah","checkout","resolved"' \
    "Observation" "user" "$(mk_ts 27)" "0.8" "0.5")
echo "  M19 -> ${ID_MAP[M19]:0:12}..."

ID_MAP[M20]=$(store "M20" \
    "End-of-sprint retro for Sprint 14. Completed 31 of 34 story points. Carried over: the enterprise SSO integration blocked by IdP provider delays and two minor UI polish tasks. Team velocity trending up with average of last 3 sprints at 30 points. Celebrating at Happy Hour Friday at The Rusty Anchor." \
    '"retro","sprint-14","velocity","team"' \
    "Observation" "user" "$(mk_ts 28)" "0.5" "0.3")
echo "  M20 -> ${ID_MAP[M20]:0:12}..."

echo "  Stored ${#ID_MAP[@]} memories. Waiting 2s for indexing..."
sleep 2

echo ""
echo "[3/4] Running 20 queries..."

# Initialize results tracking in a temp file
SCORE_FILE=$(mktemp)
echo "" > "$SCORE_FILE"

# Query function: returns rank of first match (0 if no match in top 10)
# Usage: run_query "category" "query" "expected_tag1 expected_tag2 ..."
run_query() {
    local num="$1" category="$2" query="$3"
    shift 3
    local expected_tags=("$@")

    local ranked_ids
    ranked_ids=$(recall_ids "$query" 10)

    # Convert ranked IDs to tags
    local rank=0
    local first_hit_rank=0
    local hits_at_5=0
    local hits_at_10=0
    local total_expected=${#expected_tags[@]}
    local found_tags=""
    local top3_tags=""

    # Build reverse map and check
    local i=0
    while IFS= read -r rid; do
        i=$((i + 1))
        # Find which tag this ID matches
        local tag="?"
        for t in "${!ID_MAP[@]}"; do
            if [ "${ID_MAP[$t]}" = "$rid" ]; then
                tag="$t"
                break
            fi
        done

        if [ "$i" -le 3 ]; then
            top3_tags="${top3_tags}${tag} "
        fi

        # Check if this is an expected tag
        for et in "${expected_tags[@]}"; do
            if [ "$tag" = "$et" ]; then
                if [ "$first_hit_rank" -eq 0 ]; then
                    first_hit_rank=$i
                fi
                if [ "$i" -le 5 ]; then
                    hits_at_5=$((hits_at_5 + 1))
                fi
                if [ "$i" -le 10 ]; then
                    hits_at_10=$((hits_at_10 + 1))
                fi
                found_tags="${found_tags}${tag} "
                break
            fi
        done
    done <<< "$ranked_ids"

    # Compute MRR
    local mrr="0.0000"
    if [ "$first_hit_rank" -gt 0 ]; then
        mrr=$(/usr/bin/python3 -c "print(f'{1.0/$first_hit_rank:.4f}')")
    fi

    # Compute R@5 and R@10
    local r5=$(/usr/bin/python3 -c "print(f'{$hits_at_5/$total_expected:.4f}')")
    local r10=$(/usr/bin/python3 -c "print(f'{$hits_at_10/$total_expected:.4f}')")

    # Compute Hit@5 and Hit@10
    local h5=0 h10=0
    [ "$hits_at_5" -gt 0 ] && h5=1
    [ "$hits_at_10" -gt 0 ] && h10=1

    local status="MISS"
    [ "$first_hit_rank" -gt 0 ] && status="HIT "

    printf "  Q%02d [%-11s] MRR=%s R@5=%s R@10=%s %s | expected=[%s] found=[%s] | top3=[%s]\n" \
        "$num" "$category" "$mrr" "$r5" "$r10" "$status" \
        "$(echo "${expected_tags[*]}" | tr ' ' ',')" \
        "$(echo "$found_tags" | xargs | tr ' ' ',')" \
        "$(echo "$top3_tags" | xargs | tr ' ' ',')"

    # Append to score file: category mrr h5 h10 r5 r10
    echo "$category $mrr $h5 $h10 $r5 $r10" >> "$SCORE_FILE"
}

# === SINGLE-HOP ===
run_query 1 "single-hop" "What database did we choose for the analytics pipeline?" M02
run_query 2 "single-hop" "What was the root cause of the Redis cache invalidation bug?" M05
run_query 3 "single-hop" "What is my preferred IDE theme and font?" M04
run_query 4 "single-hop" "How many story points did we commit to in Sprint 14?" M01
run_query 5 "single-hop" "What were the security vulnerabilities found in the penetration test?" M14

# === TEMPORAL ===
run_query 6 "temporal" "What happened during the production incident that caused downtime?" M09
run_query 7 "temporal" "When is my conference talk and what is it about?" M11
run_query 8 "temporal" "When does Priya start and who is she replacing?" M15
run_query 9 "temporal" "What are the dates for the company hackathon?" M17
run_query 10 "temporal" "When was the EU checkout latency issue finally resolved?" M19

# === MULTI-HOP ===
run_query 11 "multi-hop" "How did Sarah Stripe migration fix the EU checkout delay problem?" M03 M10 M19
run_query 12 "multi-hop" "What caching problems did we have and how were they resolved?" M01 M05
run_query 13 "multi-hop" "Which team member reviewed the onboarding wizard and what did they find?" M07 M12
run_query 14 "multi-hop" "What is Raj working on now and what happens when he leaves?" M05 M15
run_query 15 "multi-hop" "How do our analytics benchmark numbers compare to the Q3 OKR targets?" M13 M18

# === OPEN-DOMAIN ===
run_query 16 "open-domain" "What is the team overall morale and social dynamics like?" M06 M20
run_query 17 "open-domain" "What are the most important strategic priorities for the team?" M13 M16
run_query 18 "open-domain" "What infrastructure and deployment decisions have we made recently?" M02 M08 M16
run_query 19 "open-domain" "What performance improvements have we achieved across the product?" M05 M07 M10 M12 M18 M19
run_query 20 "open-domain" "Who are the key people on the team and what are they responsible for?" M01 M05 M06 M10 M15

echo ""
echo "[4/4] Results"
echo "========================================================================"

# Compute per-category and overall metrics
/usr/bin/python3 << 'PYEOF'
import sys

scores = []
with open("SCORE_FILE_PATH", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 6:
            cat, mrr, h5, h10, r5, r10 = parts
            scores.append({
                "cat": cat,
                "mrr": float(mrr),
                "h5": float(h5),
                "h10": float(h10),
                "r5": float(r5),
                "r10": float(r10),
            })

categories = ["single-hop", "temporal", "multi-hop", "open-domain"]

print(f"\n{'Category':<14s} {'MRR':>7s} {'Hit@5':>7s} {'Hit@10':>7s} {'R@5':>7s} {'R@10':>7s}  n")
print("-" * 72)

all_mrr, all_h5, all_h10, all_r5, all_r10 = [], [], [], [], []

for cat in categories:
    items = [s for s in scores if s["cat"] == cat]
    n = len(items)
    if n == 0:
        continue
    avg = lambda key: sum(s[key] for s in items) / n
    mrr = avg("mrr")
    h5 = avg("h5")
    h10 = avg("h10")
    r5 = avg("r5")
    r10 = avg("r10")
    print(f"  {cat:<12s} {mrr:>7.4f} {h5:>7.4f} {h10:>7.4f} {r5:>7.4f} {r10:>7.4f}  {n}")
    all_mrr.extend(s["mrr"] for s in items)
    all_h5.extend(s["h5"] for s in items)
    all_h10.extend(s["h10"] for s in items)
    all_r5.extend(s["r5"] for s in items)
    all_r10.extend(s["r10"] for s in items)

n = len(scores)
if n > 0:
    print("-" * 72)
    o_mrr = sum(all_mrr) / n
    o_h5 = sum(all_h5) / n
    o_h10 = sum(all_h10) / n
    o_r5 = sum(all_r5) / n
    o_r10 = sum(all_r10) / n
    print(f"  {'OVERALL':<12s} {o_mrr:>7.4f} {o_h5:>7.4f} {o_h10:>7.4f} {o_r5:>7.4f} {o_r10:>7.4f}  {n}")

    # Misses
    misses = [s for s in scores if s["mrr"] == 0.0]
    if misses:
        print(f"\n  Misses: {len(misses)} queries returned no expected memory in top 10")

    print()
    print("=" * 72)
    print(f"Overall MRR: {o_mrr:.4f} | Hit@5: {o_h5:.4f} | Hit@10: {o_h10:.4f}")
    print("=" * 72)
PYEOF
PYEOF
