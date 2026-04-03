#!/bin/bash
# session-start.sh — Shodh Memory session start hook for shodh-memory project
# Runs on every SessionStart event (new, resume, compact, clear).
# Auto-starts the shodh-memory-server binary if down.
# Compaction = critical save point (checkpoint before re-inject).

set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source claude-agent-lib for identity/presence
source ~/.claude/hooks/claude-agent-lib.sh 2>/dev/null || true

# Read stdin (hook JSON payload)
INPUT=$(cat)

# Extract session source (startup | resume | compact | clear)
SOURCE=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('source', 'startup'))
except:
    print('startup')
" 2>/dev/null)

# Source shodh-lib (sets SHODH_URL, SHODH_API_KEY, SHODH_USER_ID)
source "$SCRIPT_DIR/shodh-lib.sh" 2>/dev/null || true

# ── Auto-Start Server if Down ─────────────────────────────────
if ! curl -sf --max-time 2 "$SHODH_URL/health" > /dev/null 2>&1; then
  # Prefer release binary, fall back to debug
  SHODH_BIN=""
  for candidate in \
    "$PROJECT_DIR/target/release/shodh-memory-server" \
    "$PROJECT_DIR/target/debug/shodh-memory-server"; do
    if [ -x "$candidate" ]; then
      SHODH_BIN="$candidate"
      break
    fi
  done

  if [ -n "$SHODH_BIN" ]; then
    SHODH_API_KEYS="$SHODH_API_KEY" nohup "$SHODH_BIN" \
      --port 3030 \
      --storage "$PROJECT_DIR/shodh_memory_data" \
      > /tmp/shodh-memory.log 2>&1 &
    # Wait up to 8s for health
    for i in 1 2 3 4 5 6 7 8; do
      sleep 1
      curl -sf --max-time 1 "$SHODH_URL/health" > /dev/null 2>&1 && break
    done
    # Re-check — reset SHODH_HEALTHY so shodh_health() re-probes
    SHODH_HEALTHY=""
    shodh_health 2>/dev/null || exit 0
  else
    exit 0
  fi
else
  SHODH_HEALTHY="1"
fi

# ── Verify Index Integrity ─────────────────────────────────────
VERIFY_RESULT=$(curl -sf --max-time 5 -X POST "$SHODH_URL/api/verify_index" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d "{\"user_id\":\"$SHODH_USER_ID\"}" 2>/dev/null)
if [ -n "$VERIFY_RESULT" ]; then
  ORPHAN_COUNT=$(echo "$VERIFY_RESULT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('orphaned_count', 0))
except:
    print(0)
" 2>/dev/null)
  if [ "${ORPHAN_COUNT:-0}" -gt "0" ]; then
    export SHODH_INDEX_WARNING="[shodh] Index desync: $ORPHAN_COUNT orphaned memories. Run repair_index to fix."
  fi
fi

# ── Register Presence ────────────────────────────────────────
presence_register 2>/dev/null || true

# ── Compaction Checkpoint (SAVE before re-inject) ────────────
if [ "$SOURCE" = "compact" ]; then
  BRANCH=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null || echo "unknown")
  MODIFIED=$(git -C "$PROJECT_DIR" diff --name-only 2>/dev/null | head -10 | tr '\n' ', ')
  ACTIVE=$(python3 -c "
import json
try:
    with open('$CLAUDE_PRESENCE_DIR/$CLAUDE_AGENT_PID.json') as f:
        print(', '.join(json.load(f).get('active_files', [])))
except:
    print('none')
" 2>/dev/null)
  EDIT_COUNT=$(wc -l < "$CLAUDE_ACTIVITY_LOG" 2>/dev/null | tr -d ' ' || echo 0)

  shodh_upsert \
    "Mid-session checkpoint (compaction). Branch: $BRANCH. $EDIT_COUNT edits so far. Active files: $ACTIVE. Modified: $MODIFIED" \
    "Context" \
    '["compaction-checkpoint","session-state"]' \
    "compact-$CLAUDE_AGENT_PID-$(date +%Y%m%d-%H%M)" 2>/dev/null
fi

# ── Sibling Awareness ────────────────────────────────────────
SIBLINGS=$(presence_siblings 2>/dev/null)

# ── Adjust limits based on session source ────────────────────
case "$SOURCE" in
  compact)  SUM_LIMIT=100; REL_LIMIT=3; TAG_LIMIT=4 ;;
  clear)    SUM_LIMIT=50;  REL_LIMIT=2; TAG_LIMIT=2 ;;
  *)        SUM_LIMIT=200; REL_LIMIT=5; TAG_LIMIT=8 ;;
esac

TMPSUM="/tmp/shodh_sum_$$.json"
TMPREL="/tmp/shodh_rel_$$.json"
TMPTAGS="/tmp/shodh_tags_$$.json"

# ── Git-aware context for /api/relevant ──────────────────────
BRANCH=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null || echo "unknown")
RECENT=$(git -C "$PROJECT_DIR" log --oneline -3 2>/dev/null | tr '\n' '; ')
REL_CONTEXT="Branch: $BRANCH. Recent commits: $RECENT"

# Three parallel fetches
curl -s --max-time 10 -X POST "$SHODH_URL/api/context_summary" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d "{\"user_id\":\"$SHODH_USER_ID\",\"limit\":$SUM_LIMIT}" > "$TMPSUM" 2>/dev/null &

curl -s --max-time 15 -X POST "$SHODH_URL/api/relevant" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d "{\"user_id\":\"$SHODH_USER_ID\",\"context\":\"$REL_CONTEXT\",\"limit\":$REL_LIMIT}" > "$TMPREL" 2>/dev/null &

curl -s --max-time 10 -X POST "$SHODH_URL/api/recall/tags" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d "{\"user_id\":\"$SHODH_USER_ID\",\"tags\":[\"architecture\",\"retrieval\",\"memory-system\",\"regression\",\"struggle-file\",\"shodh-memory\",\"embeddings\",\"graph\",\"benchmark\",\"session-summary\"],\"limit\":$TAG_LIMIT}" > "$TMPTAGS" 2>/dev/null &

# Overlook: fetch prior AVs for cross-session learning
TMPOVERLOOK="/tmp/overlook-priors-$CLAUDE_AGENT_PID.json"
curl -s --max-time 5 -X POST "$SHODH_URL/api/recall/tags" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d "{\"user_id\":\"$SHODH_USER_ID\",\"tags\":[\"overlook-av\",\"session-end\"],\"limit\":3}" \
  > "$TMPOVERLOOK" 2>/dev/null &

wait

# Overlook: initialize AV from priors
if [ -f "$HOME/.claude/hooks/overlook-lib.sh" ]; then
  source "$HOME/.claude/hooks/overlook-lib.sh" 2>/dev/null
  overlook_init "$TMPOVERLOOK"
  rm -f "$TMPOVERLOOK"
fi

# ── Cache file patterns for PreToolUse ───────────────────────
git -C "$PROJECT_DIR" diff --name-only HEAD~3 2>/dev/null | head -15 | while IFS= read -r f; do
  [ -n "$f" ] && shodh_file_patterns "$PROJECT_DIR/$f" 2>/dev/null
done

# ── Consolidation Trigger (every 10th session) ───────────────
COUNT_FILE="/tmp/shodh_session_count_${CLAUDE_PROJECT_HASH:-default}"
COUNT=$(($(cat "$COUNT_FILE" 2>/dev/null || echo 0) + 1))
echo "$COUNT" > "$COUNT_FILE"
if [ $((COUNT % 10)) -eq 0 ]; then
  curl -s --max-time 5 -X POST "$SHODH_URL/api/consolidation/report" \
    -H "Content-Type: application/json" -H "X-API-Key: $SHODH_API_KEY" \
    -d "{\"user_id\":\"$SHODH_USER_ID\"}" > /dev/null 2>&1 &
fi

# ── Format Output ────────────────────────────────────────────
python3 - "$TMPSUM" "$TMPREL" "$TMPTAGS" "$SOURCE" "$SIBLINGS" << 'PYEOF'
import json, sys, os

sum_file, rel_file, tags_file, source = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
siblings = sys.argv[5] if len(sys.argv) > 5 else ""
lines = [f"=== SHODH MEMORY (session: {source}) ==="]

# 0. Sibling agents (if any)
if siblings and "No active" not in siblings:
    lines.append(f"\n{siblings}")

# 1. Context summary
try:
    with open(sum_file) as f:
        s = json.load(f)
    lines.append(f"[{s.get('total_memories', '?')} memories]")
    for cat, label in [("decisions", "Decisions"), ("patterns", "Patterns"), ("errors", "Known Issues")]:
        items = s.get(cat, [])
        if items:
            lines.append(f"\n{label}:")
            for m in items[:3]:
                c = m.get("content", "")[:300]
                lines.append(f"  - {c}")
except:
    pass

# 2. Relevant memories (skip low-relevance Learnings)
try:
    with open(rel_file) as f:
        r = json.load(f)
    mems = r.get("memories", [])
    good = [m for m in mems
            if m.get("relevance_score", 0) > 0.55
            or m.get("memory_type", "") in ("Decision", "Pattern", "Error")]
    if good:
        lines.append(f"\nRelevant context:")
        for m in good[:4]:
            mt = m.get("memory_type", "?")
            score = m.get("relevance_score", 0)
            lines.append(f"  [{mt} {score:.2f}] {m.get('content', '')[:250]}")
except:
    pass

# 3. Tagged memories (architectural knowledge)
try:
    with open(tags_file) as f:
        t = json.load(f)
    mems = t.get("memories", [])
    if mems:
        lines.append(f"\nArchitectural notes:")
        seen = set()
        for m in mems[:4]:
            c = m.get("content", "")
            if not c:
                c = m.get("experience", {}).get("content", "")
            c = c[:250]
            key = c[:80]
            if key and key not in seen:
                seen.add(key)
                lines.append(f"  - {c}")
except:
    pass

result = "\n".join(lines)
index_warning = os.environ.get("SHODH_INDEX_WARNING", "")
if index_warning:
    result = index_warning + "\n\n" + result
print(json.dumps({"additionalContext": result}))
PYEOF

rm -f "$TMPSUM" "$TMPREL" "$TMPTAGS"
