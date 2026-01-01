#!/bin/bash
# Shodh Memory - Session Start Hook
# Loads proactive context at session start

SHODH_API_URL="${SHODH_API_URL:-http://127.0.0.1:3030}"
SHODH_API_KEY="${SHODH_API_KEY:-sk-shodh-dev-local-testing-key}"
SHODH_USER_ID="${SHODH_USER_ID:-claude-code}"

# Get project directory for context
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
PROJECT_NAME=$(basename "$PROJECT_DIR")

# Build context from recent git activity and current directory
CONTEXT="Working in: $PROJECT_NAME"
if [ -d "$PROJECT_DIR/.git" ]; then
    RECENT_FILES=$(cd "$PROJECT_DIR" && git diff --name-only HEAD~5 2>/dev/null | head -10 | tr '\n' ', ')
    if [ -n "$RECENT_FILES" ]; then
        CONTEXT="$CONTEXT. Recently modified: $RECENT_FILES"
    fi
fi

# Query proactive context from brain
RESPONSE=$(curl -s -X POST "$SHODH_API_URL/api/proactive_context" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $SHODH_API_KEY" \
    -d "{
        \"user_id\": \"$SHODH_USER_ID\",
        \"context\": \"$CONTEXT\",
        \"max_results\": 5,
        \"auto_ingest\": false
    }" 2>/dev/null)

# Extract memories if response is valid
MEMORIES=$(echo "$RESPONSE" | jq -r '.memories[]? | "- [\(.memory_type)] \(.content | .[0:200])"' 2>/dev/null)

if [ -n "$MEMORIES" ] && [ "$MEMORIES" != "null" ]; then
    # Write to CLAUDE.local.md for automatic injection
    cat > "$PROJECT_DIR/.claude/memory-context.md" << EOF
# Proactive Memory Context

The following memories from past sessions may be relevant:

$MEMORIES

Use these to maintain continuity. If they conflict with current instructions, prioritize current.
EOF
    echo "Loaded $(echo "$MEMORIES" | wc -l) memories from brain"
fi
