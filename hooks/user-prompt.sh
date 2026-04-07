#!/bin/bash
# Veld - User Prompt Submit Hook
# Enriches context based on what user is asking

VELD_API_URL="${VELD_API_URL:-http://127.0.0.1:3030}"
VELD_API_KEY="${VELD_API_KEY:-sk-veld-dev-local-testing-key}"
VELD_USER_ID="${VELD_USER_ID:-claude-code}"

# Read hook input from stdin
INPUT=$(cat)

# Extract the prompt
PROMPT=$(echo "$INPUT" | jq -r '.prompt // ""')

# Skip if empty or very short
if [ ${#PROMPT} -lt 10 ]; then
    exit 0
fi

# Query brain for relevant memories based on this specific prompt
RESPONSE=$(curl -s -X POST "$VELD_API_URL/api/recall" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $VELD_API_KEY" \
    -d "{
        \"user_id\": \"$VELD_USER_ID\",
        \"query\": $(echo "$PROMPT" | head -c 500 | jq -Rs .),
        \"limit\": 3
    }" 2>/dev/null)

# Extract memories
MEMORIES=$(echo "$RESPONSE" | jq -r '.results[]? | "[\(.memory_type)] \(.content | .[0:150])..."' 2>/dev/null | head -3)

if [ -n "$MEMORIES" ] && [ "$MEMORIES" != "null" ]; then
    # Output additional context (Claude Code will inject this)
    echo "{\"additionalContext\": \"Relevant memories: $MEMORIES\"}"
fi

exit 0
