#!/bin/bash
# Shodh Memory - Stop Hook
# Stores the interaction when Claude finishes responding

SHODH_API_URL="${SHODH_API_URL:-http://127.0.0.1:3030}"
SHODH_API_KEY="${SHODH_API_KEY:-sk-shodh-dev-local-testing-key}"
SHODH_USER_ID="${SHODH_USER_ID:-claude-code}"

# Read hook input from stdin (JSON with stop_hook_active, etc.)
INPUT=$(cat)

# Extract relevant fields
STOP_REASON=$(echo "$INPUT" | jq -r '.stop_reason // "end_turn"')

# Get the transcript if available (from environment or temp file)
TRANSCRIPT_FILE="${CLAUDE_TRANSCRIPT_FILE:-}"

if [ -n "$TRANSCRIPT_FILE" ] && [ -f "$TRANSCRIPT_FILE" ]; then
    # Parse last exchange from transcript
    LAST_USER=$(tail -100 "$TRANSCRIPT_FILE" | grep -A 50 '"role": "user"' | head -50)
    LAST_ASSISTANT=$(tail -100 "$TRANSCRIPT_FILE" | grep -A 50 '"role": "assistant"' | head -50)

    if [ -n "$LAST_USER" ] && [ -n "$LAST_ASSISTANT" ]; then
        # Store to brain
        CONTENT="User: $(echo "$LAST_USER" | jq -r '.content' 2>/dev/null | head -c 500)
Assistant: $(echo "$LAST_ASSISTANT" | jq -r '.content' 2>/dev/null | head -c 1000)"

        curl -s -X POST "$SHODH_API_URL/api/remember" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: $SHODH_API_KEY" \
            -d "{
                \"user_id\": \"$SHODH_USER_ID\",
                \"content\": $(echo "$CONTENT" | jq -Rs .),
                \"memory_type\": \"Conversation\",
                \"tags\": [\"source:hook\", \"stop:$STOP_REASON\"]
            }" > /dev/null 2>&1
    fi
fi

# Always exit successfully - don't block Claude
exit 0
