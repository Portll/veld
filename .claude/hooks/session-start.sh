#!/bin/bash
# Session start hook - restore context from shodh-memory

API_URL="${SHODH_API_URL:-http://127.0.0.1:3030}"
API_KEY="${SHODH_API_KEY:-sk-shodh-dev-local-testing-key}"
USER_ID="${SHODH_USER_ID:-claude-code}"

# Get proactive context based on current directory
DIR_NAME=$(basename "$CLAUDE_PROJECT_DIR")

CONTEXT=$(curl -s -X POST "$API_URL/api/proactive_context" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"user_id\": \"$USER_ID\", \"context\": \"Starting session in $DIR_NAME\", \"max_results\": 3, \"auto_ingest\": false}" \
  2>/dev/null)

# Get pending todos
TODOS=$(curl -s -X POST "$API_URL/api/todos" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"user_id\": \"$USER_ID\", \"status\": [\"todo\", \"in_progress\"]}" \
  2>/dev/null)

# Output as JSON with additionalContext
cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "## Shodh Memory Context Restored\n\n### Relevant Memories:\n$CONTEXT\n\n### Pending Todos:\n$TODOS"
  }
}
EOF
