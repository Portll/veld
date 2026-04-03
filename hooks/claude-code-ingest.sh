#!/bin/bash
# Shodh-Memory Auto-Ingest Hook for Claude Code
#
# This hook runs after each Claude Code response to automatically
# store conversations in shodh-memory for persistent learning.
#
# Installation:
#   1. Copy this file to ~/.claude/hooks/
#   2. Add to ~/.claude/settings.json:
#      {
#        "hooks": {
#          "Stop": [{
#            "matcher": "",
#            "hooks": [{
#              "type": "command",
#              "command": "bash ~/.claude/hooks/claude-code-ingest.sh"
#            }]
#          }]
#        }
#      }
#   3. Ensure veld is running on localhost:3030
#
# Environment Variables:
#   SHODH_API_URL - API endpoint (default: http://127.0.0.1:3030)
#   SHODH_API_KEY - API key (default: dev key)
#   SHODH_USER_ID - User ID for memory isolation (default: claude-code)

set -e

API_URL="${SHODH_API_URL:-http://127.0.0.1:3030}"
USER_ID="${SHODH_USER_ID:-claude-code}"

# API Key - required (no hardcoded fallback for security)
if [ -z "$SHODH_API_KEY" ]; then
    echo "ERROR: SHODH_API_KEY environment variable not set" >&2
    exit 1
fi
API_KEY="$SHODH_API_KEY"

# Read hook input from stdin
INPUT=$(cat)

# Extract transcript path from hook input
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')

if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
    exit 0
fi

# Extract the last exchange (user message + assistant response)
LAST_MESSAGES=$(jq -c '.[-2:]' "$TRANSCRIPT_PATH" 2>/dev/null || echo "[]")

if [ "$LAST_MESSAGES" = "[]" ] || [ "$LAST_MESSAGES" = "null" ]; then
    exit 0
fi

# Format conversation content - extract text from both user and assistant messages
CONTENT=$(echo "$LAST_MESSAGES" | jq -r '
  map(
    if .role == "user" then
      "User: " + (.content | if type == "array" then map(select(.type == "text") | .text) | join("\n") else tostring end)
    elif .role == "assistant" then
      "Assistant: " + (.content | if type == "array" then map(select(.type == "text") | .text) | join("\n") else tostring end)
    else
      empty
    end
  ) | join("\n\n")
' 2>/dev/null || echo "")

# Skip if content is empty or too short (< 50 chars = likely noise)
if [ -z "$CONTENT" ] || [ ${#CONTENT} -lt 50 ]; then
    exit 0
fi

# Truncate if too long (max 4000 chars for a single memory)
if [ ${#CONTENT} -gt 4000 ]; then
    CONTENT="${CONTENT:0:4000}..."
fi

# Escape content for JSON
CONTENT_ESCAPED=$(echo "$CONTENT" | jq -Rs '.')

# Extract project context from working directory
CWD=$(echo "$INPUT" | jq -r '.cwd // "unknown"')
PROJECT=$(basename "$CWD")

# Send to shodh-memory API (fire and forget, don't block Claude)
curl -s -X POST "$API_URL/api/record" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    --connect-timeout 2 \
    --max-time 5 \
    -d "{
        \"user_id\": \"$USER_ID\",
        \"experience\": {
            \"content\": $CONTENT_ESCAPED,
            \"experience_type\": \"Conversation\",
            \"tags\": [\"claude-code\", \"auto-ingest\", \"$PROJECT\"]
        }
    }" > /dev/null 2>&1 || true

exit 0
