#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
MOCK_BIN="$TMP_DIR/bin"
mkdir -p "$MOCK_BIN"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

pass() {
  echo "ok - $1"
}

fail() {
  echo "not ok - $1"
  exit 1
}

# Mock curl that returns JSON expected by scripts.
cat > "$MOCK_BIN/curl" <<'EOF'
#!/bin/bash
if [ -n "${MOCK_CURL_LOG:-}" ]; then
  echo "$*" >> "$MOCK_CURL_LOG"
fi
if echo "$*" | grep -q "/api/proactive_context"; then
  cat <<JSON
{"memories":[{"memory_type":"Task","content":"Remember to run migration"}]}
JSON
elif echo "$*" | grep -q "/api/recall"; then
  cat <<JSON
{"results":[{"memory_type":"Learning","content":"Past fix for timeout"}]}
JSON
elif echo "$*" | grep -q "/api/remember"; then
  echo '{"id":"mem-1"}'
else
  echo "{}"
fi
EOF
chmod +x "$MOCK_BIN/curl"

# Mock jq for the script selectors used.
cat > "$MOCK_BIN/jq" <<'EOF'
#!/bin/bash
query="$*"
input="$(cat)"
if echo "$query" | grep -q "\.prompt // \"\""; then
  # extract prompt value from simple JSON payload
  echo "$input" | sed -n 's/.*"prompt"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
  exit 0
fi
if echo "$query" | grep -q "\.memories\[\]\?"; then
  if echo "$input" | grep -q "Remember to run migration"; then
    echo "- [Task] Remember to run migration"
  fi
  exit 0
fi
if echo "$query" | grep -q "\.results\[\]\?"; then
  if echo "$input" | grep -q "Past fix for timeout"; then
    echo "[Learning] Past fix for timeout..."
  fi
  exit 0
fi
if echo "$query" | grep -q "\.stop_reason"; then
  echo "end_turn"
  exit 0
fi
if echo "$query" | grep -q "\.content"; then
  echo "content"
  exit 0
fi
# fallback raw
cat
EOF
chmod +x "$MOCK_BIN/jq"

# Mock git for session-start recent files branch.
cat > "$MOCK_BIN/git" <<'EOF'
#!/bin/bash
echo "src/main.ts"
EOF
chmod +x "$MOCK_BIN/git"

export PATH="$MOCK_BIN:$PATH"
export MOCK_CURL_LOG="$TMP_DIR/curl.log"

# 1) session-start should create context file when memories exist.
PROJECT_DIR="$TMP_DIR/project"
mkdir -p "$PROJECT_DIR/.claude" "$PROJECT_DIR/.git"
CLAUDE_PROJECT_DIR="$PROJECT_DIR" bash "$ROOT_DIR/session-start.sh" >/dev/null 2>&1 || fail "session-start exits non-zero"
if [ -f "$PROJECT_DIR/.claude/memory-context.md" ]; then
  pass "session-start creates memory-context.md"
else
  fail "session-start did not create memory-context.md"
fi

# 2) user-prompt should emit additionalContext JSON for sufficient prompt length.
USER_PROMPT_OUT="$(echo '{"prompt":"Please help fix deployment timeout in pipeline"}' | bash "$ROOT_DIR/user-prompt.sh")"
if echo "$USER_PROMPT_OUT" | grep -q "additionalContext"; then
  pass "user-prompt emits additionalContext"
else
  fail "user-prompt missing additionalContext"
fi

# 3) user-prompt should no-op for short prompts.
SHORT_OUT="$(echo '{"prompt":"short"}' | bash "$ROOT_DIR/user-prompt.sh")"
if [ -z "$SHORT_OUT" ]; then
  pass "user-prompt short prompt no output"
else
  fail "user-prompt short prompt should be silent"
fi

# 4) stop hook should always exit success even without transcript.
echo '{}' | bash "$ROOT_DIR/stop.sh" >/dev/null 2>&1 || fail "stop hook exits non-zero"
pass "stop hook exits zero"

# 5) stop hook with transcript should call /api/remember.
TRANSCRIPT_FILE="$TMP_DIR/transcript.jsonl"
cat > "$TRANSCRIPT_FILE" <<'EOF'
{"role": "user", "content": "How do I fix this bug?"}
{"role": "assistant", "content": "Try checking logs first."}
EOF

echo '{}' | CLAUDE_TRANSCRIPT_FILE="$TRANSCRIPT_FILE" bash "$ROOT_DIR/stop.sh" >/dev/null 2>&1 || fail "stop hook transcript path exits non-zero"
if grep -q "/api/remember" "$MOCK_CURL_LOG"; then
  pass "stop hook transcript path calls remember endpoint"
else
  fail "stop hook transcript path did not call remember endpoint"
fi
