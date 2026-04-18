#!/usr/bin/env bash
# Rectification runner — launches 5 Claude Code agents in parallel
# Usage: bash scripts/run-rectification.sh
# Each agent works on a non-overlapping set of files.

set -euo pipefail
cd "$(dirname "$0")/.."

PLAN="$(cat RECTIFICATION.md)"

COMMON_PREAMBLE="You are fixing bugs in the veld codebase at $(pwd).

RULES:
- Production grade code only. No TODOs, no placeholders.
- Read each file BEFORE editing. Understand context.
- Do NOT touch files outside your assigned list.
- Run \`cargo check\` when done (ignore RocksDB linker errors — check for compile errors only).
- Do NOT add Co-Authored-By or Generated with Claude Code signatures.
- Commit your changes with a clean message when done.

RECTIFICATION PLAN:
$PLAN
"

echo "=== Veld Rectification: launching 5 agents ==="
echo ""

# Agent 1: TUI UTF-8 Safety
claude -p "$COMMON_PREAMBLE

YOU ARE AGENT 1: TUI UTF-8 Safety.

Your files: tui/src/widgets.rs, tui/src/types.rs, tests/benchmark_evaluation.rs, tests/cognitive_stress_test.rs

Execute ALL items in the Agent 1 section of the rectification plan.

1. Add a \`truncate_safe(s: &str, max_bytes: usize) -> &str\` helper function
2. Replace every \`&str[..n]\` byte-index slice on user-derived strings with \`truncate_safe()\`
3. Check for any other byte-index slices you find that aren't in the plan
4. Run \`cargo check --manifest-path tui/Cargo.toml\` to verify
5. Commit: \"fix: replace all unsafe UTF-8 byte-index slices in TUI with char-boundary-safe truncation\"
" --model sonnet --allowedTools Edit,Read,Write,Bash,Grep,Glob &
PID1=$!
echo "  Agent 1 (TUI UTF-8) launched: PID $PID1"

# Agent 2: Auth & Credential Hardening
claude -p "$COMMON_PREAMBLE

YOU ARE AGENT 2: Auth & Credential Hardening.

Your files: src/auth.rs, src/encryption.rs, src/config.rs, hooks/session-start.sh, hooks/user-prompt.sh, Cargo.toml (only to add zeroize dep)

Execute ALL items in the Agent 2 section of the rectification plan.

1. Remove full API key from HTTP 401 error response bodies in auth.rs
2. Truncate logged dev API key to first 12 chars
3. Add zeroize crate and implement Drop for FieldEncryptor
4. Remove hardcoded fallback API keys from shell hooks
5. Run \`cargo check\` to verify (ignore RocksDB linker errors)
6. Commit: \"fix: credential hardening — scrub keys from responses, zeroize on drop, remove hardcoded keys\"
" --model sonnet --allowedTools Edit,Read,Write,Bash,Grep,Glob &
PID2=$!
echo "  Agent 2 (Auth) launched: PID $PID2"

# Agent 3: Graph Concurrency & Silent Failures
claude -p "$COMMON_PREAMBLE

YOU ARE AGENT 3: Graph Concurrency & Silent Failures.

Your files: src/graph_memory.rs, src/handlers/consolidation.rs, src/handlers/recall.rs

Execute ALL items in the Agent 3 section of the rectification plan.

1. Fix the TOCTOU race in add_relationship by acquiring synapse_update_lock
2. Add per-user consolidation lock to prevent concurrent runs
3. Replace all \`let _ =\` on write operations with warn-level logging
4. Run \`cargo check\` to verify (ignore RocksDB linker errors)
5. Commit: \"fix: graph race conditions, consolidation guard, surface silent write failures\"
" --model sonnet --allowedTools Edit,Read,Write,Bash,Grep,Glob &
PID3=$!
echo "  Agent 3 (Graph) launched: PID $PID3"

# Agent 4: MCP Server & Hooks Hardening
claude -p "$COMMON_PREAMBLE

YOU ARE AGENT 4: MCP Server & Hooks Hardening.

Your files: mcp-server/index.ts, mcp-server/security-utils.ts, hooks/memory-hook.ts

Execute ALL items in the Agent 4 section of the rectification plan.

1. Guard the URL parse in memory-hook.ts with try/catch
2. Add secret pattern filter before memory ingestion
3. Cache health check with 5s TTL instead of per-call
4. Scrub internal URLs and raw errors from MCP client-facing messages
5. Disable auto_ingest when serving stale cache
6. Add console.error to the silent top-level catch
7. Run type checks if TypeScript tooling is available
8. Commit: \"fix: MCP/hooks hardening — secret filtering, health cache, error scrubbing, stale guard\"
" --model sonnet --allowedTools Edit,Read,Write,Bash,Grep,Glob &
PID4=$!
echo "  Agent 4 (MCP/Hooks) launched: PID $PID4"

# Agent 5: Error Handling, Audit & HTTP Response Scrubbing
claude -p "$COMMON_PREAMBLE

YOU ARE AGENT 5: Error Handling, Audit & HTTP Response Scrubbing.

Your files: src/errors.rs, src/handlers/users.rs, src/handlers/router.rs, src/server.rs, src/middleware.rs, src/handlers/state.rs

Execute ALL items in the Agent 5 section of the rectification plan.

1. Scrub internal error details from HTTP 5xx responses — log full detail server-side only
2. Sanitize memory IDs in error messages
3. Add admin-only gate for list_users
4. Add audit logging for list_users and delete_user
5. Fix the process::exit(1) on shutdown timeout
6. Replace method().to_string() with method().as_str()
7. Run \`cargo check\` to verify (ignore RocksDB linker errors)
8. Commit: \"fix: error scrubbing, audit logging, shutdown safety, user enumeration guard\"
" --model sonnet --allowedTools Edit,Read,Write,Bash,Grep,Glob &
PID5=$!
echo "  Agent 5 (Errors/Audit) launched: PID $PID5"

echo ""
echo "=== All 5 agents running ==="
echo "  PIDs: $PID1 $PID2 $PID3 $PID4 $PID5"
echo ""
echo "Waiting for all agents to complete..."
echo ""

FAILED=0
for pid in $PID1 $PID2 $PID3 $PID4 $PID5; do
    if ! wait $pid; then
        echo "  WARN: Agent PID $pid exited with error"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All 5 agents completed successfully ==="
else
    echo "=== $FAILED agent(s) had errors — check output above ==="
fi

echo ""
echo "Post-flight checks:"
echo "  cargo check"
echo "  cargo clippy"
echo "  grep -rn '&str\[\.\.n\]' tui/src/"
echo "  grep -rn 'sk-veld-dev-local-testing-key' hooks/"
echo "  git log --oneline -5"
