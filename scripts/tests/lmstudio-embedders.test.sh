#!/usr/bin/env bash
# Tests for scripts/lmstudio-embedders.sh against a fake lms CLI in a temp dir; the real LM Studio
# and its processes are never touched, since the script only reaps queries of its own $LMS path.
#
# Usage: bash scripts/tests/lmstudio-embedders.test.sh
#
# Output is TAP-ish ("ok - …" / "not ok - …"). Exits non-zero on any failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/lmstudio-embedders.sh"
TMP_BASE="$(mktemp -d)"
FAILED=0
SPAWNED=()

cleanup() {
    for pid in "${SPAWNED[@]}"; do kill "$pid" 2>/dev/null || true; done
    rm -rf "$TMP_BASE"
}
trap cleanup EXIT

pass() { echo "ok - $1"; }
fail() { echo "not ok - $1"; FAILED=$((FAILED + 1)); }

export LMS="$TMP_BASE/lms"
CALLS="$TMP_BASE/calls"
cat > "$LMS" <<'FAKE'
#!/bin/sh
echo "$*" >> "$FAKE_CALLS"
case "$1" in
    server) [ "$2" = status ] && echo "The server is running on port 1234." ;;
    ps) [ -n "${FAKE_PS_SLEEP:-}" ] && exec sleep "$FAKE_PS_SLEEP"; printf '%b' "${FAKE_PS_OUT:-}" ;;
esac
FAKE
chmod +x "$LMS"
export FAKE_CALLS="$CALLS"
export EMBEDDER_MODELS="model-a model-b"

# A process whose ps command line reads "$LMS <args>", as the real native binary's does; its pid
# is left in $SPAWNED_PID.
spawn_as_lms() {
    perl -e 'my $argv0 = shift; exec {"/bin/sleep"} $argv0, "60"' "$LMS $1" > /dev/null 2>&1 &
    SPAWNED_PID=$!
    disown "$SPAWNED_PID"
    SPAWNED+=("$SPAWNED_PID")
}

# 1. A hung `lms ps` is cut at LMS_QUERY_SECS, and nothing is loaded on a reading the run never got.
: > "$CALLS"
start=$(date +%s)
status=0
FAKE_PS_SLEEP=30 LMS_QUERY_SECS=1 sh "$SCRIPT" > "$TMP_BASE/out1" 2>&1 || status=$?
elapsed=$(( $(date +%s) - start ))
if [ "$elapsed" -lt 10 ] && [ "$status" -ne 0 ]; then
    pass "a hung lms ps is bounded (${elapsed}s, exit $status)"
else
    fail "a hung lms ps is bounded (took ${elapsed}s, exit $status)"
fi
if grep -q '^load' "$CALLS"; then fail "no load after a failed lms ps"; else pass "no load after a failed lms ps"; fi

# 2. Only the model lms ps does not list is loaded.
: > "$CALLS"
FAKE_PS_OUT='model-a  1.2 GB  idle\n' sh "$SCRIPT" > "$TMP_BASE/out2" 2>&1
if grep -qx 'load model-b --identifier model-b -y' "$CALLS" && ! grep -q '^load model-a' "$CALLS"; then
    pass "loads only the missing model"
else
    fail "loads only the missing model (calls: $(tr '\n' ';' < "$CALLS"))"
fi

# 3. A stray query older than LMS_STRAY_SECS is reaped; a long `lms load` is left running.
spawn_as_lms ps; stray=$SPAWNED_PID
spawn_as_lms "load model-a"; loading=$SPAWNED_PID
sleep 2
LMS_STRAY_SECS=1 FAKE_PS_OUT='model-a  x\nmodel-b  x\n' sh "$SCRIPT" > "$TMP_BASE/out3" 2>&1
sleep 0.2
if kill -0 "$stray" 2>/dev/null; then fail "reaps a stray lms ps"; else pass "reaps a stray lms ps"; fi
if kill -0 "$loading" 2>/dev/null; then pass "leaves lms load running"; else fail "leaves lms load running"; fi
if grep -q "reaping stray lms queries: $stray" "$TMP_BASE/out3"; then
    pass "logs the reaped pid"
else
    fail "logs the reaped pid ($(cat "$TMP_BASE/out3"))"
fi

# 4. A young query is not reaped.
spawn_as_lms ps; young=$SPAWNED_PID
LMS_STRAY_SECS=600 FAKE_PS_OUT='model-a  x\nmodel-b  x\n' sh "$SCRIPT" > "$TMP_BASE/out4" 2>&1
if kill -0 "$young" 2>/dev/null; then pass "leaves a young lms ps"; else fail "leaves a young lms ps"; fi

[ "$FAILED" -eq 0 ] || { echo "$FAILED failed"; exit 1; }
echo "all passed"
