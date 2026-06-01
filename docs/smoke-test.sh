#!/usr/bin/env bash
# Post-deploy smoke test for the veld docs sidecar.
#
# Hits a small set of canonical URLs and asserts they return 200 and contain
# a sentinel string. Fails fast on the first broken check.
#
# Usage:
#     bash docs/smoke-test.sh https://portll.github.io/veld
#     bash docs/smoke-test.sh http://127.0.0.1:3000   # local mdbook serve

set -euo pipefail

BASE="${1:-https://portll.github.io/veld}"
BASE="${BASE%/}" # strip trailing slash

fail=0

# `check_url <url> <sentinel-substring>`
# Fetches the URL, requires 200, requires the sentinel to appear in the body.
check_url() {
    local url="$1"
    local sentinel="$2"
    local body
    if ! body="$(curl -fsSL --max-time 30 "$url")"; then
        echo "  FAIL: $url did not return 200" >&2
        fail=$((fail + 1))
        return
    fi
    if ! grep -q "$sentinel" <<<"$body"; then
        echo "  FAIL: $url returned 200 but body lacks expected '$sentinel'" >&2
        fail=$((fail + 1))
        return
    fi
    echo "  OK:   $url"
}

# `check_status <url>`
# Just asserts 200, no body check. Use for binary assets.
check_status() {
    local url="$1"
    if ! curl -fsSL --max-time 30 -o /dev/null -I "$url"; then
        echo "  FAIL: $url did not return 200" >&2
        fail=$((fail + 1))
        return
    fi
    echo "  OK:   $url"
}

echo "==> Smoke-testing $BASE"

# 1. Root index loads.
check_url "$BASE/" "Veld"

# 2. Search index is reachable (mdBook serves it as searchindex.json).
check_status "$BASE/searchindex.json"

# 3. A known reference page exists and lists a known route.
check_url "$BASE/reference/http-api.html" "/api/remember"

# 4. MCP-tools reference exists and mentions a known tool.
check_url "$BASE/reference/mcp-tools.html" "remember"

# 5. A decision record loads.
check_url "$BASE/decisions/0001-redb-migration.html" "Redb"

# 6. The module index is present (auto-generated, easy drift indicator).
check_url "$BASE/architecture/module-index.html" "Module"

# 7. Rustdoc was stitched into /api/.
check_status "$BASE/api/veld/index.html"

echo
if (( fail == 0 )); then
    echo "==> All checks passed."
    exit 0
else
    echo "==> $fail check(s) failed." >&2
    exit 1
fi
