#!/usr/bin/env bash
# Tests for scripts/bump-version.sh.
# Runs in a fresh temp git repo with synthetic fixtures so we never touch the
# real Cargo.toml / pyproject.toml / mcp-server/package.json.
#
# Usage: bash scripts/tests/bump-version.test.sh
#
# Output is TAP-ish ("ok - …" / "not ok - …"). Exits non-zero on any failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/bump-version.sh"
TMP_BASE="$(mktemp -d)"
FAILED=0

cleanup() {
    rm -rf "$TMP_BASE"
}
trap cleanup EXIT

pass() {
    echo "ok - $1"
}

fail() {
    echo "not ok - $1"
    FAILED=$((FAILED + 1))
}

# Create a fresh sandbox repo for a single test case. Emits the path on stdout.
make_sandbox() {
    local dir
    dir="$(mktemp -d -p "$TMP_BASE")"
    (
        cd "$dir"
        git init --quiet
        git config user.email "test@example.com"
        git config user.name "test"
        mkdir -p mcp-server
    )
    echo "$dir"
}

# Write a Cargo.toml containing both a [package] version and dependency versions,
# so we can prove the script only touches the package section.
write_cargo() {
    local dir="$1" version="$2"
    cat > "$dir/Cargo.toml" <<EOF
[package]
name = "test"
version = "$version"
edition = "2021"

[dependencies]
serde = { version = "1.0.0" }
tokio = "1.40.0"
EOF
}

write_pyproject() {
    local dir="$1" version="$2"
    cat > "$dir/pyproject.toml" <<EOF
[project]
name = "test"
version = "$version"
EOF
}

write_package_json() {
    local dir="$1" version="$2"
    cat > "$dir/mcp-server/package.json" <<EOF
{
  "name": "test",
  "version": "$version",
  "dependencies": {
    "left-pad": "1.0.0"
  }
}
EOF
}

# Read the [package] version from a sandbox Cargo.toml.
read_cargo_version() {
    local dir="$1"
    awk '
        /^\[package\]/ { in_pkg=1; next }
        /^\[/ { in_pkg=0 }
        in_pkg && /^version = "/ {
            gsub(/version = "|"/, "")
            print
            exit
        }
    ' "$dir/Cargo.toml"
}

read_pyproject_version() {
    local dir="$1"
    awk '/^version = "/ { gsub(/version = "|"/, ""); print; exit }' "$dir/pyproject.toml"
}

read_npm_version() {
    local dir="$1"
    # Pure sed; avoids passing MSYS-style /tmp paths to native Python on Windows.
    sed -n 's/.*"version":[[:space:]]*"\([^"]*\)".*/\1/p' "$dir/mcp-server/package.json" | head -1
}

# JSON validity check that works under MSYS — cd into the dir so Python sees
# the cwd as a native Windows path and the relative path resolves correctly.
is_valid_json() {
    local dir="$1" rel="$2"
    ( cd "$dir" && python3 -c "import json; json.load(open('$rel'))" ) 2>/dev/null
}

# Read the *first* dependency version line in Cargo.toml — used to confirm the
# script's regex doesn't accidentally rewrite dependency pins.
read_dep_version() {
    local dir="$1"
    awk '
        /^\[dependencies\]/ { in_deps=1; next }
        /^\[/ { in_deps=0 }
        in_deps && /version = "/ {
            match($0, /"[^"]+"/)
            print substr($0, RSTART+1, RLENGTH-2)
            exit
        }
    ' "$dir/Cargo.toml"
}

# Run bump-version.sh in the sandbox, propagating exit code through $?.
run_bump() {
    local dir="$1"
    shift
    ( cd "$dir" && "$SCRIPT" "$@" >/dev/null 2>&1 )
}

# ─── tests ─────────────────────────────────────────────────────────────────

test_patch_bump() {
    local d
    d="$(make_sandbox)"
    write_cargo "$d" "0.7.7"
    write_pyproject "$d" "0.7.7"
    write_package_json "$d" "0.7.7"

    if run_bump "$d" patch; then
        if [[ "$(read_cargo_version "$d")" == "0.7.8" \
            && "$(read_pyproject_version "$d")" == "0.7.8" \
            && "$(read_npm_version "$d")" == "0.7.8" ]]; then
            pass "patch bump 0.7.7 -> 0.7.8 across all three files"
        else
            fail "patch bump did not produce 0.7.8 in every file (cargo=$(read_cargo_version "$d") py=$(read_pyproject_version "$d") npm=$(read_npm_version "$d"))"
        fi
    else
        fail "patch bump script exited non-zero"
    fi
}

test_minor_bump_resets_patch() {
    local d
    d="$(make_sandbox)"
    write_cargo "$d" "0.7.7"
    write_pyproject "$d" "0.7.7"
    write_package_json "$d" "0.7.7"

    if run_bump "$d" minor && [[ "$(read_cargo_version "$d")" == "0.8.0" ]]; then
        pass "minor bump 0.7.7 -> 0.8.0 resets patch"
    else
        fail "minor bump did not reset patch (got $(read_cargo_version "$d"))"
    fi
}

test_major_bump_resets_minor_and_patch() {
    local d
    d="$(make_sandbox)"
    write_cargo "$d" "1.2.3"
    write_pyproject "$d" "1.2.3"
    write_package_json "$d" "1.2.3"

    if run_bump "$d" major && [[ "$(read_cargo_version "$d")" == "2.0.0" ]]; then
        pass "major bump 1.2.3 -> 2.0.0 resets minor and patch"
    else
        fail "major bump did not reset minor/patch (got $(read_cargo_version "$d"))"
    fi
}

test_does_not_touch_dependency_versions() {
    local d
    d="$(make_sandbox)"
    write_cargo "$d" "0.7.7"
    write_pyproject "$d" "0.7.7"
    write_package_json "$d" "0.7.7"

    if run_bump "$d" patch && [[ "$(read_dep_version "$d")" == "1.0.0" ]]; then
        pass "package bump leaves dependency versions untouched"
    else
        fail "package bump altered a dependency version (got $(read_dep_version "$d"))"
    fi
}

test_prerelease_suffix_normalizes_to_release() {
    local d
    d="$(make_sandbox)"
    write_cargo "$d" "0.7.6.dev0"
    write_pyproject "$d" "0.7.6.dev0"
    write_package_json "$d" "0.7.6"

    if run_bump "$d" patch && [[ "$(read_cargo_version "$d")" == "0.7.7" ]]; then
        pass "prerelease suffix dropped on bump (0.7.6.dev0 -> 0.7.7)"
    else
        fail "prerelease suffix not stripped (got $(read_cargo_version "$d"))"
    fi
}

test_package_json_remains_valid() {
    local d
    d="$(make_sandbox)"
    write_cargo "$d" "0.7.7"
    write_pyproject "$d" "0.7.7"
    write_package_json "$d" "0.7.7"

    if run_bump "$d" patch && is_valid_json "$d" "mcp-server/package.json"; then
        pass "package.json is still valid JSON after bump"
    else
        fail "package.json became invalid JSON after bump"
    fi
}

test_stage_flag_runs_git_add() {
    local d
    d="$(make_sandbox)"
    write_cargo "$d" "0.7.7"
    write_pyproject "$d" "0.7.7"
    write_package_json "$d" "0.7.7"
    ( cd "$d" && git add . && git commit -m "init" --quiet )

    # Modify a file so the bump itself produces a diff, then bump --stage.
    sed -i.bak 's/0.7.7/0.7.7/' "$d/Cargo.toml" && rm "$d/Cargo.toml.bak"

    if run_bump "$d" patch --stage; then
        staged="$( cd "$d" && git diff --cached --name-only | sort | tr '\n' ' ' )"
        expected="Cargo.toml mcp-server/package.json pyproject.toml "
        if [[ "$staged" == "$expected" ]]; then
            pass "--stage adds the three version files to the index"
        else
            fail "--stage staged unexpected paths (got '$staged')"
        fi
    else
        fail "bump with --stage exited non-zero"
    fi
}

test_invalid_mode_rejected() {
    local d
    d="$(make_sandbox)"
    write_cargo "$d" "0.7.7"
    write_pyproject "$d" "0.7.7"
    write_package_json "$d" "0.7.7"

    if run_bump "$d" frobnicate; then
        fail "bump with bad mode 'frobnicate' returned success"
    else
        pass "bump rejects invalid mode"
    fi
}

# ─── runner ─────────────────────────────────────────────────────────────────

test_patch_bump
test_minor_bump_resets_patch
test_major_bump_resets_minor_and_patch
test_does_not_touch_dependency_versions
test_prerelease_suffix_normalizes_to_release
test_package_json_remains_valid
test_stage_flag_runs_git_add
test_invalid_mode_rejected

echo
if [[ "$FAILED" -gt 0 ]]; then
    echo "FAIL: $FAILED test(s) failed"
    exit 1
fi
echo "PASS: all tests"
