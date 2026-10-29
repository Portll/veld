#!/usr/bin/env bash
# Run tests locally under the tiered policy in CLAUDE.md (Build & Test Discipline).
#
#   scripts/test-local.sh <cargo test selectors and flags>   tier 1: the tests a change touches
#   scripts/test-local.sh --full                              tier 2: the CI matrix, as a push gate
#
# Every run is niced with a capped test thread count, because the live veld servers on
# this machine serve every session and full matrices running side by side pushed the
# load average to 37 on 2026-10-21. Tests get a throwaway HOME and no inherited VELD_*
# variables, so a test that forgets to set its own storage path lands in the sandbox,
# never in a live tenant; only the model cache is shared, through a symlink. Tier 2
# holds a machine-wide lock so two sessions cannot run full matrices at once.
set -euo pipefail

# macOS ships bash 3.2, where an EXIT trap sees status 0 after an abort on an unbound
# variable: the first --full run here reported success after dying in its first test
# leg. Only a run that reaches its end may exit 0; any other exit is a failure.
finished=0
sandbox=""
lock_owned=0
cleanup() {
  local status=$?
  if [[ -n "$sandbox" && -d "$sandbox" ]]; then rm -rf -- "$sandbox"; fi
  if [[ "$lock_owned" -eq 1 ]]; then rm -rf -- "$lock_dir"; fi
  if [[ "$finished" -ne 1 && "$status" -eq 0 ]]; then status=1; fi
  exit "$status"
}
trap cleanup EXIT

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
real_home="$HOME"
known_red_file="$repo_root/scripts/test-local.known-red"
lock_dir="${TMPDIR:-/tmp}/veld-test-local-full.lock"

usage() {
  cat <<'EOF'
usage: scripts/test-local.sh <cargo test selectors and flags>
       scripts/test-local.sh --full
       scripts/test-local.sh --rebase

Tier 1 needs at least one selector. A test-name filter applies to every target it is given
with, so run the touched module's tests and the whole handler suite as separate calls:
  scripts/test-local.sh --lib write_gate
  scripts/test-local.sh --test handler_tests
  scripts/test-local.sh --no-default-features --features storage-redb --test handler_tests
Tier 2 runs what CI runs: fmt, clippy, then per feature leg nextest (ci profile), the
doctests and a bench build, skipping the tests listed in scripts/test-local.known-red.
It routes by the paths changed since origin/main: the rocksdb-only leg's tests only build
unless an engine path or a cfg/feature line changed, and the engine-parity and
recall-harness tests run only when recall, storage or embedding code changed.
TEST_LOCAL_ROUTE=all runs every leg in full.
--rebase is what a tier 2 verdict carried across a rebase still runs (scripts/ci-rebase.sh):
fmt, clippy, every test and bench target built per feature leg, and the handler suite.
EOF
}

refuse_binary_builds() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --release | --profile | --profile=*)
        echo "test-local: $arg builds an optimised binary; tests run on the dev profile" >&2
        exit 2
        ;;
    esac
  done
}

isolate_environment() {
  export CARGO_HOME="${CARGO_HOME:-$real_home/.cargo}"
  export RUSTUP_HOME="${RUSTUP_HOME:-$real_home/.rustup}"
  if [[ -z "${CARGO_TARGET_DIR:-}" ]]; then
    local target_root="${VELD_TEST_TARGET_ROOT:-$real_home/.cargo-target-drive}"
    if [[ -d "$target_root" ]]; then
      CARGO_TARGET_DIR="$target_root/$(basename "$repo_root")-test"
      export CARGO_TARGET_DIR
    fi
  fi

  local var
  for var in $(compgen -e); do
    case "$var" in
      VELD_* | SHODH_*) unset "$var" ;;
    esac
  done
  export SHODH_ALLOW_FALLBACK_NER=1
  export RUST_TEST_THREADS="${RUST_TEST_THREADS:-4}"

  sandbox="$(mktemp -d "${TMPDIR:-/tmp}/veld-test-home.XXXXXX")"
  local real_cache sandbox_cache
  case "$(uname -s)" in
    Darwin)
      real_cache="$real_home/Library/Caches/veld"
      sandbox_cache="$sandbox/Library/Caches/veld"
      ;;
    *)
      real_cache="${XDG_CACHE_HOME:-$real_home/.cache}/veld"
      sandbox_cache="$sandbox/.cache/veld"
      export XDG_CACHE_HOME="$sandbox/.cache"
      export XDG_DATA_HOME="$sandbox/.local/share"
      export XDG_CONFIG_HOME="$sandbox/.config"
      ;;
  esac
  mkdir -p "$(dirname "$sandbox_cache")"
  if [[ -d "$real_cache" ]]; then
    ln -s "$real_cache" "$sandbox_cache"
  fi
  export HOME="$sandbox"
}

cargo_niced() {
  nice -n 10 "$repo_root/scripts/cargo-dev.sh" "$@"
}

known_red_names() {
  [[ -f "$known_red_file" ]] || return 0
  local name
  while IFS= read -r name; do
    name="${name%%#*}"
    name="${name//[[:space:]]/}"
    [[ -n "$name" ]] && printf '%s\n' "$name"
  done <"$known_red_file"
}

take_full_suite_lock() {
  if mkdir "$lock_dir" 2>/dev/null; then
    lock_owned=1
    echo "$$ $(basename "$repo_root")" >"$lock_dir/owner"
    return
  fi
  local holder
  holder="$(cat "$lock_dir/owner" 2>/dev/null || true)"
  if [[ -n "$holder" ]] && kill -0 "${holder%% *}" 2>/dev/null; then
    echo "test-local: a full matrix is already running ($holder); wait for it" >&2
    exit 3
  fi
  echo "test-local: reclaiming a stale full-suite lock ($holder)" >&2
  rm -rf "$lock_dir"
  take_full_suite_lock
}

step() { printf '\n== %s\n' "$*"; }

# The nextest filter expression naming every known-red test, or nothing.
known_red_expr() {
  local name expr=""
  while IFS= read -r name; do
    expr="${expr:+$expr | }test(~$name)"
  done < <(known_red_names)
  printf '%s' "$expr"
}

# fmt, then clippy on CI's three legs; appends to the caller's `failed`.
check_format_and_lints() {
  step "fmt"
  cargo fmt --all -- --check || failed+=("fmt")
  local features
  for features in "" "--no-default-features --features storage-redb" "--features zenoh"; do
    step "clippy ${features:-default}"
    # shellcheck disable=SC2086
    cargo_niced clippy --all-targets $features -- -W clippy::all -D clippy::correctness \
      >"$log_dir/clippy-${features// /_}.log" 2>&1 || failed+=("clippy ${features:-default}")
  done
}

finish() {
  if ((${#failed[@]})); then
    printf '\nFAILED: %s\n' "${failed[@]}"
    exit 1
  fi
  printf '\ngreen (filter: %s%s)\n' "${expr:-none}" "${routed:+; routed: $routed}"
  finished=1
}

# Tier 2 routes two costs by the paths a commit changes against the main it branched from.
# GitHub CI still runs every leg in full; these only decide what the local gate repeats.
# The default build already runs every test on RocksDB, so the rocksdb-only leg adds only
# feature-gated code: its tests run when an engine path or a cfg/feature line changed.
engine_paths='^(src/storage/|src/backup\.rs$|src/memory/storage\.rs$|src/config\.rs$|build\.rs$|Cargo\.(toml|lock)$|\.config/nextest\.toml$|\.github/workflows/ci\.yml$|scripts/(test-local|ci-local)\.sh$)'
# The engine-parity and recall-harness tests are 43% of the default leg's test time
# (1,621 of 3,752 s, 2026-10-29); they run when recall, storage or embedding code changed.
slow_lane_paths='^(src/memory/|src/storage/|src/embeddings/|src/recall_harness/|src/vector_db/|src/intent_log/|src/graph_memory\.rs$|src/constants\.rs$|src/handlers/recall\.rs$|tests/recall_engine_parity\.rs$|tests/recall/|Cargo\.(toml|lock)$|\.config/nextest\.toml$|scripts/(test-local|ci-local)\.sh$)'
slow_lane_expr='binary_id(veld::recall_engine_parity) | (binary_id(veld) & test(/^recall_harness::/))'

# Sets run_rocksdb_tests, run_slow_lane and routed. Every leg runs when the change cannot be
# told (no origin/main, the commit is main itself) or TEST_LOCAL_ROUTE=all.
route_legs() {
  run_rocksdb_tests=1
  run_slow_lane=1
  routed=""
  [[ "${TEST_LOCAL_ROUTE:-}" == "all" ]] && return 0
  local base head changed
  base="$(git -C "$repo_root" merge-base HEAD refs/remotes/origin/main 2>/dev/null)" || return 0
  head="$(git -C "$repo_root" rev-parse HEAD)"
  [[ "$base" != "$head" ]] || return 0
  changed="$(git -C "$repo_root" diff --name-only "$base" "$head")"
  [[ -n "$changed" ]] || return 0
  if ! grep -qE "$engine_paths" <<<"$changed" \
    && ! git -C "$repo_root" diff -U0 "$base" "$head" -- '*.rs' | grep -qE '^[+-][^+-].*(cfg\(|feature *=)'; then
    run_rocksdb_tests=0
    routed="rocksdb-only tests built, not run"
  fi
  if ! grep -qE "$slow_lane_paths" <<<"$changed"; then
    run_slow_lane=0
    routed="${routed:+$routed, }engine-parity and recall-harness tests skipped"
  fi
}

run_rebase_check() {
  take_full_suite_lock
  local log_dir="${CARGO_TARGET_DIR:-$repo_root/target}-logs"
  mkdir -p "$log_dir"
  local expr
  expr="$(known_red_expr)"
  local -a filter=() failed=()
  if [[ -n "$expr" ]]; then filter=(-E "not ($expr)"); fi

  cd "$repo_root"
  check_format_and_lints
  # nextest builds through `cargo test --no-run`, so these reuse the carried full run's artefacts
  # and rebuild only what the commits on main changed.
  local features
  for features in "" "--no-default-features --features storage-redb" "--no-default-features --features rocksdb"; do
    local build_log="$log_dir/rebase-build-${features// /_}.log"
    step "build every test and bench target ${features:-default} (log: $build_log)"
    # shellcheck disable=SC2086
    cargo_niced test --no-run $features >"$build_log" 2>&1 || failed+=("test build ${features:-default}")
    # shellcheck disable=SC2086
    cargo_niced test --benches --no-run $features >>"$build_log" 2>&1 || failed+=("bench build ${features:-default}")
  done
  local log="$log_dir/rebase-handler-tests.log"
  step "nextest handler_tests default (log: $log)"
  if ! cargo_niced nextest run --profile ci --test-threads "$RUST_TEST_THREADS" --test handler_tests \
    ${filter[@]+"${filter[@]}"} >"$log" 2>&1; then
    failed+=("handler_tests")
  fi
  grep -E '^\s+(FAIL|TIMEOUT|SIGSEGV|SIGABRT) |^\s+Summary ' "$log" | head -20 || true
  finish
}

run_full_matrix() {
  take_full_suite_lock
  local log_dir="${CARGO_TARGET_DIR:-$repo_root/target}-logs"
  mkdir -p "$log_dir"
  # The test legs run what CI's do since bff68abf: nextest with the ci profile (one process per
  # test, per-test timeouts from .config/nextest.toml), then the doctests nextest does not run,
  # then a compile of the benches. A plain `cargo test` here missed a failure that only
  # per-process isolation shows (ner_tests::test_many_entities, 2026-10-23).
  local run_rocksdb_tests run_slow_lane routed
  route_legs
  local expr
  expr="$(known_red_expr)"
  if [[ "$run_slow_lane" -eq 0 ]]; then expr="${expr:+$expr | }$slow_lane_expr"; fi
  local -a filter=() failed=()
  if [[ -n "$expr" ]]; then filter=(-E "not ($expr)"); fi

  cd "$repo_root"
  check_format_and_lints

  local features
  for features in "" "--no-default-features --features storage-redb" "--no-default-features --features rocksdb"; do
    local log="$log_dir/test-${features// /_}.log"
    if [[ "$features" == *"--features rocksdb"* && "$run_rocksdb_tests" -eq 0 ]]; then
      step "build ${features} test targets only: no engine path or cfg/feature line changed (log: $log)"
      # shellcheck disable=SC2086
      cargo_niced test --no-run $features >"$log" 2>&1 || failed+=("test build ${features}")
    else
      step "nextest ${features:-default} (log: $log)"
      # shellcheck disable=SC2086
      if ! cargo_niced nextest run --profile ci --test-threads "$RUST_TEST_THREADS" $features \
        ${filter[@]+"${filter[@]}"} >"$log" 2>&1; then
        failed+=("nextest ${features:-default}")
      fi
      grep -E '^\s+(FAIL|TIMEOUT|SIGSEGV|SIGABRT) |^\s+Summary ' "$log" | head -20 || true
    fi
    step "doctests ${features:-default}"
    # shellcheck disable=SC2086
    cargo_niced test --doc $features >"$log_dir/doctest-${features// /_}.log" 2>&1 \
      || failed+=("doctests ${features:-default}")
    step "bench build ${features:-default}"
    # shellcheck disable=SC2086
    cargo_niced test --benches --no-run $features >"$log_dir/bench-${features// /_}.log" 2>&1 \
      || failed+=("bench build ${features:-default}")
  done
  finish
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  finished=1
  exit 0
fi

isolate_environment
if [[ "$1" == "--full" || "$1" == "--rebase" ]]; then
  [[ $# -eq 1 ]] || { echo "test-local: $1 takes no other arguments" >&2; exit 2; }
  if [[ "$1" == "--full" ]]; then run_full_matrix; else run_rebase_check; fi
else
  refuse_binary_builds "$@"
  cd "$repo_root"
  cargo_niced test "$@"
  finished=1
fi
