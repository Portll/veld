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

Tier 1 needs at least one selector, for example:
  scripts/test-local.sh --lib write_gate --test handler_tests
  scripts/test-local.sh --no-default-features --features storage-redb --test handler_tests
Tier 2 runs fmt, clippy and the tests and bench builds of every feature leg CI runs,
skipping the tests listed in scripts/test-local.known-red.
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

known_red_skips() {
  [[ -f "$known_red_file" ]] || return 0
  local name
  while IFS= read -r name; do
    name="${name%%#*}"
    name="${name//[[:space:]]/}"
    [[ -n "$name" ]] && printf -- '--skip\n%s\n' "$name"
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

run_full_matrix() {
  take_full_suite_lock
  local log_dir="${CARGO_TARGET_DIR:-$repo_root/target}-logs"
  mkdir -p "$log_dir"
  local -a skips=()
  local skip
  while IFS= read -r skip; do skips+=("$skip"); done < <(known_red_skips)
  local failed=()

  cd "$repo_root"
  step "fmt"
  cargo fmt --all -- --check || failed+=("fmt")

  local features
  for features in "" "--no-default-features --features storage-redb" "--features zenoh"; do
    step "clippy ${features:-default}"
    # shellcheck disable=SC2086
    cargo_niced clippy --all-targets $features -- -W clippy::all -D clippy::correctness \
      >"$log_dir/clippy-${features// /_}.log" 2>&1 || failed+=("clippy ${features:-default}")
  done

  for features in "" "--no-default-features --features storage-redb" "--no-default-features --features rocksdb"; do
    local log="$log_dir/test-${features// /_}.log"
    step "test ${features:-default} (log: $log)"
    # shellcheck disable=SC2086
    if ! cargo_niced test --no-fail-fast $features -- ${skips[@]+"${skips[@]}"} >"$log" 2>&1; then
      failed+=("test ${features:-default}")
    fi
    grep -E '^test result: FAILED|^test .* FAILED$' "$log" | head -20 || true
    step "bench --no-run ${features:-default}"
    # shellcheck disable=SC2086
    cargo_niced bench --no-run $features >"$log_dir/bench-${features// /_}.log" 2>&1 \
      || failed+=("bench --no-run ${features:-default}")
  done

  if ((${#failed[@]})); then
    printf '\nFAILED: %s\n' "${failed[@]}"
    exit 1
  fi
  printf '\ngreen (skipped %d known-red tests)\n' "$((${#skips[@]} / 2))"
  finished=1
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
if [[ "$1" == "--full" ]]; then
  [[ $# -eq 1 ]] || { echo "test-local: --full takes no other arguments" >&2; exit 2; }
  run_full_matrix
else
  refuse_binary_builds "$@"
  cd "$repo_root"
  cargo_niced test "$@"
  finished=1
fi
