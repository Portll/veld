#!/bin/sh
# What green means for veld before anything that starts CI: the jobs of .github/workflows/ci.yml,
# run here. spine's bin/ci-verdict.mjs runs this on a clean checkout of one commit and records the
# verdict its push gate reads; it can also be run by hand from any worktree.
#
# Not mirrored: the upstream watcher heartbeat (an environment check, not the tree's) and the
# release build, which CI runs only on tags and dispatch.
set -eu
cd "$(dirname "$0")/.."

step() { printf '== ci-local: %s\n' "$1"; }

step "version consistency"
core() { python3 -c 'import re, sys; m = re.match(r"^(\d+\.\d+\.\d+)", sys.argv[1].strip()); print(m.group(1) if m else sys.argv[1])' "$1"; }
cargo_v=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
npm_v=$(grep '"version"' mcp-server/package.json | head -1 | sed 's/.*"version": "\(.*\)".*/\1/')
pypi_v=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
if [ "$(core "$cargo_v")" != "$(core "$npm_v")" ] || [ "$(core "$cargo_v")" != "$(core "$pypi_v")" ]; then
  echo "version cores differ: Cargo $cargo_v, npm $npm_v, PyPI $pypi_v" >&2
  exit 1
fi

step "upstream ledger"
python3 scripts/upstream/ledger.py check

if [ ! -x scripts/test-local.sh ]; then
  echo "scripts/test-local.sh is missing or not executable; it runs the format, clippy, test and bench legs" >&2
  exit 1
fi
step "format, clippy, tests and benches on every engine set"
# Under ci-verdict every run uses one scratch checkout, so one target dir keeps its builds
# incremental; test-local.sh's per-checkout default would otherwise be shared with a working tree
# of the same name. Its --full lock serialises the runs that write it.
if [ -n "${CI_VERDICT_SHA:-}" ]; then
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cargo-target-drive/veld-verdict}"
fi
exec scripts/test-local.sh --full
