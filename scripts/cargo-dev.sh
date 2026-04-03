#!/usr/bin/env bash
# Cargo wrapper that applies the local macOS libclang workaround when needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./setup-libclang-env.sh
source "$SCRIPT_DIR/setup-libclang-env.sh"

exec cargo "$@"
