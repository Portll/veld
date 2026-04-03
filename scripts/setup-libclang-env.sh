#!/usr/bin/env bash
# Configure macOS libclang loader paths for local Cargo builds that transitively
# invoke bindgen through librocksdb-sys.

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "source this file instead of executing it:"
    echo "  source ./scripts/setup-libclang-env.sh"
    exit 1
fi

if [[ "$(uname)" != "Darwin" ]]; then
    return 0
fi

CLANG_DIR="/Library/Developer/CommandLineTools/usr/lib"
if [[ ! -f "$CLANG_DIR/libclang.dylib" ]]; then
    return 0
fi

export LIBCLANG_PATH="${LIBCLANG_PATH:-$CLANG_DIR}"

case ":${DYLD_FALLBACK_LIBRARY_PATH:-}:" in
    *":$CLANG_DIR:"*) ;;
    *) export DYLD_FALLBACK_LIBRARY_PATH="$CLANG_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}" ;;
esac
