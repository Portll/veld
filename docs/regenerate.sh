#!/usr/bin/env bash
# Regenerate all auto-generated docs pages. Run before committing changes that
# touch any of the source files the generators read.
#
# This script is invoked by the pre-commit hook and by CI. Manual invocation:
#
#     cd docs && bash regenerate.sh
#
# Generators are in `docs/generators/`. Each one reads veld source and writes
# a markdown file under `docs/src/reference/` or `docs/src/architecture/`.

set -euo pipefail

cd "$(dirname "$0")/generators"

echo "==> gen-http-api"
cargo run --quiet --bin gen-http-api

echo "==> gen-mcp-tools"
cargo run --quiet --bin gen-mcp-tools

echo "==> gen-config-ref"
cargo run --quiet --bin gen-config-ref

echo "==> gen-errors"
cargo run --quiet --bin gen-errors

echo "==> gen-metrics"
cargo run --quiet --bin gen-metrics

echo "==> gen-module-index"
cargo run --quiet --bin gen-module-index

echo "==> gen-claude-sections"
cargo run --quiet --bin gen-claude-sections

echo "==> lint-decisions"
cargo run --quiet --bin lint-decisions

echo
echo "All generators completed. Generated reference files:"
ls -la ../src/reference/
