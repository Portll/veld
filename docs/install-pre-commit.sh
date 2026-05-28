#!/usr/bin/env bash
# Install a pre-commit hook that re-runs docs generators before each commit.
# Run once per clone:
#
#     bash docs/install-pre-commit.sh
#
# What it installs: a `.git/hooks/pre-commit` script that:
#   1. Runs the cheap docs generators (gen-http-api, gen-mcp-tools,
#      gen-config-ref, gen-errors, gen-metrics, gen-module-index,
#      gen-claude-sections, lint-decisions).
#   2. Stages any modified docs/src/ files so the generated output is
#      committed alongside the source change that prompted it.
#
# Generators that need a build (gen-cli-ref, deferred) are NOT run in
# pre-commit — they fire in CI only.
#
# If a pre-commit hook already exists, this script declines to overwrite
# and prints the diff so you can merge by hand.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"

if [[ -e "$HOOK_PATH" ]]; then
    echo "A pre-commit hook already exists at $HOOK_PATH."
    echo "Refusing to overwrite. To install the veld docs hook, merge the following snippet:"
    echo
    cat <<'EOF'
# --- veld docs hook (begin)
if [[ -d docs/generators ]]; then
    (cd docs && bash regenerate.sh) || exit 1
    git add docs/src/
fi
# --- veld docs hook (end)
EOF
    exit 1
fi

cat > "$HOOK_PATH" <<'EOF'
#!/usr/bin/env bash
# Veld docs pre-commit hook — auto-installed by docs/install-pre-commit.sh.
# Re-runs cheap docs generators and stages any updated files.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
if [[ -d "$REPO_ROOT/docs/generators" ]]; then
    (cd "$REPO_ROOT/docs" && bash regenerate.sh)
    git add "$REPO_ROOT/docs/src/"
fi
EOF

chmod +x "$HOOK_PATH"
echo "Installed pre-commit hook at $HOOK_PATH."
echo
echo "To test: stage some changes to src/ and run 'git commit'."
echo "To uninstall: rm $HOOK_PATH"
