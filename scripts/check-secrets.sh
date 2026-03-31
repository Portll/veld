#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

mode="${1:---staged}"
tmp_output="$(mktemp)"
trap 'rm -f "$tmp_output"' EXIT

pattern='(OPENAI_API_KEY[[:space:]]*=[[:space:]]*sk-[A-Za-z0-9_-]{20,}|sk-(proj|svcacct)-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]+|-----BEGIN (EC|OPENSSH|PGP|PRIVATE KEY|RSA|DSA))'

scan_staged() {
    while IFS= read -r -d '' path; do
        if git show ":$path" | LC_ALL=C grep -nE "$pattern" >"$tmp_output"; then
            printf 'Potential secret in staged file: %s\n' "$path"
            cat "$tmp_output"
            return 1
        fi
    done < <(git diff --cached --name-only --diff-filter=ACM -z)
}

scan_all() {
    while IFS= read -r -d '' path; do
        if [[ -f "$path" ]] && LC_ALL=C grep -nE "$pattern" "$path" >"$tmp_output"; then
            printf 'Potential secret in file: %s\n' "$path"
            cat "$tmp_output"
            return 1
        fi
    done < <(git ls-files -z --cached --others --exclude-standard)
}

case "$mode" in
    --staged)
        scan_staged
        ;;
    --all)
        scan_all
        ;;
    *)
        printf 'Usage: %s [--staged|--all]\n' "$0" >&2
        exit 2
        ;;
esac

printf 'Secret scan passed (%s).\n' "$mode"