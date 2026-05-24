#!/usr/bin/env bash
# bump-version.sh — bump the version in Cargo.toml, mcp-server/package.json, and
# pyproject.toml in lockstep. Used by the pre-commit hook and on demand.
#
# Usage:
#   bump-version.sh {major|minor|patch} [--stage]
#
#   --stage  also `git add` the three files so the bump rides on the current commit
#
# Reads the current version from Cargo.toml's [package] section.

set -euo pipefail

mode="${1:-patch}"
stage="${2:-}"

case "$mode" in
    major|minor|patch) ;;
    *) echo "usage: $0 {major|minor|patch} [--stage]" >&2; exit 2 ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

python3 - "$mode" "$stage" <<'PY'
import json
import pathlib
import re
import subprocess
import sys

mode, stage = sys.argv[1], sys.argv[2]

cargo_path = pathlib.Path("Cargo.toml")
pyproject_path = pathlib.Path("pyproject.toml")
npm_path = pathlib.Path("mcp-server/package.json")

cargo_text = cargo_path.read_text(encoding="utf-8")

# Find version inside [package] section only — must not pick up dependency versions.
pkg_match = re.search(
    r'^\[package\][\s\S]*?^version = "([0-9]+\.[0-9]+\.[0-9]+)((?:[.-][^"]*)?)"',
    cargo_text,
    re.M,
)
if not pkg_match:
    print("could not find [package] version in Cargo.toml", file=sys.stderr)
    sys.exit(1)
core = pkg_match.group(1)
suffix = pkg_match.group(2) or ""
old = core + suffix

major, minor, patch = (int(p) for p in core.split("."))
if mode == "major":
    major += 1
    minor = 0
    patch = 0
elif mode == "minor":
    minor += 1
    patch = 0
else:
    patch += 1
new_core = f"{major}.{minor}.{patch}"
new = new_core  # bumps strip any prerelease suffix; the new version is a release version

if old == new:
    print(f"bump-version: nothing to do ({old})")
    sys.exit(0)

print(f"bump-version: {old} -> {new}")

# Cargo.toml — only the [package] section's version line
new_cargo, n = re.subn(
    r'(^\[package\][\s\S]*?^version = ")[^"]*(")',
    lambda m: m.group(1) + new + m.group(2),
    cargo_text,
    count=1,
    flags=re.M,
)
if n != 1:
    print("Cargo.toml replacement failed", file=sys.stderr)
    sys.exit(1)
cargo_path.write_text(new_cargo, encoding="utf-8")

# pyproject.toml — single top-level version line
py_text = pyproject_path.read_text(encoding="utf-8")
new_py, n = re.subn(
    r'^version = "[^"]+"',
    f'version = "{new}"',
    py_text,
    count=1,
    flags=re.M,
)
if n != 1:
    print("pyproject.toml replacement failed", file=sys.stderr)
    sys.exit(1)
pyproject_path.write_text(new_py, encoding="utf-8")

# mcp-server/package.json — preserve formatting via json round-trip is risky;
# do a scoped regex on the first top-level "version": ".." line.
npm_text = npm_path.read_text(encoding="utf-8")
new_npm, n = re.subn(
    r'^(\s*)"version":\s*"[^"]+"',
    rf'\1"version": "{new}"',
    npm_text,
    count=1,
    flags=re.M,
)
if n != 1:
    print("mcp-server/package.json replacement failed", file=sys.stderr)
    sys.exit(1)
npm_path.write_text(new_npm, encoding="utf-8")

# Confirm package.json is still valid JSON.
try:
    json.loads(new_npm)
except json.JSONDecodeError as exc:
    print(f"package.json became invalid after bump: {exc}", file=sys.stderr)
    sys.exit(1)

if stage == "--stage":
    subprocess.run(
        ["git", "add", str(cargo_path), str(pyproject_path), str(npm_path)],
        check=True,
    )
PY
