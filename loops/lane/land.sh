#!/bin/zsh
# Commits paths on feat/pli through commit-phase, only when every PL/I test passes.
#   land.sh <subject> <body or ""> <path>...
W=/Users/portll/Repositories/Portll/cobolwork-pli-20261001
subject=$1; body=$2; shift 2
out=$(cd $W && node --test --test-concurrency=1 test/pli-*.test.mjs 2>&1)
if ! print -r -- "$out" | grep -q "^ℹ fail 0$"; then
  print -r -- "$out" | grep -E "^✖|^ℹ (pass|fail)" | head -20
  echo "land.sh: tests fail; nothing committed" >&2
  exit 1
fi
args=(-m "$subject")
[ -n "$body" ] && args+=(-m "$body")
CW_COMMIT_REPO=$W node /Users/portll/Repositories/Portll/commitwork/bin/commit-phase.mjs $args -- "$@" 2>&1 | grep -E "landed|REFUSED|·" | grep -o "landed [0-9a-f]*\|REFUSED.*\|· .*" | head -3
