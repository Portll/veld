#!/bin/sh
# What a carried verdict still has to pass. spine's bin/ci-verdict.mjs runs this in place of
# ci-local.sh when the commit rebases a patch whose full run was green, across commits on main that
# share no path with the patch and none listed in scripts/ci-rebase.global. Main moved under a
# running verdict twice in one day on 2026-10-24, each time through files the patch never touched,
# and each move cost another 30 to 90 minute matrix.
set -eu
exec "$(dirname "$0")/ci-local.sh" --rebase
