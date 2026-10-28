# Tracking upstream shodh-memory

veld forked shodh-memory at upstream `2737b4f` and follows it: [PLAN.md](PLAN.md) says how, in
what order, and which of veld's own layers stay on top.

`LEDGER.tsv` holds one row per upstream first-parent commit after the fork base (`direction`
`in`) and one per veld change offered upstream (`out`).

| Column | Meaning |
| :--- | :--- |
| `upstream_commit` | upstream first-parent commit; for `out`, the merge commit once merged |
| `pr` | upstream pull request |
| `theme`, `wave` | the subsystem and the catch-up wave in PLAN.md; assigned from the subject and touched paths, corrected when a wave triages its rows |
| `status` | `in`: `pending`, `ported`, `adapted`, `partial`, `already-in-veld`, `veld-origin`, `declined`, `superseded`, `veld-layer`. `out`: `open`, `merged`, `closed` |
| `veld_commit` | the veld commit(s) carrying the change, comma-separated |
| `reason` | required for `adapted`, `partial`, `declined`, `superseded`, `veld-layer` and `closed` |

```bash
git remote add upstream https://github.com/varun29ankuS/shodh-memory.git
git fetch upstream
scripts/upstream/ledger.py sync    # appends new upstream commits as pending
scripts/upstream/ledger.py check   # every status valid, every reason present, every veld commit real and attributed
```

`RELEASES.tsv` records each upstream release tag: `before-fork` for tags the fork already
contains, `pending` until a veld release matches it. Upstream's `archive/*` tags preserve
retired branches and are not releases.

The `Upstream Watch` workflow runs daily: it fetches upstream, appends new commits and release
tags as `pending`, and commits only when upstream has moved. `ledger.py heartbeat` reads its
last successful run from GitHub Actions and fails past 48 hours.

`scripts/upstream/translate <upstream commit>` rewrites an upstream commit's diff into veld's
names and layout and applies it with `git apply --reject` in a throwaway worktree, reporting
the hunks that apply and keeping the rejects. On 2026-10-03, against veld `b1d9a948`, PR #559
rejected 4 of 4 hunks (one file absent in veld), #330 3 of 3 and #425 19 of 34; with
`--context 1`, 3 of 4, 3 of 3 and 14 of 34. A port is written by hand from that guide, through
the storage traits.

A port commit made from 2026-10-04 on names its upstream PR (`#559`) or commit (`355e0b7`) in
its message, and `check` fails on one that does not. The ports made before then are recorded in
the ledger instead; their commit messages mostly do not name the PR.
