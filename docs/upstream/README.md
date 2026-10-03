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
scripts/upstream/ledger.py check   # every status valid, every reason present, every veld commit real
```
