# Plan: veld tracks shodh-memory, with veld's layers on top

Status: accepted by the operator 2026-10-03. Work is tracked in SPINE plan
`veld-shodh-upstream-20261003`; each upstream commit's outcome is a row in `LEDGER.tsv`.

## Operator direction

- veld builds on RocksDB while redb is unfinished (2026-10-03).
- veld takes everything usable from shodh-memory, follows upstream as changes land, and keeps
  veld's own layers on top.
- The work runs through to redb's equivalence with RocksDB: the gates in 0.1.

## Operator decisions (2026-10-03)

1. **ADR 0007 and the cargo defaults: approved.** RocksDB is the default engine and the port
   target; the default features are `rocksdb` plus `storage-redb`
   ([ADR 0007](../src/decisions/0007-rocksdb-default-engine.md)).
2. **Live back on RocksDB.** Done by the write-frozen cutover (0.5) on 2026-10-03.
3. **Web UI accepted (W11). Robotics (W12a) and anomaly finding (W12b) both taken.**
4. **Boot-guard fix (0.3): landed** as `f25a17fd`.
5. **The plan and the ledger live here,** in `docs/upstream/`.

## Where things stand (measured 2026-10-03)

| Fact | Value | Source |
|---|---|---|
| Fork base | upstream `2737b4f` (2026-03-27, PR #135), not `618d7b1` | normalised diff of veld's import `3e4d7ca7` against upstream `src/`: 14,348 differing lines, against 21,809 at `618d7b1` |
| Upstream since then | 307 first-parent commits, 768 non-merge, to `902daf8` (2026-09-27) | `git rev-list --first-parent` |
| Upstream releases since | none: v0.2.0 (2026-04-09) is the only tag, plus a GLiNER asset release | `gh release list` |
| veld since import | 493 commits | `git rev-list 3e4d7ca7..b2a47508` |
| Files changed on both sides | 163 (144 still in conflict) | per-file numstat, shodh→veld paths |
| Files upstream changed that veld did not | 454: 371 new, 26 clean in-place ports, 50 deletions, 7 veld removed | same |
| Translated patches applying cleanly | none of three trials: #559 rejected 2 of 4 hunks (one file absent), #330 rejected 3 of 3, #425 rejected 17 of 32 | `sed` rename then `git apply --reject` on veld main |
| Stored format | upstream postcard (#194), veld bincode with a pinned layout; neither reads the other's store | both reports |
| Upstream interface growth | HTTP routes 153→176, MCP tools 46→60, `SHODH_*` env vars 47→185 | upstream report |
| Already ported to veld | 20+ veld commits cite upstream PRs (#225, #235, #252, #281, #287, #290, #325, #334, #362, #389, #393, #400, #409, #411, #416, #442, #513, #518, #538, #539) | upstream report |
| veld fixes sent upstream | 10 merged Portll PRs since the fork; #318, #555 and #556 still open | upstream report |
| Live store | RocksDB since 2026-10-03 17:05Z, after the frozen cutover (0.5): 21 units, 2,703,826 records, every unit verified; tenants with data RocksDB 15, redb 15 (kept for the rollback window to 2026-10-10) | SPINE task 1.5 |
| Live binary | legacy-rocksdb build of veld `bdaa1263`, both engines compiled | `/health` |

The full inventories (upstream delta, veld's layers, conflict hotspots) were measured from clones
of both trees on 2026-10-03; `LEDGER.tsv` carries the per-commit result.

## Principles

1. **RocksDB is the engine ports are proven on.** Upstream is RocksDB, so a port tested on RocksDB
   is tested against upstream's own behaviour, and live runs RocksDB. redb stays behind the storage
   traits.
2. **Every port goes through the traits.** veld's ratchet test keeps direct `rocksdb::` use falling;
   ports are rewritten against `ColumnStore` and the other traits, so a new column family lands on
   both engines at once where the trait covers it. Where a trait lacks a capability, the trait
   gains it with a RocksDB implementation, and redb gets an explicit refusal that the parity test
   names (1.6). redb catching up is then tracked work, not a surprise at run time.
3. **Stored records only grow at the end.** bincode, like postcard, is positional. A ported struct
   field is appended; an enum variant goes last; every record type keeps a decoder for its old
   shape. Upstream inserted `EntityLabel` variants mid-enum and 74% of a live graph's nodes stopped
   decoding until a legacy decoder was added (upstream report). veld does not switch to postcard:
   export, import, migrate and encryption-v2 verify bincode records.
4. **Measure before a port changes ranking.** Upstream's recall gate runs; veld's has never run.
   Retrieval and graph ports wait on it.
5. **Every upstream commit gets a recorded outcome.** Ported, adapted, already in veld, declined
   with a reason, or superseded by a veld layer.
6. **Attribution travels with the code.** Upstream-derived files keep their Apache-2.0 notice;
   commit bodies name the upstream PR and commit; NOTICE stays current; third-party data keeps its
   licence duties (GeoNames is CC BY 4.0).

## Phase 0: storage direction (gates everything else)

0.1 **ADR 0007, amending ADR 0001.** RocksDB is veld's default engine and the target of ports.
    redb remains a second engine behind the traits, built by default, and becomes the default only
    when these hold, each with a named test or measurement:
    - merge pass and tombstone GC on redb (`run_merge_pass`, `gc_merge_tombstones`);
    - record-level encryption (encryption-v2) on redb;
    - backup and point-in-time restore on redb proven against the RocksDB engine's;
    - recall-gate scores equal on both engines over the same corpus;
    - a 14-day live soak with no boot-guard refusal and no decode failure.
0.2 **Cargo defaults.** Rename `legacy-rocksdb` to `rocksdb`, keeping the old name as an alias.
    Default features become `rocksdb` plus `storage-redb`; the default engine is RocksDB for a new
    data directory. CI tests the default, plus a redb-only build
    (`--no-default-features --features storage-redb`), so neither engine rots.
0.3 **Boot guard in the redb-only build.** Landed as `f25a17fd`: with no RocksDB code, an empty
    RocksDB directory reads as empty, and one with table files still refuses.
0.4 **CI shows clippy's verdict.** The clippy step pipes through `tee` without `pipefail`, which hides
    two deny-level errors in `tests/adaptive_memory_tests.rs` and `tests/memory_tiering_tests.rs`;
    fix both and set `pipefail`.
0.1a **Gate status (2026-10-04).** Merge pass and tombstone GC on redb: done (one
    implementation over the traits; the coverage matrix has no gap). Encryption-v2 on redb: done.
    Backup and point-in-time restore parity: done; the parity test found a RocksDB restore that
    deleted the tenant's data, fixed. Equal recall-gate scores: the gate runs, but veld's pipeline
    gives different rankings between repeats on fresh stores, which blocks an equality check
    until it is deterministic. 14-day live soak: needs the operator, since live runs RocksDB; a
    daily soak rehearsal on a clone can run meanwhile.
0.5 **Live back on RocksDB, with writes frozen.** Done 2026-10-03: rehearsed on an APFS clone
    (78 s migration, census and probes checked), then run on the stopped live store. Steps:
    1. Set aside the 6 pre-cutover RocksDB stores, since migrate refuses a target that holds records.
    2. Rehearse `veld migrate --from redb --to rocksdb --user all` on a clone, dry run first.
    3. Stop the server. Nothing writes while the real migration runs, so no write can land in redb
       after the snapshot.
    4. Run the real migration on the stopped store.
    5. Verify frame counts and digests per unit, on the frozen state.
    6. Flip `VELD_STORAGE_BACKEND` and start the server.
    7. Check the boot census: 15 RocksDB tenants with data.
    8. Rollback window, 7 days: a rollback runs `veld migrate --from rocksdb --to redb` into a set-aside
       copy, so writes made on RocksDB come back too. After 7 days, the redb stores go to the backup
       set.

## Phase 1: tracking machinery

1.1 **Remotes.** `upstream` remote in veld checkouts, pointing at varun29ankuS/shodh-memory; the
    Portll/shodh-memory-upstream mirror synced by the watcher.
1.2 **Translator.** `scripts/upstream/translate` turns an upstream commit into a veld-shaped
    patch:
    - the rename families: crate, binary, module and env var names, headers, data directories,
      MCP names, URLs;
    - veld's layout moves: `src/mcp.rs` is four files, `main.rs` is gone, and `serialization.rs`
      and `migration.rs` are absent.

    It writes the patch, the hunks that apply and the ones that do not. The output is a guide for
    an adapted port, not a merge.
1.3 **Ledger.** `docs/upstream/LEDGER.tsv`, one row per upstream first-parent commit after
    `2737b4f`, 307 rows today. Columns: upstream commit, PR, date, theme, wave, status, veld
    commit, reason. Seed the statuses from the 20+ commits already ported and the 10 merged
    Portll PRs. A test fails when a row has no status or names a veld commit that does not exist.
1.4 **Watcher.** A daily job:
    - fetches upstream main and appends each new first-parent commit as `pending`, with its theme
      and touched files;
    - opens a veld todo and a SPINE task per new PR;
    - records any upstream tag as a release and opens a "match upstream release" item.

    Upstream has not tagged a release since v0.2.0, so tracking follows merged PRs, `openapi.yaml`
    and the recall-gate baseline files.
1.5 **Recall gate first.** Port upstream's `recall_eval` binary and `recall.yml` workflow, then run
    it on veld main to set veld's baseline. No retrieval, graph or NER port merges without a gate
    run. The gate compares veld with veld on the same corpus and embedders. Upstream's published
    numbers are hypotheses to test, never a reason to accept or decline a port.
1.6 **Port checks: failing tests, not conventions.** Every wave starts only when these exist:
    - **Old-shape decode fixtures.** Each persisted type (Memory, Experience, RelationshipEdge,
      EntityNode, SemanticFact, Todo and the rest) has golden records at the fork shape and at each
      released shape, and a test decodes them all.
    - **Pinned enum discriminants.** A test fails when a stored enum's variant indices change, so an
      inserted variant fails CI instead of mis-decoding silently.
    - **Route and tool auth inventory.** Extends `router.rs`'s `public_router_paths_match_const`:
      every HTTP route is in the public allowlist or behind `auth_middleware`, and every MCP tool
      requires the API key.
    - **Trait parity.** Every storage-trait method has an implementation on both engines, or an
      explicit refusal a test names, so a RocksDB-only capability shows up in CI and not at run time.
    - **Attribution.** A ledger row names each port's commit, and each commit's body names its
      upstream PR; NOTICE and the data-licence list are checked against the vendored assets.
    - **Live clone dry run.** Before each wave reaches live, `veld migrate --dry-run` runs on a clone
      and must report 0 undecodable records.
1.7 **Ownership and watcher health.**
    - The watcher writes a heartbeat with the last upstream commit it fetched, and the ledger test
      fails when the heartbeat is more than 48 hours old.
    - Hotspot files take SPINE file claims for the length of a wave: `graph_memory.rs`,
      `memory/mod.rs`, `handlers/state.rs`, `memory/storage.rs`, `handlers/recall.rs` and the
      rest of the top 25.
1.8 **Performance budget.** Budgets for `/api/recall`, `/api/relevant` and remember are in
    `latency-budgets.json`, measured on 2026-10-04 on a clone of the live store:
    `scripts/upstream/latency_probe.py --budgets docs/upstream/latency-budgets.json` checks them.
    p95 budgets: recall 30 s, relevant 60 s (the request timeout), remember 1 s. The machine's
    load moved between 22 and 70 during the measurement and the cross-encoder rerank dominates
    every recall, so these catch a gross regression, not upstream's +190 ms per recall; a port
    that changes ranking cost is measured on a quiet machine before and after.

## Phase 2: catch-up waves

Each wave works the same way:

1. ledger rows;
2. translated patches;
3. adapted ports through the traits;
4. the struct-growth rule and its legacy decoders;
5. tests, plus a gate run where ranking can move;
6. one commit per upstream PR, naming it;
7. ledger status;
8. deployment to live through a clone rehearsal.

Waves run in this order because each needs the one before it, or carries more risk than it.

| Wave | Upstream work | Notes for veld |
|---|---|---|
| W0 Measurement | recall harness and eval gates: RH #271–#281, #312, #328, #548, #561, #570 | the harness is mostly ported; the gate binary and workflow are not |
| W1 Fixes and security | storage and integrity fixes, #559 audit writes, the security commits (#399, #492, #513, #518 partly in) | #513 turns `ENFORCE_HTTPS` on by default, but it governs only outbound integration-API URL overrides (Linear, GitHub), so it ports without touching loopback clients. #399 adds an authenticated local IPC transport (owner-scoped Unix socket or named pipe, mutual auth, on by default); W1 opens with an ADR deciding whether veld's loopback clients (desktop, MCP, hooks) take it beside HTTP with the API key |
| W2 Retrieval and ranking | #181 temporal credit, #320, #324, #536 cross-encoder by default, #538 (in), #560, #562, FLAT fusion, PPR | veld has its own cross-encoder registry and LLM refiner; reconcile, then measure: upstream reports p@1 0.33→0.49 and +190 ms per recall |
| W3 Knowledge graph | #213 curvature, #321, #323, #336/#364, #340, #442 (in), #461, #509; `relation_stats` column family; four edge tail fields | veld decays edges exponentially, upstream by power law: measure both and record the choice; veld still makes the CoRetrieved edges upstream stopped making |
| W4 Causal lineage | #330, #389 (partly in), #419; `RelationType::Precedes`; `/api/lineage/root-cause` | veld's lineage test is skipped today; upstream reports root-cause P@1 0→0.867 |
| W5 Facts, consolidation, todos | #471, #486, `Todo.blocked_by`; #235, #393, #411 are in | appended fields only |
| W6 NER and entities | GLiNER (149 MB model, replaces TinyBERT), `entity_alias` column family, 12 `EntityLabel` variants, spaCy `en_core_web_sm` (vendored `spacy-rusty`, 14 MiB), gazetteer, Wikidata slice | variants appended last; model fetched by pinned hash; GeoNames attribution in NOTICE; build-size budget agreed first |
| W7 Storage and integrity | #425 oplog, `sources`, `source_cursor`, #532 integrity scrub (80 commits squashed) | through the traits on both engines; checked by export/import round trip |
| W8 Embeddings and vectors | #496 token-aware chunking, #514 Vamana insert pruning (self-recall 45.6%→99.9%) | veld has its own chunking fix and Vamana drift rebuild; merge, then measure |
| W9 MCP and hooks | 14 new tools (#428, #464, #470); rmcp 0.12→1.4 | veld's MCP server is split into four files; tool names keep the veld prefix |
| W10 HTTP API | 23 new routes, `openapi.yaml` | veld's auth middleware covers every new route |
| W11 Web UI, seat workbench, demo | 139 commits (#430, #437, #466, #484): a React app under `front/` served by the binary, the seat workbench, demo seeders | accepted 2026-10-03. Served behind veld's auth, beside the desktop and TUI, which stay. It runs after W10 because it calls the routes W10 brings. It includes the workbench's anomaly view (W12b) |
| W12 Geo, robotics, anomaly | described below | taken 2026-10-03. W12b runs after W3 and W6, W12a after W10; toponyms (#435) are ported with W6 |

### W12 in detail

**W12a Robotics.** Memory for robots over Eclipse Zenoh, with ROS2 robots attached through
`zenoh-bridge-ros2dds` or `rmw_zenoh`. veld already carries the transport from the fork
(`src/zenoh_transport/`, behind `--features zenoh`). Upstream has changed it 13 times since.
Since the fork upstream added:

- **Zenoh key expressions:** remember, recall, forget, streaming ingestion (sensor, event,
  conversation), mission start and end, fleet join and leave discovery, a fleet roster and a
  health check.
- **Robotics recall modes (#239):** spatial, mission and reinforcement-learning modes.
- **Learning from rewards:** reward-based Hebbian learning (#239), a tunable reward learning rate
  with a reward-gradient harness (`272e450`), and an entity-salience reward loop (#217).
- **Input validation:** checks on reward, heading, sensor data and failures-only queries
  (`1951bc4`).
- **Geo composition (#418):** geo constraints composed with semantic recall, a GDELT geo ingest,
  and a geotemporal evaluation set.
- **Coordinates in recall:** recall returns coordinates and validates the robotics geo filter
  (#429).
- **Pinned dependency:** zenoh held at 1.8.0 so `lz4_flex` stays clear of RUSTSEC-2026-0041
  (#539).
- **Demo:** a warehouse-robot mission seeder.

Cost: the zenoh feature's dependency tree, which carries an advisory to watch, and
`Experience`'s robotics and geo fields. It is useful only if veld serves robots or edge devices.

**W12b Anomaly finding.** Deterministic, with no model in the loop:

- **Surprise components (#353):** every episode stores five measures, computed in the
  entity-pair loop at ingest:
  - mean birth-PMI, where low means familiar entities in an unfamiliar combination;
  - the share of entities new to the user's graph;
  - the share of pairs the relation typer could not type;
  - the share of pairs the PMI gate dropped;
  - the share anchored on hub entities.
- **`/api/anomalies`:** scores each episode at read time as z-scores against the user's rolling
  baseline, so thresholds can change without re-ingesting, and every flag is explained component
  by component.
- **Entities and the graph (#357):** each anomaly names its entities, and the dashboard lights them
  up in the knowledge graph.
- **MCP tool** (`2cf1dfd`) for agents.
- **Workbench view:** finds places far from the corpus's median location (modified z above 3.5)
  and figures that disagree with their unit's corpus median. Pure arithmetic in the web UI, so it
  comes with W11.
- **Not on upstream's main:** FreeGAD, a training-free graph anomaly detector ported from PyGOD and
  matched on the BOND enron set, and a planted-needle benchmark. Both are retired on
  `archive/feat/anomaly-engine`.

Cost: new fields on stored graph episodes (append-only, with a decoder). It depends on the PMI
gate (W3) and the relation typer (W6), so it runs after both. For veld it would flag memories
that do not fit a user's history, a check no veld layer makes today.

Draft translations of the 26 clean in-place files and the new files that touch no veld layer
can go to the local Qwen lanes; hotspot files (`graph_memory.rs`, `memory/mod.rs`,
`handlers/state.rs`, `memory/storage.rs`, `handlers/recall.rs` and the rest of the top 25) are
ported by hand.

## Phase 3: veld's layers on top

These stay and are maintained as the base moves. Each keeps its seam, so ports land beside it,
not through it:

- **Storage:** the redb engine, the storage traits, export, import and migrate, the boot guard.
  Seam: the traits.
- **Security:** encryption at rest (keystore v2, key rotation, `veld-keyctl`), and auth with
  password, TOTP and recovery codes. Seams: the store layer and handler middleware.
- **Memory model:** bi-temporal "as of" queries, record kinds and facets, the intent log, the query
  planner, the slow store, runtime dials, reconsolidation, the sleep-time rewriter. Seam: veld's
  own modules.
- **Embeddings:** LM Studio and Nomic embedders, embedding alignment, the cross-encoder registry.
  Seam: the embedder trait.
- **Fixes upstream lacks:** the bounded `/api/relevant` walk (`b2a47508`) and the `chunk_text`
  progress fix.
- **Desktop, TUI, docs site, wiki rendering.** Seam: the HTTP API.

The fixes are offered upstream as PRs, as ten were before, so the two trees converge where
neither needs to differ. Each offer gets an outbound ledger row (veld commit, upstream PR,
state), so traffic in both directions is recorded. Divergence is recorded in the ledger as
`veld-layer`, with the reason.

## Phase 4: steady state

- The watcher runs daily, with triage weekly. Every new upstream PR gets a ledger status within
  14 days; a security fix within 2.
- veld cuts a release after each wave, then monthly. Each release note lists the upstream PRs it
  takes and the ledger counts by status.
- A quarterly review of the declined rows, since upstream work can make a declined item worth
  taking.

## Evaluation

Bifocal+ scored the first draft at 68.4 of 120 and 67.1 (an independent rater, blind to those
scores), with no blocker. Eleven FMEA findings were mandatory; three were
contested, with the raters three or more points apart on a factor (F1, F3, F6). The changes above
answer them:

- the write-frozen cutover and its rollback window (F1, F2);
- old-shape fixtures and pinned discriminants (F3, F4);
- the route and tool auth inventory (F5), and the #399 ADR (F6; re-read on 2026-10-04: #513's HTTPS default covers only outbound integration URLs, and #399 is a local IPC transport, so F6's loopback-HTTPS concern does not arise);
- the attribution check (F7), the watcher heartbeat (F8) and the trait parity test (F9);
- veld-against-veld gating (F10) and SPINE claims on hotspot files (F11).

The amended plan has not been re-scored. The evaluation is kept with the cobolwork evaluations
(`bifocal_2026-10-03_veld-shodh-upstream.json`).
