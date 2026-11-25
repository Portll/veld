# Recall Harness Fixtures & Baseline

This directory holds the fixtures, ground truth, and frozen baselines for the recall harness
(`src/recall_harness/`).

## What is wired

| Asset | State |
| --- | --- |
| `src/recall_harness/` library (`run_smoke_suite`, `run_longmemeval`) | wired, unit-tested |
| `recall-eval` binary (`src/bin/recall_eval.rs`, `--suite`, `--layer`, `--baseline`) | ported from upstream `20de49c3`, the revision veld's harness tracks |
| `.github/workflows/recall.yml` | the recall gate: weekly and on dispatch, once per storage engine |
| `scripts/recall_diff.py`, `scripts/recall_engines_equal.py` | the diff table, and the check that both engines score the same |
| Smoke corpus + `smoke_cases.jsonl` + `baseline.json` | present, consumed by the library |
| LOCOMO gate fixtures, `locomo-gate-baseline.json`, `locomo-gate-baseline.per-case.json` | read by the gate |

The gate compares veld with veld: upstream's published numbers are hypotheses to test, never a
reason to accept or decline a port (`docs/upstream/PLAN.md`, 1.5).

## Who did what to whom (`directional`)

`directional` (ported from upstream #571) asks 240 questions over 60 pairs of twin memories made
of exactly the same words with the roles swapped ("Priya invited Caroline …" and "Caroline invited
Priya …"), in three categories: `dir_object` ("Who did Priya invite?"), `dir_subject` ("Who invited
Caroline?") and `dir_passive` (the same questions over passive memories). Each question's twin is
its decisive distractor, so a retriever that ignores direction scores exactly 0.5 p@1. Upstream
measured the embedder at cosine 0.979 between twins and 51% correct. Diagnostic, not gated:
`gh workflow run recall.yml -f suite=directional -f record=true`. The fixtures test
`directional_fixtures_give_every_gold_a_same_words_twin` pins the twin property; the generator is
`benchmarks/build_directional_suite.py`.

## The gate baseline (`locomo-gate`)

`.github/workflows/recall.yml` runs the 100-case `locomo-gate` suite and compares each engine's
run against two checked-in files:

- `locomo-gate-baseline.json`: aggregate metrics, the rerank setting they were measured
  under, and any candidate-generation dial the run overrode (`retrieval_dials`, today
  `date_window_leg`). `recall-eval` gates on these, and refuses a run reranked differently or
  run under other dial overrides; its exit code is the pass or fail. A baseline without
  `retrieval_dials` was recorded with no overrides, which is how the gate workflow runs, so it
  is compared, and a change to a dial's engine default is measured as code.
- `locomo-gate-baseline.per-case.json`: the per-case records from the same run.
  `scripts/recall_diff.py` uses them to list every case whose result moved: lost gold, gained
  gold, p@1 flips and rank-only moves.

At n=100 the 2% tolerance is below one case for p@1 (one case is 0.01), so on that metric the
gate is a no-net-loss check. Repeats must give identical rank lists, so a case that moves on the
same runner class is a real change; the case list is how a reviewer decides whether a change is
worth what it moved.

Regenerate both files together, from one run on `main`, after a merged change that intentionally
moved quality:

1. `gh workflow run recall.yml --ref main -f record=true`.
2. `gh run download <run-id> -n recall-rocksdb`.
3. Copy `current-rocksdb.json` to `locomo-gate-baseline.json` and `per-case-rocksdb.json` to
   `locomo-gate-baseline.per-case.json`. Check that `git_sha` names the commit you meant.
4. Add a row to the regeneration history below, naming the run.

## Model tapes

ONNX Runtime chooses its kernels by CPU, so the embedder and the cross-encoder return different low
bits on different runner classes and near-ties flip. A model tape (`src/embeddings/model_tape.rs`)
takes the models out of the measurement: `VELD_MODEL_TAPE_RECORD=<path>` records every output the
MiniLM embedder, the cross-encoder and, when it loads, the neural NER give, keyed by the exact input;
`VELD_MODEL_TAPE_REPLAY=<path>` answers every call from the tape and runs no model arithmetic. A
replay miss, a tape recorded from other model files, or a model the replay never loaded fails the
run as infrastructure rather than falling back. A recording also scores every (query, corpus item)
cross-encoder pair the run did not, so a retrieval change that hands the reranker another candidate
still finds it.

The workflow's `model_tape` input drives it. `pinned`, the default and what the schedule runs,
replays the tape named in `tests/recall/model-tape.json` (`{"tag", "asset", "sha256"}`, a GitHub
release asset) after checking its sha256; with no pin file, as now, the models run live. `record`
uploads each engine's tape as `model-tape-<engine>`; `replay` with `tape_run` replays another run's
RocksDB tape. Every report records the tape it replayed as `model_tape`, and `recall-eval` refuses
to compare a live run with a replayed one, or two different tapes. So pinning a tape is three steps
taken together:

1. `gh workflow run recall.yml --ref main -f model_tape=record`, and check the run reports no tape
   problem and that repeats after the first added no entries.
2. Publish `model-tape-rocksdb.jsonl.gz` as a release asset under a new tag, and write the pin
   file with that tag, the asset name and its sha256.
3. Record the baseline again with `model_tape=pinned` and `record=true`. The baseline and the pin
   must name the same tape.

A change to what a model is fed (another corpus text, a prefix, another candidate source) or a
model or ONNX Runtime upgrade needs a new tape: replay misses fail the run, and the tape refuses
model files other than the ones it was recorded from.

## Embedder arms

The workflow's `embedder` input picks the primary embedder: `minilm-l6-v2`, the default and what
the baseline measures, or `arctic-embed-xs` (snowflake-arctic-embed-xs, see
`docs/src/guides/deploying.md`). It sets `VELD_EMBEDDER` for the eval, makes `model-fetch` fetch
that model, and names the models cache, so an arm restores and runs its own model. The report's
`embedder` field names the model that produced the vectors, and `recall-eval` refuses to gate one
embedder against another's baseline, so an alternative arm is a recording:

```text
gh workflow run recall.yml --ref <branch> -f record=true -f embedder=arctic-embed-xs
```

A pinned tape holds one embedder's outputs, and it refuses to replay under another embedder's
model files. Once a tape is pinned, run an alternative arm with `model_tape=off` or record a tape
for it.

## Files

- `corpora/shodh-smoke.jsonl` — L1 smoke corpus (memories ingested by the runner).
- `smoke_cases.jsonl` — 108 hand-crafted query cases with graded relevance labels,
  spanning six categories (decision / code / temporal / entity / multi_hop / negation).
- `baseline.json` — frozen per-metric scores for the smoke suite.
- `corpora/locomo-gate.jsonl`, `locomo-gate-cases.jsonl` — the LOCOMO gate suite;
  `locomo-gate-baseline.json` and `locomo-gate-baseline.per-case.json` — its baseline.
- `corpora/locomo.jsonl`, `locomo_cases.jsonl` — all 1,531 LOCOMO cases over ten conversations, run
  ungated by dispatching the gate with `suite=locomo`. They include every gate case, so this set
  measures fit as well as generalisation.
- `corpora/locomo-holdout.jsonl`, `locomo-holdout-cases.jsonl` — the same set without the gate's
  conversation (conv-42) and without the cases in `locomo-holdout-spent.txt`: 5,253 turns in nine
  conversations, and 1,132 cases on 2026-11-17 (1,166 before the cases spent on 2026-10-30 and
  2026-11-15). Run ungated with `suite=locomo-holdout`. This is
  the number to report, with its interval.
- `locomo-holdout-spent.txt` — hold-out cases whose individual results someone has examined.
- `corpora/agent-episodes.jsonl`, `agent-episodes-cases.jsonl` — `suite=agent-episodes`: 187
  coding-agent memories over five fictional projects, 147 of them in 14 sessions threaded the way
  the Claude Code hook threads a session (each item's `episode` names its episode id, sequence
  number and the item it follows; the harness sends them as `episode_id`, `sequence_number` and
  `preceding_memory_id`), and 40 unthreaded project notes. Every memory carries its project tag.
  80 cases: `adj` asks about a fix or outcome in the words of the memory before it, `back` about a
  cause in the words of the memory after it, `dir` are direct lookups (controls), `dec` decisions,
  `tmp` a fact that changed between sessions, `xp` the same kind of event in two projects. Cases
  `agent-dev-*` may be read case by case; `agent-ho-*` are reported in aggregate only, as with
  `locomo-holdout`. Synthetic, authored 2026-10-21 without reference to the ranking code; ungated.

## The held-out set (`locomo-holdout`)

Nothing is tuned against it: do not diagnose a miss on it and then change recall to fix that case,
or it stops being held out. Diagnose on the gate, measure on the hold-out.

A case whose individual result you have read (a moved-case list, a miss you looked into) is spent.
Add its id to `locomo-holdout-spent.txt` under a dated comment naming the source, and regenerate.
It stays in the corpus, so the retrieval problem is unchanged; it leaves the scored cases. The
first entries are the 163 hold-out cases in the moved list behind
`evaluations/recall/2026-10-20-heldout-loss-attribution.md`, which shaped the frame-word and
short-acronym changes.

- Regenerate it with `python3 scripts/build_locomo_holdout.py` whenever `locomo.jsonl`,
  `locomo_cases.jsonl`, the gate cases or the spent list change. The fixtures test
  `locomo_holdout_is_the_full_suite_without_the_gate_conversations_or_spent_cases` fails until
  you do.
- Report it with intervals: `python3 scripts/recall_ci.py <per-case.json>` prints each metric's
  mean and 95% percentile-bootstrap interval, overall and per category. Two runs whose difference
  lies inside the intervals have not been shown to differ. For an A/B on the same cases pair them:
  `recall-eval --ablation` reports paired bootstrap intervals per arm, and `scripts/recall_diff.py`
  lists the cases that moved. For two dispatched runs (a dial arm against the shipped default),
  `python3 scripts/recall_compare.py <baseline.per-case.json> <candidate.per-case.json>` gives the
  paired difference twice: per case, and per conversation, where every question about one
  conversation moves as a unit. LoCoMo's questions share their conversation's memories, so the
  per-case interval is too narrow; decide on the per-conversation columns. With the hold-out's
  nine conversations the per-conversation randomisation test is exact and cannot go below
  p = 2/512.
- These are retrieval metrics (where the gold turn ranks). They are not the LLM-judged answer
  accuracy that Mem0, Zep and Letta publish for LoCoMo, and do not compare with it.

## Running the harness

Through the library, from a test or a caller you write:

```rust
let report = veld::recall_harness::run_smoke_suite(&inputs)?;
```

The two end-to-end runner tests are `#[ignore]`d because they pay the ingest cost:

```bash
cargo test --lib recall_harness::bridge_harness -- --ignored --nocapture
cargo test --release --lib -- --ignored runner_repeats_2_produces_same_quality_as_repeats_1
cargo test --release --lib -- --ignored runner_layer_all_emits_six_modes_with_per_mode_determinism
```

The gate's repeat check runs locally as an integration test, with the settings `recall-eval`
pins. It ingests the LoCoMo gate corpus into two fresh stores and fails on any case whose top-10
differs; `RECALL_DET_LAYERS=all` names the first pipeline stage that diverges:

```bash
cargo test --release --test recall_determinism -- --ignored --nocapture
```

### Aging: the only way decay is visible

`RunInputs::age_days` ages knowledge-graph edges via `MemorySystem::simulate_edge_aging` at the
production ~6h cadence, **after** ingest and **before** the query passes. It defaults to `0.0`, and
`age_days <= 0` is an explicit no-op.

This matters more than it looks: at the default, a run ingests and queries within seconds, so every
edge decays by the same ≈1.0 factor and **no change to any decay constant can move any metric**. A
uniform monotone factor cannot reorder a ranking. Tuning edge decay against a run at `age_days = 0`
measures nothing. Run at 0 / 7 / 30 / 90 and diff the reports.

### Storage path

**Storage MUST point outside directories watched by a search indexer, antivirus, or sync daemon.**
Watcher processes intermittently lock freshly written tantivy segment files, which silently drops
BM25 commit batches and makes rankings depend on ambient machine state (root-caused 2026-06-12:
432/432 smoke comparisons diverged with storage under a watched tree; 0–1 outside one). The harness
now hard-fails ingest when a commit batch is lost after retries, so a watched path errors out rather
than producing garbage numbers.

### Frozen clock

The eval scoring clock is pinned to a fixed anchor by `pin_harness_threads()`
(`src/recall_harness/runner.rs`) via the `SHODH_EVAL_NOW` environment variable, so recency
components cannot drift between repeat passes or rot a baseline as the static corpus ages against
wall-clock time. The variable name is inherited from shodh and is the live name in this codebase —
do not "correct" it without changing the readers in `runner.rs` and `bridge_harness.rs`.

## Per-pipeline-layer attribution (`LayerMode`)

`RunInputs::layer_modes` selects which subset of the retrieval pipeline runs; `recall-eval --layer
all` runs every rung, and the gate's `layer` dispatch input passes it. An empty vec is treated as
`[LayerMode::Full]`. Modes are **cumulative**:
each row adds one stage on top of the row above it. Report keys come from `LayerMode::report_key()`
in `src/memory/types.rs`.

| Report key    | Stages added                                                           |
| ------------- | ---------------------------------------------------------------------- |
| `vamana_only` | Layer 3 vector ANN only (cosine + tie-break).                          |
| `+spreading`  | + Layer 2 graph spreading activation (RRF over vector ⊕ graph).        |
| `+bm25`       | nothing in veld: the Layer 4 BM25 leg runs on every rung (see below).   |
| `+rerank`     | + Layer 4.9 ontological re-rank and the Layer 5.3 cross-encoder.       |
| `+facts`      | + Layer 0.7 / 4.8 fact-source boost from consolidated knowledge.       |
| `full`        | + Layer 0.4/0.5/0.6 pre-filters, Layer 4.6 interference, Layer 4.7     |
|               | prospective signal, full Layer 5 unified scoring (recency × importance |
|               | × arousal × credibility × tags × feedback × quality), retrieval        |
|               | competition, Hebbian coactivation, hierarchy expansion.                |

Pass `LayerMode::ALL` to run every mode in one invocation; the report's `layers` map gains one entry
per mode.

### Caveats — read these before staring at the numbers

1. **`+bm25` equals `+spreading`.** `memory/recall.rs` gates no stage on `PlusBm25`, so the BM25
   leg is fused on every rung, `vamana_only` included. Upstream gates it (#270); veld's port of the
   ladder did not, and the gate is owed in `recall.rs`.

2. **Modes below `full` skip Layer 5 unified scoring.** Per-layer ndcg numbers will look strictly
   *lower* than `full` for reasons that are *not* "this stage didn't help" — they include the
   absence of recency, importance, arousal, credibility, feedback, quality-gate, and Hebbian
   multipliers. Read the table as **deltas between adjacent rows**, not as standalone absolutes.

3. **Cumulative-only by design.** You cannot ask for "BM25 without spreading" or "rerank without
   facts" — only the six cumulative modes exist. A non-cumulative ablation is a different feature.

4. **Only the LoCoMo gate is gated.** `recall.yml` gates the `locomo-gate` suite at `--layer
   full`; the smoke baseline and the per-layer numbers are advisory.

5. **Small suites move in whole cases.** At n cases a whole-case metric moves in steps of 1/n, and
   a category's in steps of one over its own count: 0.04 on the gate's 25 multi_hop cases. Read a
   delta against that step before reading it as an effect.

## Regeneration history

| Date       | SHA       | Embedder       | Notes                          |
| ---------- | --------- | -------------- | ------------------------------ |
| 2026-05-03 | `6756665` | minilm-l6-v2   | Initial capture (RH-6, #268).  |
| 2026-08-08 | `ec7abd2` | minilm-l6-v2   | LOCOMO gate baseline; producer not in repo. |
| 2026-10-06 | `ef3e2ec` | minilm-l6-v2   | `locomo-gate`, veld's own: RocksDB report of run 37485608452, 3 repeats, redb equal on every metric. The per-case file from the same run was added on 2026-10-07, and so was its `rerank` field (mxbai-rerank-xsmall-v1, cap 20): an arm64 rerun of the tree with reranking on differed from it in 1 case of 100, with reranking off in 40. |
| 2026-10-18 | `0a841a9` | minilm-l6-v2   | RocksDB report of run 39132263820, 5 repeats, models live, `RECALL_DIAG_K=100` so the per-case `recall_at_50`/`recall_at_100` fields are real for the first time (41 cases miss at 10; 21 of them hold gold in 11..100, 20 hold none). p@1 0.37 → 0.35 and recall@10 0.522 → 0.519 against `ef3e2ec`: the drop the 2026-10-12 schedule reported, now the baseline. redb differed in 1 case of 100 (`conv-42_q58`, a gold at rank 9 vs 10), so `recall_engines_equal.py` was red on this run by a near-tie, not a defect. Miss taxonomy: `scripts/recall_miss_taxonomy.py`. |
| 2026-10-21 | `90572ea` | minilm-l6-v2   | RocksDB report of run 39603628147, 5 repeats, models live, redb equal (`Engines score equal` green). Re-recorded because the measurement changed, not only the code: `f744e186` ingests with the tags `remember` stores (YAKE keywords, cold-start entities, redaction, the selectivity cap) and `90572ea9` keeps a keyword that overlaps an event name out of causal typing. Against `0a841a9`: p@1 0.35 → 0.38, recall@10 0.519 → 0.559, MRR 0.425 → 0.459, nDCG@10 0.427 → 0.460; 3 cases lost gold from the top 10, 9 gained. p50 890 ms with `VELD_CROSS_ENCODER_CACHE=0`: the 332 ms a `294ceada` gate showed came from the shared pair cache serving repeats 2..5. 38 cases miss at 10; 19 of them hold gold in 11..100. Decide defaults on `locomo-holdout`, not on this suite alone. |
| 2026-11-18 | `60a0400` | minilm-l6-v2   | RocksDB report of run 44262500801, 5 repeats, models live, redb equal. Re-recorded because the cross-encoder defaults changed (share 0.5 for every shape but Identifier, top 30 of a 50-candidate pool; chosen on `locomo-holdout`, see `docs/src/benchmarks.md`), so the gate refused the cap-20 baseline. Against `90572ea`: p@1 0.38 → 0.46, MRR 0.459 → 0.530, recall@10 0.559 → 0.641, nDCG@10 0.460 → 0.541 (9 cases gained rank 1, 1 lost it). That includes the `c10715e`/`1ceb1e6` ranking fixes, which had moved the scheduled gate to p@1 0.43 and turned it red on one open_domain case. p50 1214 ms against 659 ms on the 2026-11-16 schedule, a different runner; the paired hold-out runs put the cost of the larger budget at about 4%. |
| 2026-11-18 | `5f6390f` | minilm-l6-v2   | RocksDB report of run 44268537774 on `87f16f0`, train C3 before its rebase onto `6925ad4` (the two trees differ only in these baseline files), 5 repeats, models live, redb equal. Re-recorded because train C3 changes the ranking: neighbour expansion and turn context for episode-threaded memories (`162b7a2`), equal vector similarities ordered by content and Person typing only when the text says so (`462d6b2`), entity word matches weighed by specificity and post-fusion boosts as shares of the fused score (`e9c6606`), with Layer 5.97 off (`3ddeb36`). Against `60a0400`: p@1 0.46 → 0.48, MRR 0.530 → 0.569, recall@10 0.641 → 0.697, nDCG@10 0.541 → 0.582 (4 cases gained rank 1; `conv-42_q119` and `conv-42_q49` lost it); 9 cases gained gold in the top 10, `conv-42_q30` and `conv-42_q49` lost some. p50 1207 ms against 1214 ms. 23 cases miss at 10; 14 of them hold gold in 11..100. The report now records the ingest path and the similarity-bar counts, and the per-case file each case's top results with their signals (`2c568b8`), so it is four times larger. |
| 2026-11-26 | `0ee1649` | minilm-l6-v2   | RocksDB report of run 45628700508 on `7842936`, 5 repeats, models live, redb equal. Re-recorded because main turns the cross-encoder off unless the `use_reranking` dial turns it on (`0ee1649`), so the gate's comparison against `87f16f0` measured another pipeline and was refused. Rerank on lives on `land/c3-20261124-reranker`. Against `87f16f0` (cross-encoder mxbai-rerank-xsmall-v1, cap 30): p@1 0.48 → 0.35, MRR 0.569 → 0.478, recall@10 0.697 → 0.669, nDCG@10 0.582 → 0.505; 12 cases lost rank 1 with reach unchanged and 1 gained it (`conv-42_q119`); 9 cases lost gold from the top 10, mostly multi_hop, and `conv-42_q106` and `conv-42_q66` gained some. p50 117 ms against 1207 ms. 24 cases miss at 10; 15 of them hold gold in 11..100. |
