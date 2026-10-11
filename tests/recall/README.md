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

## The gate baseline (`locomo-gate`)

`.github/workflows/recall.yml` runs the 100-case `locomo-gate` suite and compares each engine's
run against two checked-in files:

- `locomo-gate-baseline.json`: aggregate metrics and the rerank setting they were measured
  under. `recall-eval` gates on these, and refuses a run reranked differently; its exit code is
  the pass or fail.
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

## Files

- `corpora/shodh-smoke.jsonl` — L1 smoke corpus (memories ingested by the runner).
- `smoke_cases.jsonl` — 108 hand-crafted query cases with graded relevance labels,
  spanning six categories (decision / code / temporal / entity / multi_hop / negation).
- `baseline.json` — frozen per-metric scores for the smoke suite.
- `corpora/locomo-gate.jsonl`, `locomo-gate-cases.jsonl` — the LOCOMO gate suite;
  `locomo-gate-baseline.json` and `locomo-gate-baseline.per-case.json` — its baseline.
- `corpora/locomo.jsonl`, `locomo_cases.jsonl` — the 1,531-case held-out LOCOMO set, run ungated by
  dispatching the gate with `suite=locomo`.

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
| 2026-10-06 | `2f1f823` | minilm-l6-v2   | `locomo-gate`, veld's own: RocksDB report of run 37485608452, 3 repeats, redb equal on every metric. The per-case file from the same run was added on 2026-10-07, and so was its `rerank` field (mxbai-rerank-xsmall-v1, cap 20): an arm64 rerun of the tree with reranking on differed from it in 1 case of 100, with reranking off in 40. |
