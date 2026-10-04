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
| LOCOMO gate fixtures + `locomo-gate-baseline.json` | read by the gate |

`locomo-gate-baseline.json` gates every run. A baseline is recorded by dispatching the workflow
with `record=true`, which runs ungated and uploads each engine's report; the RocksDB report is
committed as the baseline. Until a veld recording replaces it, the file holds upstream's numbers
from 2026-08-08 (`ec7abd2`), which veld has never reproduced.

The gate compares veld with veld: upstream's published numbers are hypotheses to test, never a
reason to accept or decline a port (`docs/upstream/PLAN.md`, 1.5). Agent sessions may not build
binaries locally (`CLAUDE.md`), so the gate runs in CI.

## Files

- `corpora/shodh-smoke.jsonl` — L1 smoke corpus (memories ingested by the runner).
- `smoke_cases.jsonl` — 30 hand-crafted query cases with graded relevance labels,
  spanning six categories (decision / code / temporal / entity / multi_hop / negation).
- `baseline.json` — frozen per-metric scores for the smoke suite.
- `corpora/locomo-gate.jsonl`, `locomo-gate-cases.jsonl`, `locomo-gate-baseline.json` — LOCOMO gate
  fixtures. Currently unreferenced; see above.

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

Note that `veld/CLAUDE.md` bars agent sessions from `cargo build`/`cargo run`, so binary-producing
paths are unavailable to them regardless; `cargo test --no-run` and `cargo check` are permitted.

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

`RunInputs::layer_modes` selects which subset of the retrieval pipeline runs. This is a **library
option**, not a CLI flag. An empty vec is treated as `[LayerMode::Full]`. Modes are **cumulative**:
each row adds one stage on top of the row above it. Report keys come from `LayerMode::report_key()`
in `src/memory/types.rs`.

| Report key    | Stages added                                                           |
| ------------- | ---------------------------------------------------------------------- |
| `vamana_only` | Layer 3 vector ANN only (cosine + tie-break).                          |
| `+spreading`  | + Layer 2 graph spreading activation (RRF over vector ⊕ graph).        |
| `+bm25`       | + Layer 4 BM25 leg of three-way RRF fusion.                            |
| `+rerank`     | + Layer 4.9 ontological re-rank (see naming caveat below).             |
| `+facts`      | + Layer 0.7 / 4.8 fact-source boost from consolidated knowledge.       |
| `full`        | + Layer 0.4/0.5/0.6 pre-filters, Layer 4.6 interference, Layer 4.7     |
|               | prospective signal, full Layer 5 unified scoring (recency × importance |
|               | × arousal × credibility × tags × feedback × quality), retrieval        |
|               | competition, Hebbian coactivation, hierarchy expansion.                |

Pass `LayerMode::ALL` to run every mode in one invocation; the report's `layers` map gains one entry
per mode.

### Caveats — read these before staring at the numbers

1. **`+rerank` is a misnomer in this codebase.** The originating spec describes a cross-encoder
   rerank stage. There is no cross-encoder here. The mode wraps the **ontological re-ranker** at
   Layer 4.9 (multiplicative boost when episode entity types match the query's expected ontology
   labels). The label is preserved for spec fidelity; if a cross-encoder ever lands, it joins this
   same gate.

2. **Modes below `full` skip Layer 5 unified scoring.** Per-layer ndcg numbers will look strictly
   *lower* than `full` for reasons that are *not* "this stage didn't help" — they include the
   absence of recency, importance, arousal, credibility, feedback, quality-gate, and Hebbian
   multipliers. Read the table as **deltas between adjacent rows**, not as standalone absolutes.

3. **Cumulative-only by design.** You cannot ask for "BM25 without spreading" or "rerank without
   facts" — only the six cumulative modes exist. A non-cumulative ablation is a different feature.

4. **Only the LoCoMo gate is gated.** `recall.yml` gates the `locomo-gate` suite at `--layer
   full`; the smoke baseline and the per-layer numbers are advisory.

5. **30 cases is a small sample.** A single document rank flip moves per-category recall by ~0.20
   and per-mode ndcg by ~0.05. A `+0.01` per-layer delta is noise; trust direction, not magnitude.

## Regeneration history

| Date       | SHA       | Embedder       | Notes                          |
| ---------- | --------- | -------------- | ------------------------------ |
| 2026-05-03 | `6756665` | minilm-l6-v2   | Initial capture (RH-6, #268).  |
| 2026-08-08 | `ec7abd2` | minilm-l6-v2   | LOCOMO gate baseline; producer not in repo. |
