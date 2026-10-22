//! End-to-end runner for the L1 smoke suite.
//!
//! Loads fixtures via [`crate::recall_harness::fixtures`], stands up a
//! `MemorySystem` against an isolated storage directory, ingests the corpus,
//! runs every smoke case through `MemorySystem::recall`, and emits a
//! [`Report`] aggregating per-case [`Metrics`] into per-layer and
//! per-category sections.
//!
//! See issue #266 for the runner's CLI and acceptance criteria.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use uuid::Uuid;

use crate::config::ServerConfig;
use crate::handlers::remember::{assemble_tags, AssembledTags, TagInputs};
use crate::handlers::MultiUserMemoryManager;
use crate::memory::retrieval::RetrievalOutcome;
use crate::memory::types::MemoryId;
use crate::memory::types::{ExperienceType, LayerMode, NerEntityRecord};
use crate::memory::{Experience, MemorySystem, Query};

use super::fixtures::{
    self, CorpusItem, SmokeCase, SmokeCategory, SMOKE_CASES_PATH, SMOKE_CORPUS_PATH,
};
use super::metrics::{paired_bootstrap_ci, Metrics};
use super::report::{
    aggregate_category, aggregate_layer, median, AblationCaseRow, AblationReport, AblationRow,
    CategoryReport, DeltaCi, Failure, FunnelReport, FunnelStageRow, GraphStructure, KernelClass,
    LayerReport, LearningCurveArm, LearningCurveReport, LinkingReport, LinkingRow, PerCaseRecord,
    ReachabilityCategory, ReachabilityReport, Report, RerankSetting, RetrievalDials, SMOKE_K,
};

/// Embedder identifier emitted in the report. Matches the model wired into
/// `MemorySystem::new` today; bump when the embedder changes.
pub const EMBEDDER_ID: &str = "minilm-l6-v2";

/// Inputs to [`run_smoke_suite`].
#[derive(Debug, Clone)]
pub struct RunInputs {
    /// Storage directory for the harness's `MemorySystem`. The directory
    /// must be writeable; the runner does not delete it on completion so
    /// callers can inspect state after a failed run. With `repeats > 1`
    /// each repeat gets its own subdirectory `<storage_path>/repeat_<i>/`
    /// so storage state is independent across repeats — that is what gives
    /// the median across repeats meaning as a measure of cross-process
    /// noise, not just within-process noise.
    pub storage_path: PathBuf,
    /// Optional path overrides; if `None` the canonical fixture paths are used.
    pub corpus_path: Option<PathBuf>,
    pub cases_path: Option<PathBuf>,
    /// Suite identifier echoed into the report. Currently fixed to `"smoke"`.
    pub suite: String,
    /// Git SHA of the working tree the run was produced from. Caller is
    /// responsible for resolution because the runner does not shell out.
    pub git_sha: String,
    /// Number of independent ingest+query repeats. RH-12 (#272). Each repeat
    /// stands up its own isolated `MemorySystem`, ingests the corpus, and
    /// runs all cases. Per-case latency is the median across repeats; quality
    /// metrics must be byte-identical across repeats (any divergence is an
    /// infrastructure failure that fails the run).
    ///
    /// Default in the CLI is `5`. Setting `0` is treated as `1`.
    pub repeats: usize,

    /// Per-pipeline-layer attribution modes to run. RH-8 (#270). Each mode
    /// is a cumulative subset of the production retrieval pipeline; running
    /// the same query set under multiple modes lets us attribute quality
    /// deltas to specific stages rather than treating the pipeline as a
    /// single number. Within one repeat, ingest runs once and the case loop
    /// runs once per mode (cost ≈ ingest + N_modes × queries).
    ///
    /// CI gating still keys on `LayerMode::Full` only; the other modes are
    /// diagnostic. A single-element vec containing `Full` reproduces the
    /// pre-RH-8 behavior bit-for-bit.
    ///
    /// An empty vec is treated as `[LayerMode::Full]` so callers that don't
    /// care about per-layer attribution don't have to populate this.
    pub layer_modes: Vec<LayerMode>,

    /// Simulated edge age in days, applied AFTER ingest and BEFORE the query
    /// passes (RH decay study). When `> 0`, the harness ages the
    /// knowledge-graph edges via `MemorySystem::simulate_edge_aging` at the
    /// production ~6h cadence, so recall quality is measured as if the edges
    /// were `age_days` old. `0.0` (the default) means no aging — the
    /// pre-existing behavior, bit-for-bit. Run at 0 / 7 / 30 / 90 and diff the
    /// reports to see how edge decay erodes recall.
    pub age_days: f64,
}

/// One case's retrieved rank list, in score-descending order.
///
/// Used by the determinism gate (RH-11) to assert byte-identical
/// rank lists across consecutive runs of the same suite.
///
/// `retrieved` holds **corpus item IDs** (stable string handles from the
/// fixture), not the random UUIDs assigned by `MemorySystem::remember`.
/// UUIDs are freshly generated per ingest and therefore differ across
/// runs even when the rank order is byte-identical; comparing them would
/// guarantee a false negative. Items that were retrieved but do not
/// belong to the corpus (e.g. system-injected memories) are emitted as
/// `"<unknown:...>"` sentinels so divergence in those slots is still
/// observable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseRankList {
    pub case_id: String,
    pub retrieved: Vec<String>,
}

/// Wrapper around [`Report`] plus the per-case rank lists. Returned by
/// [`run_smoke_suite_with_ranks`]; the ranks are stripped by the public
/// [`run_smoke_suite`] wrapper to preserve the simpler caller contract.
#[derive(Debug, Clone)]
pub struct ReportWithRanks {
    pub report: Report,
    pub ranks: Vec<CaseRankList>,
    /// Per-case diagnostics keyed by layer `report_key` (e.g. `"full"`,
    /// `"vamana-only"`), each aligned with the fixture case order. With
    /// `--layer all` this carries every mode, so a missed item can be traced
    /// to the stage that dropped it (e.g. present in `vamana-only`, gone in
    /// `full` ⇒ a later layer demoted it). Side output only — never folded
    /// into `report`.
    pub per_case_by_layer: BTreeMap<String, Vec<PerCaseRecord>>,
}

/// Run the smoke suite end-to-end and return a populated [`Report`].
///
/// On infrastructure errors (corpus parse failure, system init failure,
/// remember failure) the call returns `Err`. Per-case recall errors are
/// recorded as `Failure { kind = "case", .. }` entries so a single broken
/// query does not abort the whole run.
pub fn run_smoke_suite(inputs: &RunInputs) -> Result<Report> {
    run_smoke_suite_with_ranks(inputs).map(|rwr| rwr.report)
}

/// Same as [`run_smoke_suite`] but also returns the per-case rank lists.
///
/// Exists for the RH-11 determinism gate: two consecutive calls with
/// identical inputs must return byte-identical [`CaseRankList`]s. If they
/// don't, there is a non-deterministic source somewhere in the retrieval
/// pipeline that the tie-break + thread pinning failed to cover.
pub fn run_smoke_suite_with_ranks(inputs: &RunInputs) -> Result<ReportWithRanks> {
    // ------------------------------------------------------------------
    // Determinism: pin every parallel runtime to a single thread before
    // any work touches ONNX or rayon. Multi-threaded float reductions
    // accumulate in non-deterministic order, which flips ranks on the
    // recall harness even when no source code has changed.
    //
    // - SHODH_ONNX_THREADS=1 → MiniLM/NER intra-op runs single-threaded
    //   (already plumbed through src/embeddings/{minilm,ner}.rs).
    // - RAYON_NUM_THREADS=1  → any par_iter() in scoring runs serially.
    //
    // We only set these vars if the caller hasn't pinned them already,
    // so production callers (which never invoke the harness) stay
    // unaffected.
    // ------------------------------------------------------------------
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));

    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading smoke corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading smoke cases from {}", cases_path.display()))?;

    // Only the smoke suite is balance-gated (fixed 108 / even categories). Other
    // suites (e.g. LoCoMo) just need structural integrity: unique ids, resolvable
    // evidence. Picking by suite lets a second corpus reuse this runner.
    if inputs.suite == "smoke" {
        fixtures::validate_smoke_suite(&corpus, &cases)
            .context("smoke suite failed structural validation — fix the JSONL fixtures")?;
    } else {
        fixtures::validate_structure(&corpus, &cases)
            .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;
    }

    // Optional deterministic case subsample for fast directional reads
    // (SHODH_MAX_CASES=N). The CORPUS is always ingested in full so every gold
    // item still resolves; only the QUERY set is strided down to ~N cases evenly
    // across the file ordering (which groups by conversation/category), so the
    // category mix is preserved. Unset / 0 / >= len → no subsampling. This is a
    // diagnostic-speed knob, never used by the gated CI baseline.
    let cases = match std::env::var("SHODH_MAX_CASES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n < cases.len())
    {
        Some(n) => {
            let total = cases.len();
            let stride = (total / n).max(1);
            let sampled: Vec<_> = cases.into_iter().step_by(stride).take(n).collect();
            tracing::info!(
                "SHODH_MAX_CASES={n}: subsampled {} of {} cases (stride {})",
                sampled.len(),
                total,
                stride
            );
            sampled
        }
        None => cases,
    };

    // Optional deterministic CORPUS subsample for MAX-speed directional reads
    // (SHODH_MAX_CORPUS=N). Unlike SHODH_MAX_CASES — which keeps the full corpus
    // so each query still faces all distractors and recall stays comparable to
    // full scale — this SHRINKS the distractor pool. Fewer distractors inflate
    // recall and compress the gaps between candidates, so it is a directional
    // speed knob ONLY (never the gated baseline); a winner picked here must be
    // confirmed at full corpus scale. Safety invariant: every gold item
    // referenced by the (already-subsampled) surviving cases is ALWAYS retained,
    // so no case is silently zeroed — only non-gold distractors are strided down
    // to fill the remaining budget. Original corpus order is preserved so
    // conversation grouping (which drives graph co-occurrence edges) is intact.
    let corpus = match std::env::var("SHODH_MAX_CORPUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n < corpus.len())
    {
        Some(n) => {
            let gold: std::collections::HashSet<&str> = cases
                .iter()
                .flat_map(|c| c.relevant.iter().map(|r| r.corpus_item_id.as_str()))
                .collect();
            let total = corpus.len();
            let gold_count = corpus
                .iter()
                .filter(|it| gold.contains(it.id.as_str()))
                .count();
            let keep_distractors = n.saturating_sub(gold_count);
            let distractor_total = total - gold_count;
            let stride = if keep_distractors == 0 {
                usize::MAX
            } else {
                (distractor_total / keep_distractors).max(1)
            };
            let mut seen_distractor = 0usize;
            let mut kept_distractors = 0usize;
            let reduced: Vec<_> = corpus
                .into_iter()
                .filter(|it| {
                    if gold.contains(it.id.as_str()) {
                        return true;
                    }
                    let take = kept_distractors < keep_distractors && seen_distractor % stride == 0;
                    seen_distractor += 1;
                    if take {
                        kept_distractors += 1;
                    }
                    take
                })
                .collect();
            tracing::info!(
                "SHODH_MAX_CORPUS={n}: reduced corpus {} → {} ({} gold kept + {} distractors). \
                 Directional only — recall inflates vs full scale.",
                total,
                reduced.len(),
                gold_count,
                kept_distractors
            );
            reduced
        }
        None => corpus,
    };

    let repeats = inputs.repeats.max(1);
    // RH-8 (#270): default to `[Full]` when caller passed an empty vec so
    // the existing single-mode contract is preserved by construction.
    let layer_modes: Vec<LayerMode> = if inputs.layer_modes.is_empty() {
        vec![LayerMode::Full]
    } else {
        let mut m = inputs.layer_modes.clone();
        m.sort();
        m.dedup();
        m
    };

    // Run N independent ingest+query passes against fresh storage
    // directories. RH-12 (#272): per-case latency is the median across
    // these passes; rank lists are required to be byte-identical across
    // every pass. The for-loop is intentionally serial — running passes
    // in parallel would defeat the determinism check, since concurrent
    // RocksDB instances under the same root would race on file handles.
    //
    // RH-8: within each pass, ingest happens once and the case loop runs
    // once per `LayerMode`. Determinism check is per-mode.
    let mut passes: Vec<OnePassResult> = Vec::with_capacity(repeats);
    // Model tape coverage: inputs the models were first asked for in each repeat.
    let mut tape_growth: Vec<(usize, usize)> = Vec::new();
    for i in 0..repeats {
        let tape_before = crate::embeddings::model_tape::entry_count();
        let pass_storage = if repeats == 1 {
            // Preserve the historical layout for single-repeat runs so
            // existing CI artifacts and the integration test stay valid.
            inputs.storage_path.clone()
        } else {
            inputs.storage_path.join(format!("repeat_{i}"))
        };
        let pass = run_one_pass(
            &pass_storage,
            &corpus,
            &cases,
            &layer_modes,
            inputs.age_days,
        )
        .with_context(|| format!("repeat {i} of {repeats}"))?;
        passes.push(pass);
        let added = crate::embeddings::model_tape::entry_count() - tape_before;
        if i > 0 && added > 0 {
            tape_growth.push((i, added));
        }
    }
    let tape_failures = finish_model_tape(
        &corpus,
        &tape_growth,
        &inputs.git_sha,
        passes[0].rerank.model.as_deref(),
    );

    // ------------------------------------------------------------------
    // Determinism check across repeats. Two passes against the same
    // fixtures, with thread pinning + the RH-10/11 tie-break in place,
    // MUST produce byte-identical rank lists per mode. If they don't,
    // something upstream regressed. Surface every diverging case as an
    // `infrastructure` failure so reviewers see the full picture.
    // ------------------------------------------------------------------
    let mut failures: Vec<Failure> = passes[0].failures.clone();
    failures.extend(tape_failures);
    for (i, pass) in passes.iter().enumerate().skip(1) {
        if pass.rerank != passes[0].rerank {
            failures.push(Failure {
                kind: "infrastructure".to_string(),
                detail: format!(
                    "repeat {i} reranked under {:?}, repeat 0 under {:?}",
                    pass.rerank, passes[0].rerank
                ),
            });
        }
        if pass.retrieval_dials != passes[0].retrieval_dials {
            failures.push(Failure {
                kind: "infrastructure".to_string(),
                detail: format!(
                    "repeat {i} ran under retrieval dials {:?}, repeat 0 under {:?}",
                    pass.retrieval_dials, passes[0].retrieval_dials
                ),
            });
        }
    }
    if repeats > 1 {
        for mode in &layer_modes {
            let ref_pass = passes[0]
                .per_mode
                .get(mode)
                .expect("repeat 0 must have all modes");
            for (i, pass) in passes.iter().enumerate().skip(1) {
                let cur_pass = pass
                    .per_mode
                    .get(mode)
                    .expect("every repeat must have every mode");
                for (k, ref_rank) in ref_pass.ranks.iter().enumerate() {
                    let cur_rank = &cur_pass.ranks[k];
                    debug_assert_eq!(ref_rank.case_id, cur_rank.case_id);
                    if ref_rank.retrieved != cur_rank.retrieved {
                        failures.push(Failure {
                            kind: "infrastructure".to_string(),
                            detail: format!(
                                "non-determinism [mode={}]: case {} rank list diverged between repeat 0 and repeat {i} \
                                 — repeat 0 = {:?}, repeat {i} = {:?}",
                                mode.report_key(), ref_rank.case_id, ref_rank.retrieved, cur_rank.retrieved
                            ),
                        });
                    }
                }
            }
        }
    }

    // Quality metrics: per-mode aggregation. Take per-case metrics from
    // repeat 0; per-case latency is the median across repeats.
    let mut layers: BTreeMap<String, LayerReport> = BTreeMap::new();
    for mode in &layer_modes {
        let per_case = &passes[0]
            .per_mode
            .get(mode)
            .expect("repeat 0 must have mode")
            .per_case;
        let mut latencies_median_ms: Vec<f64> = Vec::with_capacity(cases.len());
        for k in 0..cases.len() {
            let samples: Vec<f64> = passes
                .iter()
                .map(|p| {
                    p.per_mode
                        .get(mode)
                        .expect("every repeat must have every mode")
                        .latencies_ms[k]
                })
                .collect();
            latencies_median_ms.push(median(&samples));
        }
        layers.insert(
            mode.report_key().to_string(),
            aggregate_layer(per_case, &latencies_median_ms),
        );
    }

    // Per-category breakdown is reported for the highest mode present
    // (typically `Full`). Lower modes' per-category numbers are encoded
    // implicitly via the per-layer ndcg/recall tables; surfacing six
    // category-mode crosstabs would explode the report without adding
    // signal at the current corpus size.
    let highest_mode = layer_modes.last().copied().unwrap_or(LayerMode::Full);
    let by_category_cases = &passes[0]
        .per_mode
        .get(&highest_mode)
        .expect("repeat 0 must have highest mode")
        .by_category_cases;
    // Report whatever categories the suite actually contains (the smoke suite
    // has its six; LoCoMo has single_hop/open_domain/multi_hop/temporal), so a
    // second suite does not need its categories hard-coded into `ALL`.
    let mut by_category: BTreeMap<String, CategoryReport> = BTreeMap::new();
    for (cat, cases_for_cat) in by_category_cases {
        by_category.insert(
            category_name(*cat).to_string(),
            aggregate_category(cases_for_cat),
        );
    }

    // The embedder that ACTUALLY produced this run's vectors, read from the process-wide
    // record the memory system writes when it chooses its primary. This used to be the
    // constant EMBEDDER_ID, so a run under VELD_EMBEDDER=http (a binary, not cfg(test))
    // would have labelled nomic vectors "minilm-l6-v2" and compare_to_baseline's embedder
    // guard would have had nothing true to read. When no record exists the report says
    // EMBEDDER_UNKNOWN — not the constant, which would be the same defect with a narrower
    // entrance: a plausible label the guard accepts. Unknown is a value the guard refuses.
    let embedder = crate::embeddings::primary_embedder_info()
        .map(|info| info.model)
        .unwrap_or_else(|| crate::recall_harness::report::EMBEDDER_UNKNOWN.to_string());

    let report = Report {
        suite: inputs.suite.clone(),
        embedder,
        rerank: Some(passes[0].rerank.clone()),
        retrieval_dials: Some(passes[0].retrieval_dials.clone()),
        model_tape: crate::embeddings::model_tape::replay_digest(),
        kernel: KernelClass::detect(),
        git_sha: inputs.git_sha.clone(),
        timestamp: chrono::Utc::now(),
        layers,
        by_category,
        case_count: cases.len(),
        repeats,
        failures,
    };

    // Per-case diagnostics for every mode, from the same repeat-0 pass the
    // aggregates use and aligned with `cases` by index. With `--layer all`
    // this lets a missed item be traced to the stage that dropped it.
    let per_case_by_layer: BTreeMap<String, Vec<PerCaseRecord>> = layer_modes
        .iter()
        .map(|mode| {
            let mp = passes[0]
                .per_mode
                .get(mode)
                .expect("repeat 0 must have mode");
            (
                mode.report_key().to_string(),
                build_per_case_records(&cases, &mp.per_case, &mp.ranks, &mp.deep_ranks),
            )
        })
        .collect();

    // The public ranks output is the repeat-0 rank list for the highest
    // mode — that matches existing RH-11 test expectations (which assume
    // a single mode). Cross-repeat divergence is already surfaced via
    // `failures` above on a per-mode basis.
    let ranks = passes
        .into_iter()
        .next()
        .expect("repeats >= 1")
        .per_mode
        .remove(&highest_mode)
        .expect("repeat 0 must have highest mode")
        .ranks;

    Ok(ReportWithRanks {
        report,
        ranks,
        per_case_by_layer,
    })
}

/// Build per-case diagnostics from one mode's aligned outputs.
///
/// `metrics[i]`, `ranks[i]`, `deep_ranks[i]`, and `cases[i]` must describe the
/// same query (the runner guarantees this by pushing them in lockstep over
/// `cases`). `missed` is the set of relevant `corpus_item_id`s absent from the
/// top-`k` PRODUCTION list, which is what makes a weak case actionable; the
/// `recall_at_50/100` fields read the `RECALL_DIAG_K`-deep list, which is a
/// separate query at a deeper operating point (see `run_one_pass`).
fn build_per_case_records(
    cases: &[SmokeCase],
    metrics: &[Metrics],
    ranks: &[CaseRankList],
    deep_ranks: &[CaseRankList],
) -> Vec<PerCaseRecord> {
    cases
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let m = &metrics[i];
            let topk: HashSet<&str> = ranks[i]
                .retrieved
                .iter()
                .take(SMOKE_K)
                .map(|s| s.as_str())
                .collect();
            let missed: Vec<String> = case
                .relevant
                .iter()
                .map(|r| r.corpus_item_id.clone())
                .filter(|id| !topk.contains(id.as_str()))
                .collect();
            let relevant_total = case.relevant.len();
            // Recall over wider cutoffs of the DEEP list (RECALL_DIAG_K).
            // These split "gold ranked >10 at depth diag_k" from "gold never
            // retrieved at any depth". Measured at the diag_k operating point,
            // not the production one — the headline fields above stay on the
            // unperturbed production list.
            let gold: HashSet<&str> = case
                .relevant
                .iter()
                .map(|r| r.corpus_item_id.as_str())
                .collect();
            let recall_at = |k: usize| -> f64 {
                if gold.is_empty() {
                    return 0.0;
                }
                let topn: HashSet<&str> = deep_ranks[i]
                    .retrieved
                    .iter()
                    .take(k)
                    .map(|s| s.as_str())
                    .collect();
                let hit = gold.iter().filter(|g| topn.contains(*g)).count();
                hit as f64 / gold.len() as f64
            };
            PerCaseRecord {
                case_id: case.id.clone(),
                category: category_name(case.category).to_string(),
                query: case.query.clone(),
                ndcg_at_k: m.ndcg_at_k,
                recall_at_k: m.recall_at_k,
                mrr: m.mrr,
                p_at_1: m.p_at_1,
                relevant_total,
                relevant_found: relevant_total - missed.len(),
                missed,
                recall_at_50: recall_at(50),
                recall_at_100: recall_at(100),
            }
        })
        .collect()
}

/// Per-case results for one `LayerMode` within a single ingest+query pass.
struct ModePassResult {
    per_case: Vec<Metrics>,
    by_category_cases: HashMap<SmokeCategory, Vec<Metrics>>,
    latencies_ms: Vec<f64>,
    ranks: Vec<CaseRankList>,
    /// `RECALL_DIAG_K`-deep rank lists, aligned with `ranks`. Sourced from a
    /// SECOND recall per case at `max_results = diag_k` so the production
    /// query (which feeds `ranks`, headline metrics, latency, and the RH-11
    /// determinism gate) is never perturbed by the diagnostic. When the
    /// diagnostic is off this is a copy of `ranks`. Diagnostic side channel
    /// only — never folded into gated aggregates or the determinism check.
    deep_ranks: Vec<CaseRankList>,
}

/// Output of one ingest pass plus N mode-keyed query passes over the
/// fixture suite. Ingest happens once per pass; the case loop runs once
/// per `LayerMode` so per-mode quality and latency can be attributed
/// without paying the ingest cost N times. RH-8 (#270), RH-12 (#272).
struct OnePassResult {
    per_mode: BTreeMap<LayerMode, ModePassResult>,
    failures: Vec<Failure>,
    rerank: RerankSetting,
    retrieval_dials: RetrievalDials,
}

/// Run a single ingest pass, then run the case loop once per `LayerMode`.
///
/// The caller is responsible for choosing the storage dir so multiple
/// repeats in the same harness invocation get independent state. The
/// corpus and cases are passed in by reference so we don't reload fixtures
/// per pass. The system is built once and reused across modes — the modes
/// themselves are pure read-side gates and do not mutate persisted state
/// in lower modes (Hebbian/access-count/competition writes are gated to
/// `Full`), so cross-mode contamination is impossible.
fn run_one_pass(
    storage_path: &Path,
    corpus: &[CorpusItem],
    cases: &[SmokeCase],
    layer_modes: &[LayerMode],
    age_days: f64,
) -> Result<OnePassResult> {
    // Ingest through the production manager so the graph/lineage/ontology layer
    // is actually built, then query the per-user `MemorySystem` (which now has a
    // populated graph wired in). The manager is kept alive for the whole pass.
    let manager = build_manager(storage_path)?;
    let cross_encoder = guard_cross_encoder(&manager)?;

    // The NER-backend gate now lives in `ingest_corpus`, so every measurement
    // path is covered rather than only the two that remembered to ask.
    let id_map = ingest_corpus(&manager, corpus)?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // Typed-fraction scoreboard: the substrate's creation metric, printed once
    // per pass at the ingest→query boundary. generic = CoOccurs/RelatedTo/
    // CoRetrieved (walk-invisible bridges); everything else is typed signal.
    // Greppable in CI logs as EDGE_TYPE_DISTRIBUTION.
    if let Ok(graph) = manager.get_user_graph(EVAL_USER) {
        if let Ok(dist) = graph.read().relation_type_distribution() {
            let total: usize = dist.iter().map(|(_, c)| c).sum();
            if total > 0 {
                let generic: usize = dist
                    .iter()
                    .filter(|(t, _)| t == "CoOccurs" || t == "RelatedTo" || t == "CoRetrieved")
                    .map(|(_, c)| c)
                    .sum();
                let breakdown = dist
                    .iter()
                    .map(|(t, c)| format!("{t}={c}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!(
                    "EDGE_TYPE_DISTRIBUTION total={total} typed_pct={:.1} {breakdown}",
                    100.0 * (total - generic) as f64 / total as f64
                );
            }
        }
    }

    // SHODH_VAMANA_QUALITY_REBUILD=1 — A/B lever: bulk alpha-RNG rebuild at the
    // ingest→query boundary. The harness is the only place that KNOWS this
    // boundary; a search-time trigger fires on remember()'s dedup search before
    // any insert and silently no-ops (caught by run 27255269454's gate). Vector
    // composition and the id mapping are preserved — only graph topology changes,
    // so the arm isolates greedy-insert vs alpha-RNG construction.
    if std::env::var("SHODH_VAMANA_QUALITY_REBUILD")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        system
            .read()
            .force_vector_quality_rebuild()
            .context("vamana quality rebuild at the ingest→query boundary")?;
    }

    // RH decay study: optionally age the knowledge-graph edges before querying
    // so recall quality reflects decayed/pruned edges. Driven at the production
    // ~6h cadence (see `MemorySystem::simulate_edge_aging`); a single large jump
    // would skip into the power-law decay phase and misrepresent the curve.
    // `age_days <= 0` is a no-op, preserving pre-existing behavior bit-for-bit.
    if age_days > 0.0 {
        system
            .read()
            .simulate_edge_aging(age_days, super::decay_sim::PRODUCTION_CADENCE_HOURS)
            .context("aging knowledge-graph edges for the decay study")?;
    }
    // Reverse map: random per-run UUIDs → stable corpus item IDs. The
    // determinism gate (RH-11) compares rank lists across runs, so we MUST
    // emit stable handles. Comparing UUIDs would always diverge because
    // each ingest assigns a fresh `Uuid::new_v4()`.
    let uuid_to_corpus_id: HashMap<Uuid, String> = id_map
        .iter()
        .map(|(corpus_id, uuid)| (*uuid, corpus_id.clone()))
        .collect();

    // Stage fingerprints (`VELD_STAGE_EXPORT=<jsonl path>`, Full mode only).
    //
    // The repeat check compares stages between passes of ONE process. This writes the same
    // stages, plus the ingest-side ones it cannot see (embedding bytes, NER spans, the graph's
    // node and edge sets), so two processes can be diffed stage by stage with
    // scripts/stage_diff.py: two machines, two commits or two settings. Scores travel as f32
    // bit patterns, so a one-ULP move shows as a move.
    let stage_export: Option<PathBuf> = std::env::var_os("VELD_STAGE_EXPORT")
        .or_else(|| std::env::var_os("SHODH_STAGE_EXPORT"))
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let mut stage_lines: Vec<String> = Vec::new();
    if stage_export.is_some() {
        stage_lines.extend(ingest_fingerprints(
            &manager,
            &system.read(),
            corpus,
            &id_map,
        )?);
    }

    let mut per_mode: BTreeMap<LayerMode, ModePassResult> = BTreeMap::new();
    let mut failures: Vec<Failure> = Vec::new();

    // Diagnostic cutoff: when RECALL_DIAG_K is set, a SECOND recall per case
    // fetches a `diag_k`-deep list so the per-case recall@50/@100 fields are
    // meaningful. The deep fetch is a separate query on purpose: `max_results`
    // is an OPERATING POINT, not a view — it sizes the vector candidate pool
    // (`max_results * 3`, semantic_retrieve_inner Layer 3) and the rerank
    // budget (`max_results * 2`, Layer 4.9) — so the previous design, which
    // folded `diag_k` into the one production query, silently changed the
    // headline numbers whenever the diagnostic was on (measured on the
    // LoCoMo-100 gate: open_domain recall@10 0.275 → 0.150 under
    // RECALL_DIAG_K=100). Headline metrics, rank lists (RH-11 determinism),
    // latency, and the fusion-feature export all come exclusively from the
    // production-depth query below. Defaults to SMOKE_K (no deep query, no
    // behavior change). Parse once per pass.
    let diag_k = std::env::var("RECALL_DIAG_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|k| *k >= SMOKE_K)
        .unwrap_or(SMOKE_K);

    // Approach C stage 2: per-query fusion-feature export for the offline trust
    // fit (SHODH_FUSION_FEATURE_EXPORT=<jsonl path>). Full mode only — that is
    // the production path the fit targets. Lines accumulate across modes/repeats
    // and append at pass end.
    let feature_export: Option<std::path::PathBuf> = std::env::var("SHODH_FUSION_FEATURE_EXPORT")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from);
    let mut feature_lines: Vec<String> = Vec::new();

    // Candidate-pool export (`SHODH_POOL_EXPORT=<jsonl path>`, Full mode only).
    //
    // Writes, per case: the query, the gold ids, and the RECALL_DIAG_K-deep
    // candidate pool in rank order. That is everything an OFFLINE rescoring
    // experiment needs -- a cross-encoder, a late-interaction MaxSim scorer, a
    // fine-tuned or JEPA-style adapter -- because the pool is exactly the set a
    // reranker would see and the gold ids are the labels. Candidate TEXT is
    // deliberately not duplicated here: the corpus jsonl already holds it keyed
    // by id, so the consumer joins rather than the harness bloating every line.
    //
    // The point is to convert "would X improve ranking?" from a 45-minute CI run
    // into a script. ~32pp of gold sits in ranks 11-100, so every rescoring idea
    // is testable against this file without touching the Rust pipeline at all.
    let pool_export: Option<std::path::PathBuf> = std::env::var("SHODH_POOL_EXPORT")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from);
    let mut pool_lines: Vec<String> = Vec::new();

    for mode in layer_modes {
        let mut per_case = Vec::with_capacity(cases.len());
        let mut latencies_ms = Vec::with_capacity(cases.len());
        let mut by_category_cases: HashMap<SmokeCategory, Vec<Metrics>> = HashMap::new();
        let mut ranks: Vec<CaseRankList> = Vec::with_capacity(cases.len());
        let mut deep_ranks: Vec<CaseRankList> = Vec::with_capacity(cases.len());

        for case in cases {
            let mut query = Query {
                query_text: Some(case.query.clone()),
                max_results: SMOKE_K,
                layers: Some(*mode),
                ..Default::default()
            };
            // SHODH_QUERY_NER A/B lever: neural-NER graph seeding (no-op when unset).
            manager.annotate_query_ner(&mut query);

            // Pure function of fixtures + ingest (hoisted above recall so the
            // feature exporter can arm with this query's gold ids).
            let relevance = build_relevance_map(case, &id_map);
            if feature_export.is_some() && matches!(*mode, LayerMode::Full) {
                crate::memory::fusion_features::begin(
                    relevance
                        .keys()
                        .map(|u| crate::memory::types::MemoryId(*u))
                        .collect(),
                );
            }

            let started = Instant::now();
            let result = system.read().recall(&query);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            latencies_ms.push(elapsed_ms);

            let memories = match result {
                Ok(m) => m,
                Err(e) => {
                    failures.push(Failure {
                        kind: "case".to_string(),
                        detail: format!(
                            "recall failed for {} [mode={}]: {e:#}",
                            case.id,
                            mode.report_key()
                        ),
                    });
                    // Treat as zero-recall so aggregates keep their denominator.
                    Vec::new()
                }
            };

            let retrieved_uuids: Vec<Uuid> = memories.iter().map(|m| m.id.0).collect();
            let retrieved_corpus_ids: Vec<String> = retrieved_uuids
                .iter()
                .map(|u| {
                    uuid_to_corpus_id
                        .get(u)
                        .cloned()
                        .unwrap_or_else(|| format!("<unknown:{u}>"))
                })
                .collect();
            ranks.push(CaseRankList {
                case_id: case.id.clone(),
                retrieved: retrieved_corpus_ids,
            });

            // Drain the per-query fusion features (None unless armed above AND
            // the FLAT fusion path ran) and pair them with the case outcome.
            if let Some(features) = crate::memory::fusion_features::take() {
                let gold_final_rank = retrieved_uuids
                    .iter()
                    .position(|u| relevance.contains_key(u));
                feature_lines.push(
                    serde_json::json!({
                        "case_id": case.id,
                        "category": format!("{:?}", case.category),
                        "query_len": case.query.len(),
                        "gold_final_rank": gold_final_rank,
                        "features": features,
                    })
                    .to_string(),
                );
            }

            // Only emit the "missing relevance map" failure once across modes
            // — the relevance map is a function of fixtures + ingest, not
            // the mode, so duplicating it would clutter the failure list.
            if relevance.is_empty() && *mode == layer_modes[0] {
                failures.push(Failure {
                    kind: "case".to_string(),
                    detail: format!(
                        "case {}: every relevant corpus item was missing from id_map (ingest skipped them)",
                        case.id
                    ),
                });
            }

            let metrics = Metrics::compute(&retrieved_uuids, &relevance, SMOKE_K);
            by_category_cases
                .entry(case.category)
                .or_default()
                .push(metrics);
            per_case.push(metrics);
        }

        // Diagnostic recalls run after the last timed recall, never between two of them, so
        // whatever they leave behind in caches or the allocator cannot shape the gated
        // latency. They are read-only and aligned with the rank lists by index.
        for (k, case) in cases.iter().enumerate() {
            let relevance = build_relevance_map(case, &id_map);
            // Depth diagnostics: a second, production-shaped recall at
            // `max_results = diag_k` feeds ONLY the per-case recall@50/@100
            // fields (see the diag_k comment above for why it must not be the
            // same query). The two lists are different operating points by
            // construction — the deeper vector pool admits new competitors, so
            // a gold item's deep rank can even sit below its production rank;
            // that is inherent to fetching a deeper list, not an
            // inconsistency. The deep query is side-effect-free because the
            // harness pins SHODH_RECALL_READONLY=1 (pin_harness_threads), so
            // it cannot contaminate Hebbian/access state between cases.
            let deep_retrieved: Vec<String> = if diag_k > SMOKE_K {
                let mut deep_query = Query {
                    query_text: Some(case.query.clone()),
                    max_results: diag_k,
                    layers: Some(*mode),
                    ..Default::default()
                };
                manager.annotate_query_ner(&mut deep_query);
                match system.read().recall(&deep_query) {
                    Ok(deep) => deep
                        .iter()
                        .map(|m| {
                            uuid_to_corpus_id
                                .get(&m.id.0)
                                .cloned()
                                .unwrap_or_else(|| format!("<unknown:{}>", m.id.0))
                        })
                        .collect(),
                    Err(e) => {
                        failures.push(Failure {
                            kind: "case".to_string(),
                            detail: format!(
                                "diagnostic deep recall (RECALL_DIAG_K={diag_k}) failed for {} [mode={}]: {e:#}",
                                case.id,
                                mode.report_key()
                            ),
                        });
                        Vec::new()
                    }
                }
            } else {
                // Diagnostic off: the depth fields degrade to the production
                // list, exactly as before the two-query split.
                ranks[k].retrieved.clone()
            };
            if pool_export.is_some() && matches!(*mode, LayerMode::Full) {
                // Corpus ids, matching `pool` -- `relevance` is keyed by Uuid while
                // the rank lists are already corpus ids, so exporting the raw Uuid
                // made the two sets disjoint and every offline join scored zero.
                let mut gold: Vec<String> = relevance
                    .keys()
                    .map(|u| {
                        uuid_to_corpus_id
                            .get(u)
                            .cloned()
                            .unwrap_or_else(|| u.to_string())
                    })
                    .collect();
                gold.sort();
                pool_lines.push(
                    serde_json::json!({
                        "case_id": case.id,
                        "category": category_name(case.category),
                        "query": case.query,
                        "gold": gold,
                        "pool": deep_retrieved,
                    })
                    .to_string(),
                );
            }
            if stage_export.is_some() && matches!(*mode, LayerMode::Full) {
                // One more production-shaped recall with every stage captured, the
                // cross-encoder's scores over its pool (`ce`) among them.
                let mut query = Query {
                    query_text: Some(case.query.clone()),
                    max_results: SMOKE_K,
                    layers: Some(*mode),
                    ..Default::default()
                };
                manager.annotate_query_ner(&mut query);
                crate::memory::gold_funnel::begin_capture();
                let captured = system.read().recall(&query);
                let stages = crate::memory::gold_funnel::take_capture().unwrap_or_default();
                let corpus_id = |id: &MemoryId| {
                    uuid_to_corpus_id
                        .get(&id.0)
                        .cloned()
                        .unwrap_or_else(|| format!("<unknown:{}>", id.0))
                };
                let final_captured: Vec<String> = match captured {
                    Ok(memories) => memories.iter().map(|m| corpus_id(&m.id)).collect(),
                    Err(e) => {
                        failures.push(Failure {
                            kind: "case".to_string(),
                            detail: format!(
                                "stage-export recall failed for {} [mode={}]: {e:#}",
                                case.id,
                                mode.report_key()
                            ),
                        });
                        Vec::new()
                    }
                };
                let stages_json: Vec<serde_json::Value> = stages
                    .iter()
                    .map(|(stage, entries)| {
                        let items: Vec<(String, Option<String>)> = entries
                            .iter()
                            .map(|(id, score)| {
                                (corpus_id(id), score.map(|v| format!("{:08x}", v.to_bits())))
                            })
                            .collect();
                        serde_json::json!({ "stage": stage, "items": items })
                    })
                    .collect();
                stage_lines.push(
                    serde_json::json!({
                        "kind": "case",
                        "case_id": case.id,
                        "ranks": ranks[k].retrieved,
                        "deep": deep_retrieved,
                        "final_captured": final_captured,
                        "stages": stages_json,
                    })
                    .to_string(),
                );
            }
            deep_ranks.push(CaseRankList {
                case_id: case.id.clone(),
                retrieved: deep_retrieved,
            });
        }

        per_mode.insert(
            *mode,
            ModePassResult {
                per_case,
                by_category_cases,
                latencies_ms,
                ranks,
                deep_ranks,
            },
        );
    }

    if let Some(path) = &stage_export {
        if !stage_lines.is_empty() {
            use std::io::Write as _;
            // Truncated, as the pool export is: every repeat runs this pass, the repeat check
            // requires the passes to agree, and the last pass's records stand for the run.
            let mut f = std::fs::File::create(path)
                .with_context(|| format!("creating stage export {}", path.display()))?;
            for line in &stage_lines {
                writeln!(f, "{line}")?;
            }
            eprintln!(
                "  stage export: {} records -> {}",
                stage_lines.len(),
                path.display()
            );
        }
    }

    if let Some(path) = &pool_export {
        if !pool_lines.is_empty() {
            use std::io::Write as _;
            let mut f = std::fs::File::create(path)
                .with_context(|| format!("creating pool export {}", path.display()))?;
            for line in &pool_lines {
                writeln!(f, "{line}")?;
            }
            eprintln!(
                "  pool export: {} cases -> {}",
                pool_lines.len(),
                path.display()
            );
        }
    }

    if let Some(path) = &feature_export {
        if !feature_lines.is_empty() {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("opening feature export {}", path.display()))?;
            for line in &feature_lines {
                writeln!(f, "{line}")?;
            }
        }
    }

    let rerank = RerankSetting::of(&system.read(), cross_encoder);
    let retrieval_dials = RetrievalDials::of(&system.read().cognitive_dials());
    Ok(OnePassResult {
        per_mode,
        failures,
        rerank,
        retrieval_dials,
    })
}

/// Ingest-side fingerprints for `VELD_STAGE_EXPORT`: one record per corpus item (its stored
/// embedding and its NER spans with confidence bits) and one for the graph (node and edge sets,
/// keyed by entity name because uuids are minted per process, with strength bits). Embeddings
/// are hashed over their raw little-endian bytes with the leading values' bit patterns beside
/// the hash, so a diff between two machines shows both whether and by how much a stage moved.
fn ingest_fingerprints(
    manager: &MultiUserMemoryManager,
    system: &MemorySystem,
    corpus: &[CorpusItem],
    id_map: &HashMap<String, Uuid>,
) -> Result<Vec<String>> {
    use sha2::{Digest, Sha256};
    let head_bits = |values: &[f32], n: usize| -> Vec<String> {
        values
            .iter()
            .take(n)
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect()
    };
    let sha_f32 = |values: &[f32]| -> String {
        let mut h = Sha256::new();
        for v in values {
            h.update(v.to_le_bytes());
        }
        hex::encode(h.finalize())
    };
    let ner = manager.get_neural_ner();
    let mut lines = Vec::with_capacity(corpus.len() + 1);
    let mut all_embeddings = Sha256::new();
    let mut all_ner = Sha256::new();
    for item in corpus {
        let Some(uuid) = id_map.get(&item.id) else {
            continue;
        };
        let embedding = system
            .get_memory(&MemoryId(*uuid))
            .with_context(|| format!("stage export: reading memory {}", item.id))?
            .experience
            .embeddings
            .unwrap_or_default();
        for v in &embedding {
            all_embeddings.update(v.to_le_bytes());
        }
        // The call ingest_corpus made for this item. The session is deterministic within a
        // process (the repeat check holds it to that), so these are the spans the graph was
        // built from.
        let spans: Vec<String> = ner
            .extract(&item.content)
            .unwrap_or_default()
            .iter()
            .map(|e| {
                format!(
                    "{}|{}|{:08x}|{}|{}",
                    e.text,
                    e.entity_type.as_str(),
                    e.confidence.to_bits(),
                    e.start,
                    e.end
                )
            })
            .collect();
        let ner_line = spans.join(";");
        all_ner.update(ner_line.as_bytes());
        lines.push(
            serde_json::json!({
                "kind": "ingest",
                "id": item.id,
                "embedding_dim": embedding.len(),
                "embedding_sha256": sha_f32(&embedding),
                "embedding_head": head_bits(&embedding, 4),
                "ner_sha256": hex::encode(Sha256::digest(ner_line.as_bytes())),
                "ner": spans,
            })
            .to_string(),
        );
    }

    let graph = manager.get_user_graph(EVAL_USER)?;
    let graph = graph.read();
    let entities = graph.get_all_entities()?;
    let names: HashMap<Uuid, String> = entities.iter().map(|e| (e.uuid, e.name.clone())).collect();
    let name_of = |uuid: &Uuid| names.get(uuid).cloned().unwrap_or_else(|| uuid.to_string());
    let mut nodes: Vec<String> = entities
        .iter()
        .map(|e| format!("{}|{:?}|{}", e.name, e.labels, e.mention_count))
        .collect();
    nodes.sort();
    let mut edges: Vec<String> = graph
        .get_all_relationships()?
        .iter()
        .map(|r| {
            format!(
                "{}|{}|{}|{:08x}",
                name_of(&r.from_entity),
                name_of(&r.to_entity),
                r.relation_type.as_str(),
                r.strength.to_bits()
            )
        })
        .collect();
    edges.sort();
    lines.push(
        serde_json::json!({
            "kind": "graph",
            "embeddings_sha256": hex::encode(all_embeddings.finalize()),
            "ner_sha256": hex::encode(all_ner.finalize()),
            "node_count": nodes.len(),
            "nodes_sha256": hex::encode(Sha256::digest(nodes.join("\n").as_bytes())),
            "edge_count": edges.len(),
            "edges_sha256": hex::encode(Sha256::digest(edges.join("\n").as_bytes())),
            "nodes": nodes,
            "edges": edges,
        })
        .to_string(),
    );
    Ok(lines)
}

/// Close out a model tape at the end of a suite run, returning what must fail the run.
///
/// Recording: every (query, corpus item) cross-encoder pair the run did not score is
/// scored now, so a later retrieval change that puts another candidate in the pool
/// still finds it on the tape; then the tape is written. Replaying: every model in the
/// tape's header must have been checked against the files this process loaded. Either
/// way, any miss, mismatch or failed model call becomes an `infrastructure` failure,
/// because the numbers did not come from the tape they claim to. `growth` lists the
/// repeats after the first that asked the models for new inputs.
fn finish_model_tape(
    corpus: &[CorpusItem],
    growth: &[(usize, usize)],
    git_sha: &str,
    rerank_model: Option<&str>,
) -> Vec<Failure> {
    use crate::embeddings::model_tape as tape;
    let mut failures = Vec::new();
    if !tape::active() {
        return failures;
    }
    let infra = |detail: String| Failure {
        kind: "infrastructure".to_string(),
        detail: format!("model tape: {detail}"),
    };
    for (repeat, added) in growth {
        failures.push(infra(format!(
            "repeat {repeat} asked the models for {added} inputs that no earlier repeat \
             did, so what a run asks for is not stable and one recording cannot cover it"
        )));
    }
    if tape::recording() {
        match fill_cross_encoder_table(corpus, rerank_model) {
            Ok(n) => eprintln!("  model tape: scored {n} cross-encoder pairs the run did not"),
            Err(e) => failures.push(infra(format!("{e:#}"))),
        }
        let conflicts = tape::conflicts();
        let provenance = serde_json::json!({
            "git_sha": git_sha,
            "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "conflicts": conflicts,
            "recorded_at": chrono::Utc::now().to_rfc3339(),
        });
        match tape::write_recording(provenance) {
            Ok((path, n)) => eprintln!(
                "  model tape: {n} entries -> {} (conflicts {conflicts:?})",
                path.display()
            ),
            Err(e) => failures.push(infra(format!("writing the tape: {e:#}"))),
        }
    } else {
        tape::require_verified_models();
    }
    let problems = tape::take_problems();
    const SHOWN: usize = 20;
    for p in problems.iter().take(SHOWN) {
        failures.push(infra(p.clone()));
    }
    if problems.len() > SHOWN {
        failures.push(infra(format!(
            "and {} more problems",
            problems.len() - SHOWN
        )));
    }
    failures
}

/// Score, through the tape, every (query, corpus item) pair it does not hold yet.
///
/// The queries are the ones the run sent to the reranker. The candidates are the
/// corpus contents, which must include every text the reranker saw apart from the
/// pairs that only load the model (the harness's probe and the background pre-warm):
/// if one is not a corpus item, candidates are built some other way and the corpus
/// cannot complete the table.
///
/// `model` is the registry id the run reranked with (its `RerankSetting`), so the table is
/// completed by the model the `rerank_model` dial selected and not by whatever the
/// environment names.
fn fill_cross_encoder_table(corpus: &[CorpusItem], model: Option<&str>) -> Result<usize> {
    use crate::embeddings::cross_encoder::WARMUP_TEXT;
    use crate::embeddings::model_tape as tape;
    use crate::memory::hybrid_search::CROSS_ENCODER_PROBE;
    let loads_only = |t: &str| t == CROSS_ENCODER_PROBE || t == WARMUP_TEXT;
    let ce = crate::embeddings::cross_encoder::load_cross_encoder(model)?;
    let mut seen = std::collections::HashSet::new();
    let docs: Vec<&str> = corpus
        .iter()
        .map(|c| c.content.as_str())
        .filter(|d| seen.insert(*d))
        .collect();
    let foreign: Vec<String> = tape::cross_encoder_docs()
        .into_iter()
        .filter(|d| !loads_only(d) && !seen.contains(d.as_str()))
        .collect();
    if let Some(first) = foreign.first() {
        anyhow::bail!(
            "{} candidate texts the reranker saw are not corpus contents (first: {:?}), \
             so the corpus cannot complete the cross-encoder table",
            foreign.len(),
            first.chars().take(80).collect::<String>()
        );
    }
    let mut scored = 0;
    for query in tape::cross_encoder_queries() {
        if loads_only(&query) {
            continue;
        }
        let missing: Vec<&str> = docs
            .iter()
            .copied()
            .filter(|d| !tape::has_cross_encoding(&query, d))
            .collect();
        if missing.is_empty() {
            continue;
        }
        ce.score_pairs(&query, &missing)
            .with_context(|| format!("scoring pairs for query {query:?}"))?;
        scored += missing.len();
    }
    Ok(scored)
}

/// Pin parallel runtimes to a single thread for the harness process.
///
/// Recall harness reproducibility requires that two runs of the same query
/// produce byte-identical rank lists. Multi-threaded float reductions
/// (RAYON par_iter) and ONNX intra-op parallelism both accumulate sums in
/// non-deterministic order, which is enough to flip ranks at the
/// fourth-decimal level.
///
/// This function only sets each variable when it is currently unset, so a
/// caller that explicitly chose a different value (e.g. for a benchmark)
/// keeps their override.
fn pin_harness_threads() {
    // SAFETY: env mutation is process-wide. The harness is the sole entry
    // point that calls this; the production server never invokes the
    // recall harness, so we are not racing other readers in any deployed
    // binary. The recall-eval CLI is single-threaded at startup.
    unsafe {
        if std::env::var_os("SHODH_ONNX_THREADS").is_none() {
            std::env::set_var("SHODH_ONNX_THREADS", "1");
        }
        if std::env::var_os("RAYON_NUM_THREADS").is_none() {
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }
        // Repeats measure variance, not learning curves: recall must not
        // mutate usage state mid-eval. FLAT fusion made graph magnitude
        // load-bearing, so first-repeat co-activation writes were shifting
        // later repeats' rankings (L1 smoke non-determinism). Learning-curve
        // experiments opt out by setting SHODH_RECALL_READONLY=0 explicitly.
        if std::env::var_os("SHODH_RECALL_READONLY").is_none() {
            std::env::set_var("SHODH_RECALL_READONLY", "1");
        }
        // Freeze the scoring clock. Repeat passes execute minutes apart; with
        // a live clock the recency component of every score drifts between
        // repeats, which is enough to flip near-tie adjacent ranks (smoke-094)
        // and trip the determinism gate on noise unrelated to the code under
        // test. The anchor is FIXED (not run-start) because corpus created_at
        // values are static fixtures: a live anchor would also erode recency
        // a little more every real day, silently rotting baseline.json until
        // some near-tie flips weeks after it was generated.
        if std::env::var_os("SHODH_EVAL_NOW").is_none() {
            std::env::set_var("SHODH_EVAL_NOW", HARNESS_CLOCK_ANCHOR);
        }
        // The cross-encoder is one instance per process, so its pair-score cache outlives a
        // repeat's MemorySystem and would serve repeats 2..n from memory. Off, each repeat's
        // latency is the model's; the scores are the same either way.
        if std::env::var_os("VELD_CROSS_ENCODER_CACHE").is_none() {
            std::env::set_var("VELD_CROSS_ENCODER_CACHE", "0");
        }
    }
}

/// The harness's frozen scoring clock (see `pin_harness_threads`). Must stay
/// later than every `created_at` in the eval corpora (smoke max 2025-09-19,
/// LoCoMo 2023, lineage fixtures 2024) so ages never go negative.
pub const HARNESS_CLOCK_ANCHOR: &str = "2026-06-12T00:00:00Z";

/// Construct an isolated `MemorySystem` rooted at `storage_path`.
/// Single tenant the harness ingests under.
pub const EVAL_USER: &str = "recall-eval";

/// Build the production `MultiUserMemoryManager` (NOT a bare `MemorySystem`).
///
/// FIDELITY (critical): a bare `MemorySystem` has `graph_memory = None` and
/// `remember()` deliberately does NOT build the entity graph — production builds
/// it in the HTTP handler via `process_experience_into_graph`. So the previous
/// harness measured the pipeline with the ENTIRE knowledge-graph / spreading-
/// activation / lineage / ontology layer disabled. The manager wires a per-user
/// graph + NER, so the eval exercises the same ingest path production does.
pub(crate) fn build_manager(storage_path: &Path) -> Result<MultiUserMemoryManager> {
    // Pin here, not at the entry points. The pinning was applied per analysis
    // fn and two paths never got it: bridge_harness::ingest_fresh and
    // forgetting_harness::analyze_selective_forgetting, both called straight
    // from the recall-eval dispatch. Unpinned they run with ONNX threads at
    // the PRODUCTION default of 24 on a 4-core runner, rayon on every core,
    // SHODH_RECALL_READONLY unset so recall MUTATES usage state mid-eval, and
    // a live scoring clock -- the last two documented in pin_harness_threads
    // itself as the causes of repeat non-determinism it exists to prevent.
    //
    // This is the chokepoint rather than ingest_corpus because the env has to
    // be set before MultiUserMemoryManager::new below, which initialises the
    // embedder and reads the thread counts. Every measurement path calls
    // build_manager, and it is crate-internal to the harness, so the
    // production server is unaffected. The per-entry-point calls that remain
    // are now redundant, and harmless: the function only sets what is unset.
    pin_harness_threads();
    std::fs::create_dir_all(storage_path)
        .with_context(|| format!("creating storage dir {}", storage_path.display()))?;
    MultiUserMemoryManager::new(storage_path.to_path_buf(), ServerConfig::default())
        .context("initialising MultiUserMemoryManager for recall harness")
}

/// Ingest the corpus through the production ingest path (NER → remember →
/// graph), returning the `string handle → Uuid` map used to translate
/// ground-truth references to system memory IDs.
///
/// Mirrors the HTTP `remember` handler: run NER, merge entity names, store the
/// memory, then build the entity graph from it. Without the
/// `process_experience_into_graph` step the graph stays empty and Layer 2
/// spreading activation is a no-op.
/// Refuse to measure on the rule-based fallback NER.
///
/// This lives at the INGEST chokepoint rather than at each entry point, and the
/// placement is the fix. The gate previously existed only inside `run_one_pass`
/// and `run_longmemeval`; the other seven measurement paths that build a corpus
/// -- `analyze_ablation`, `analyze_funnel`, `analyze_graph_reachability`,
/// `analyze_linking`, `run_learning_arm`, `ingest_fresh` and
/// `analyze_selective_forgetting` -- had none, so they would quietly measure a
/// different recogniser than the one production ships and report the result as
/// if it were the system.
///
/// That is not hypothetical. `.github/workflows/locomo-recall.yml` drives
/// `--ablation`, `--funnel`, `--forgetting` and `--learning` and fetches no
/// GLiNER model at all, while `recall.yml` fetches and verifies it. The
/// ablation matrix -- the instrument behind the graph lever verdicts -- was on
/// the fallback path, and nothing in its output said so.
///
/// Every path reaches ingest, so a guard here cannot be forgotten by a path
/// written later. `SHODH_ALLOW_FALLBACK_NER=1` remains the visible escape
/// hatch, and `cfg!(test)` is exempt because lib tests exercise the harness
/// MACHINERY on runners that may lack the model -- the gate protects
/// MEASUREMENTS, and the recall-eval binary is never `cfg(test)`.
pub(crate) fn guard_ner_backend(manager: &MultiUserMemoryManager) -> Result<()> {
    let backend = if manager.get_neural_ner().is_fallback_mode() {
        "fallback"
    } else {
        "neural"
    };
    eprintln!("NER_BACKEND={backend}");
    if backend == "fallback"
        && !cfg!(test)
        && !std::env::var("SHODH_ALLOW_FALLBACK_NER")
            .map(|v| v == "1")
            .unwrap_or(false)
    {
        anyhow::bail!(
            "recall harness refusing to run on the FALLBACK NER (model not              loaded) -- results would not measure the shipped system. Provide              the pinned model or set SHODH_ALLOW_FALLBACK_NER=1 to override              visibly."
        );
    }
    Ok(())
}

pub fn ingest_corpus(
    manager: &MultiUserMemoryManager,
    corpus: &[CorpusItem],
) -> Result<HashMap<String, Uuid>> {
    guard_ner_backend(manager)?;
    let mut map = HashMap::with_capacity(corpus.len());
    let ner = manager.get_neural_ner();
    let keyword_extractor = manager.get_keyword_extractor();
    let user_mem = manager.get_user_memory(EVAL_USER)?;
    // Eval-fidelity gate: production embeds entity names so Tier-4 embedding
    // concept-merge runs (cross-memory entity resolution). The eval passed None →
    // Tier-4 OFF → entities under-resolved vs prod (every reach number measured
    // without it). SHODH_EVAL_NAME_EMB=1 populates name embeddings, matching the
    // handler (remember.rs:740). Default off → byte-identical (None).
    let name_emb_on = std::env::var("SHODH_EVAL_NAME_EMB")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    for item in corpus {
        // Pass 1: NER (faithful to the handler's NerEntityRecord shape).
        let ner_entities: Vec<NerEntityRecord> = match ner.extract(&item.content) {
            Ok(entities) => entities
                .into_iter()
                .map(|e| NerEntityRecord {
                    text: e.text,
                    entity_type: e.entity_type.as_str().to_string(),
                    confidence: e.confidence,
                    start_char: Some(e.start),
                    end_char: Some(e.end),
                    // veld's NerEntity carries no fine label; the coarse
                    // entity_type above is the whole classification here.
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        let AssembledTags {
            entities: merged,
            ner_entities,
            ..
        } = assemble_tags(
            manager,
            EVAL_USER,
            &mut Vec::new(),
            TagInputs {
                content: &item.content,
                filename: None,
                tags: &item.tags,
                intent_tags: &[],
                ingest_tags: &[],
                ner_entities,
                keywords: keyword_extractor.extract_texts(&item.content),
            },
        );

        // Embed entity/tag names so Tier-4 concept-merge runs (gated; see above).
        // Computed before `merged` is moved into the experience.
        let name_embeddings: Option<HashMap<String, Vec<f32>>> =
            if name_emb_on && !merged.is_empty() {
                let refs: Vec<&str> = merged.iter().map(|s| s.as_str()).collect();
                match user_mem.read().get_embedder().encode_batch(&refs) {
                    Ok(vecs) => Some(merged.iter().cloned().zip(vecs).collect()),
                    Err(_) => None,
                }
            } else {
                None
            };

        let experience = Experience {
            experience_type: experience_type_for(&item.memory_type),
            content: item.content.clone(),
            entities: merged.clone(),
            tags: merged,
            ner_entities,
            ..Default::default()
        };

        let memory_id = user_mem
            .read()
            .remember(experience.clone(), Some(item.created_at))
            .with_context(|| format!("remembering corpus item {}", item.id))?;

        // Pass 2: build the entity graph from this memory (the step the bare
        // MemorySystem path skipped entirely).
        manager
            // veld's signature takes three: it derives name embeddings inside
            // rather than accepting them, so the harness cannot hold them
            // constant across arms the way upstream does.
            .process_experience_into_graph(EVAL_USER, &experience, &memory_id)
            .with_context(|| format!("graph-processing corpus item {}", item.id))?;

        map.insert(item.id.clone(), memory_id.0);
    }

    // Consolidation fidelity (SHODH_EVAL_CONSOLIDATE): production runs a periodic
    // HEAVY maintenance cycle that the harness previously skipped — so the eval
    // never exercised semantic-fact extraction, episodic replay, or tier
    // promotion (W6/W7), leaving the whole consolidation stack unmeasured. When
    // enabled, run one heavy maintenance pass after ingest so those
    // consolidation-driven features are actually populated. decay_factor = 1.0 →
    // no activation decay (forgetting/aging is the separate --age-days path), so
    // this isolates consolidation from decay. Default off → prior behaviour, so a
    // consolidate-off vs -on A/B measures the consolidation/fact contribution.
    if std::env::var("SHODH_EVAL_CONSOLIDATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        match user_mem.read().run_maintenance(
            1.0,
            EVAL_USER,
            crate::memory::MaintenanceDepth::Heavy,
        ) {
            Ok(r) => tracing::info!(
                "eval consolidation cycle: facts_extracted~maintenance ran ({:?})",
                r
            ),
            Err(e) => tracing::warn!("eval consolidation cycle failed: {e}"),
        }
    }

    // Read-your-writes integrity gate: a lost BM25 commit batch means the
    // queries below would measure a partial index — recall numbers that
    // depend on ambient machine state, not the code under test. Production
    // tolerates the (retried, logged) loss for availability; the eval must
    // not. Root-caused 2026-06-12: an external process locking fresh tantivy
    // segment files under the user-profile Documents tree (search indexer /
    // antivirus class) made 432/432 smoke comparisons diverge, silently.
    let bm25_lost = user_mem.read().bm25_commit_failure_count();
    if bm25_lost > 0 {
        anyhow::bail!(
            "BM25 index lost {bm25_lost} commit batch(es) during ingest after retries — \
             the searchable index is missing documents and recall measurements would be \
             invalid. Common cause: eval storage under a directory watched by a search \
             indexer, antivirus, or sync daemon (e.g. the user-profile Documents tree). \
             Move --storage to an unwatched location such as %LOCALAPPDATA%."
        );
    }

    Ok(map)
}

/// Graph-reachability diagnostic — see [`ReachabilityReport`].
///
/// Ingests the corpus through the production graph-building path (identical to
/// the recall run), then for every case performs a pure breadth-first topology
/// walk from the query's seed entities over entity→entity relationships,
/// collecting the episodes attached to each entity at each depth. A gold memory
/// is "reachable within h hops" if its uuid appears among the episodes of any
/// entity within h relationship-hops of a seed entity. Activation weights, tier
/// trust, and the prune threshold are deliberately IGNORED: this measures
/// whether an associative PATH exists, not whether the current scoring would
/// traverse it. `within_2` (one bridge entity) is the canonical double-hop
/// signal for multi_hop.
/// Unified ablation matrix (E1): ingest the suite ONCE, then run the full case
/// set under each named query-time config and tabulate recall@10 / ndcg / mrr /
/// p@1 + per-category recall. The single, re-runnable place to see and update
/// One LongMemEval question: its private haystack (corpus) + turn-level gold.
#[derive(serde::Deserialize)]
struct LongMemEvalCase {
    id: String,
    question: String,
    category: String,
    gold_ids: Vec<String>,
    corpus: String,
}

/// Aggregate LongMemEval result.
///
/// Serialisable so that a sharded run can be recombined: every mean is carried
/// with its own count, so shards aggregate as a count-weighted mean rather than
/// a mean of means.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LongMemEvalReport {
    pub questions: usize,
    pub recall_at_k: f64,
    pub p_at_1: f64,
    pub by_category: BTreeMap<String, (f64, usize)>,
    pub k: usize,
    pub ner_backend: String,
    /// Per-pipeline-layer recall@k attribution: `report_key` -> (mean recall,
    /// scored). When only the production pass is requested (`--layer full`, the
    /// default) this holds a single `full` entry; `--layer all` adds the cumulative
    /// ladder (vamana_only -> +spreading -> +bm25 -> +rerank -> +facts -> full),
    /// measured on the SAME haystacks as `recall_at_k` so the two reconcile.
    pub layers: BTreeMap<String, (f64, usize)>,
}

/// Resolve a shard's `[start, end)` window over a manifest of `len` cases.
///
/// `offset` skips that many cases; `limit` bounds how many follow. Both saturate
/// at `len`, so an out-of-range shard yields an empty window rather than
/// panicking — with a fixed shard count and a variable question total, the tail
/// shards are routinely out of range.
///
/// The correctness claim this exists to pin: for `offset = i * per` and
/// `limit = per`, the windows are disjoint and cover `0..len` exactly once.
fn shard_window(len: usize, offset: Option<usize>, limit: Option<usize>) -> (usize, usize) {
    let start = offset.unwrap_or(0).min(len);
    let end = match limit {
        Some(n) => start.saturating_add(n).min(len),
        None => len,
    };
    (start, end)
}

/// Run the LongMemEval-S suite (ICLR 2025, the current SOTA long-term memory
/// benchmark). UNLIKE LoCoMo, each question carries its OWN ~48-session haystack
/// (~115K tokens), so this is a LOOP of mini-evals: per question, ingest its
/// haystack into fresh storage, run the question as a query, and score
/// turn-level recall@k against the `has_answer` gold turns. No LLM judge — the
/// same gold-evidence retrieval signal the smoke/LoCoMo suites use.
///
/// Fixtures come from `benchmarks/longmemeval_to_harness.py` (manifest.jsonl +
/// per-question corpora/<id>.jsonl). `limit` bounds the question count for a
/// cheap pilot (each haystack is a full ~500-turn ingest).
pub fn run_longmemeval(
    base_dir: &Path,
    storage_root: &Path,
    offset: Option<usize>,
    limit: Option<usize>,
    k: usize,
    layer_modes: &[LayerMode],
) -> Result<LongMemEvalReport> {
    pin_harness_threads();

    let manifest_path = base_dir.join("manifest.jsonl");
    let manifest_txt = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let mut cases: Vec<LongMemEvalCase> = Vec::new();
    for line in manifest_txt.lines().filter(|l| !l.trim().is_empty()) {
        cases.push(serde_json::from_str(line).context("parsing LongMemEval manifest line")?);
    }
    // Shard selection: skip `offset` cases, then take `limit`. LongMemEval-S
    // gives every question its OWN ~490-turn haystack, so cost is linear in
    // questions and a full run cannot fit one job's timeout. Disjoint
    // (offset, limit) windows let parallel jobs cover the suite exactly once.
    // The manifest order is deterministic - the converter's `--shuffle` is a
    // stable sort - so the windows are stable across jobs and across reruns.
    let (start, end) = shard_window(cases.len(), offset, limit);
    cases = cases.drain(start..end).collect();

    let mut sum_recall = 0.0f64;
    let mut sum_p1 = 0.0f64;
    let mut scored = 0usize;
    let mut by_cat: HashMap<String, (f64, usize)> = HashMap::new();
    let mut ner_backend = String::from("unknown");

    // Per-layer attribution: each question's haystack is ingested ONCE, then the
    // recall runs once per requested LayerMode (a query-time setting), so the
    // ladder is measured on the SAME corpus as the headline. `full` (the default)
    // runs a single pass and is byte-identical to the pre-attribution path. The
    // headline aggregates (recall_at_k, by_category) always track the production
    // (primary) pass; the per-mode means populate `layers`.
    let modes: Vec<LayerMode> = if layer_modes.is_empty() {
        vec![LayerMode::Full]
    } else {
        layer_modes.to_vec()
    };
    let primary_mode = if modes.contains(&LayerMode::Full) {
        LayerMode::Full
    } else {
        modes[0]
    };
    // report_key -> (sum recall, scored count) over every requested mode.
    let mut mode_agg: BTreeMap<String, (f64, usize)> = BTreeMap::new();

    for (qi, case) in cases.iter().enumerate() {
        let corpus_path = base_dir.join(&case.corpus);
        let corpus_txt = std::fs::read_to_string(&corpus_path)
            .with_context(|| format!("reading {}", corpus_path.display()))?;
        let corpus: Vec<CorpusItem> = corpus_txt
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()
            .context("parsing LongMemEval corpus")?;

        // Fresh, isolated storage per question so haystacks never bleed across.
        let q_storage = storage_root.join(format!("q{qi}"));
        let _ = std::fs::remove_dir_all(&q_storage);
        let manager = build_manager(&q_storage)?;
        if qi == 0 {
            ner_backend = if manager.get_neural_ner().is_fallback_mode() {
                "fallback".to_string()
            } else {
                "neural".to_string()
            };
            eprintln!("NER_BACKEND={ner_backend}");
            if ner_backend == "fallback"
                && !cfg!(test)
                && !std::env::var("SHODH_ALLOW_FALLBACK_NER")
                    .map(|v| v == "1")
                    .unwrap_or(false)
            {
                anyhow::bail!(
                    "LongMemEval refusing to run on FALLBACK NER — set \
                     SHODH_ALLOW_FALLBACK_NER=1 to override visibly."
                );
            }
        }

        let id_map = ingest_corpus(&manager, &corpus)?;
        let system = manager.get_user_memory(EVAL_USER)?;

        // Turn-level gold (`has_answer` mapped through ingest) — mode-independent,
        // resolved once before the per-layer recall loop.
        let gold_uuids: Vec<Uuid> = case
            .gold_ids
            .iter()
            .filter_map(|g| id_map.get(g).copied())
            .collect();
        if gold_uuids.is_empty() {
            // Gold turn dropped at ingest (e.g. empty content) — unscorable.
            let _ = std::fs::remove_dir_all(&q_storage);
            continue;
        }
        // NER annotation is mode-independent — annotate once, then vary only the
        // LayerMode across the per-layer loop.
        let mut query = Query {
            query_text: Some(case.question.clone()),
            max_results: k,
            layers: Some(primary_mode),
            ..Default::default()
        };
        manager.annotate_query_ner(&mut query);

        for &mode in &modes {
            query.layers = Some(mode);
            let memories = system.read().recall(&query).unwrap_or_default();
            let topk: HashSet<Uuid> = memories.iter().take(k).map(|m| m.id.0).collect();

            // Turn-level recall@k: fraction of gold turns in the top-k.
            let hit = gold_uuids.iter().filter(|g| topk.contains(g)).count();
            let recall = hit as f64 / gold_uuids.len() as f64;

            let me = mode_agg
                .entry(mode.report_key().to_string())
                .or_insert((0.0, 0));
            me.0 += recall;
            me.1 += 1;

            // Headline aggregates track the production (primary) pass only.
            if mode == primary_mode {
                // SHODH_DUMP_CONTEXT=<path>: append one JSONL record per question
                // with the retrieved top-k turn TEXTS (production pass only).
                // Feeds the reader-composition study: an external reader model
                // answers from exactly what the memory system surfaced, so
                // answer accuracy can be attributed to retrieval vs reading.
                // Gold answers are NOT exported -- the scorer joins them from
                // the pinned upstream dataset by question id, so this dump can
                // never leak gold into the reader's context by construction.
                if let Ok(dump_path) = std::env::var("SHODH_DUMP_CONTEXT") {
                    use std::io::Write as _;
                    let retrieved: Vec<serde_json::Value> = memories
                        .iter()
                        .take(k)
                        .enumerate()
                        .map(|(rank, m)| {
                            serde_json::json!({
                                "rank": rank + 1,
                                "content": m.experience.content,
                                "is_gold": gold_uuids.contains(&m.id.0),
                            })
                        })
                        .collect();
                    let line = serde_json::json!({
                        "question_id": case.id,
                        "question": case.question,
                        "category": case.category,
                        "gold_in_topk": hit,
                        "gold_total": gold_uuids.len(),
                        "retrieved": retrieved,
                    });
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&dump_path)
                    {
                        let _ = writeln!(f, "{line}");
                    }
                }
                sum_recall += recall;
                // p@1: is the single top-ranked memory a gold turn? A finer signal
                // than recall@10, which only moves when gold crosses the top-k boundary.
                if memories
                    .first()
                    .map(|m| gold_uuids.contains(&m.id.0))
                    .unwrap_or(false)
                {
                    sum_p1 += 1.0;
                }
                scored += 1;
                let entry = by_cat.entry(case.category.clone()).or_insert((0.0, 0));
                entry.0 += recall;
                entry.1 += 1;
            }
        }

        if (qi + 1) % 25 == 0 {
            eprintln!(
                "  longmemeval: {}/{} questions, running recall@{k}={:.4}",
                qi + 1,
                cases.len(),
                sum_recall / scored as f64
            );
        }

        // Reclaim disk — each haystack is a full store.
        drop(manager);
        let _ = std::fs::remove_dir_all(&q_storage);
    }

    let by_category: BTreeMap<String, (f64, usize)> = by_cat
        .into_iter()
        .map(|(c, (s, n))| (c, (s / n.max(1) as f64, n)))
        .collect();

    // Proof-of-execution for the edge-direction fix: with the flag off this is
    // always 0; with it on, a non-zero count proves the patched traversal ran
    // and changed targets, independent of whether recall@10 moved.
    // n/a, not 0: veld has no edge_neighbor and no dir_fix branch, so there is
    // no traversal to flip. A 0 here would read as "the fix ran and changed
    // nothing", which is a different claim.
    eprintln!("EDGE_DIR_FLIPS=n/a (no dir_fix traversal in this build)");

    // #8 measurement: how many CAUSAL attestations collapsed into an existing
    // edge in the REVERSE direction during ingest. Gates whether a
    // direction-sensitive pair key (+ index migration) is worth building.
    // n/a for the same reason: nothing tracks reverse-direction collapse here.
    eprintln!("DIRECTED_REVERSE_COLLAPSE=n/a (not tracked in this build)");

    let layers: BTreeMap<String, (f64, usize)> = mode_agg
        .into_iter()
        .map(|(key, (s, n))| (key, (s / n.max(1) as f64, n)))
        .collect();

    // Per-layer ladder to the log when more than the production pass ran, in
    // canonical pipeline order so the CI summary can grep it.
    if layers.len() > 1 {
        eprintln!("LONGMEMEVAL_LAYERS (recall@{k}, same haystacks, query-time attribution):");
        for mode in LayerMode::ALL {
            if let Some((r, n)) = layers.get(mode.report_key()) {
                eprintln!(
                    "  LONGMEMEVAL_LAYER {:14} n={n:<4} recall@{k} = {r:.4}",
                    mode.report_key()
                );
            }
        }
    }

    Ok(LongMemEvalReport {
        questions: scored,
        recall_at_k: if scored > 0 {
            sum_recall / scored as f64
        } else {
            0.0
        },
        p_at_1: if scored > 0 {
            sum_p1 / scored as f64
        } else {
            0.0
        },
        by_category,
        k,
        ner_backend,
        layers,
    })
}

/// The arm every other ablation arm is measured against.
pub(crate) const ABLATION_BASELINE: &str = "baseline (facts on)";
/// Bootstrap resamples per paired delta. At 2000 the Monte-Carlo wobble of an
/// interval endpoint is about 0.06 standard errors of the mean difference, small
/// against the interval's own width.
const ABLATION_BOOTSTRAP_RESAMPLES: usize = 2000;
/// Fixed so a report regenerated from the same per-case data carries the same
/// intervals.
const ABLATION_BOOTSTRAP_SEED: u64 = 0x5EED_AB1A;
/// How a row names the dial the `-ce` arm turns.
pub(crate) const RERANK_OFF_FLAG: &str = "dials.use_reranking=false";

/// One arm of the ablation matrix: the env overrides it applies, whether it turns
/// the cross-encoder rerank off, and the arm it is measured against.
pub(crate) struct AblationConfig {
    pub(crate) name: &'static str,
    pub(crate) env: Vec<(&'static str, &'static str)>,
    /// Runs the arm with the `use_reranking` dial off. A dial, not an env flag:
    /// the dials read the environment once, when the memory system is built.
    pub(crate) rerank_off: bool,
    /// `None` only for the baseline. Every other arm is its reference plus what it
    /// adds, so its delta is attributable to exactly that.
    pub(crate) reference: Option<&'static str>,
}

impl AblationConfig {
    #[cfg(test)]
    fn sets(&self, key: &str) -> bool {
        self.env.iter().any(|(k, _)| *k == key)
    }

    fn flags(&self) -> Vec<String> {
        let mut flags: Vec<String> = self.env.iter().map(|(k, v)| format!("{k}={v}")).collect();
        if self.rerank_off {
            flags.push(RERANK_OFF_FLAG.to_string());
        }
        flags
    }
}

/// Every arm of the ablation matrix.
///
/// Add an arm whenever a query-time lever lands; that is how a lever becomes a
/// measured, comparable line in the study. An arm that cannot differ from its
/// reference reports a number attributable to nothing, so before adding one
/// confirm the flag is read on the live path, or register it in
/// [`UNAVAILABLE_IN_VELD`]. The guard test
/// `every_ablation_arm_flag_is_read_in_src_or_declared_unavailable` enforces this.
///
/// The baseline is the pipeline as veld ships it, cross-encoder rerank included,
/// so the leave-one-out arms read each component's marginal contribution at the
/// operating point users get, and `-ce` reads the reranker's.
///
/// Setting `SHODH_DISABLE_BOOSTS` replaces its default-disabled set, so every arm
/// that sets it names `hebbian` (or `all`) to keep the term off, except `+hebbian`,
/// which empties the list to switch it on.
pub(crate) fn ablation_configs() -> Vec<AblationConfig> {
    fn arm(name: &'static str, env: &[(&'static str, &'static str)]) -> AblationConfig {
        AblationConfig {
            name,
            env: env.to_vec(),
            rerank_off: false,
            reference: Some(ABLATION_BASELINE),
        }
    }
    vec![
        AblationConfig {
            name: ABLATION_BASELINE,
            env: vec![],
            rerank_off: false,
            reference: None,
        },
        arm(
            "facts-off (lexical+graph)",
            &[("SHODH_DISABLE_FACT_LAYERS", "1")],
        ),
        arm("graph-off", &[("SHODH_GRAPH_FUSION_WEIGHT", "0")]),
        arm("+graph-expand(K5)", &[("SHODH_GRAPH_EXPAND_K", "5")]),
        arm(
            "+graph-expand+margin",
            &[
                ("SHODH_GRAPH_EXPAND_K", "5"),
                ("SHODH_GRAPH_EXPAND_MIN_STRENGTH", "0.3"),
            ],
        ),
        arm(
            "+graph-boost-mult",
            &[("SHODH_GRAPH_BOOST_MULTIPLICATIVE", "1")],
        ),
        arm("+edge-dir", &[("SHODH_GRAPH_EDGE_DIR", "1")]),
        arm("+graph-traverse", &[("SHODH_GRAPH_TRAVERSE", "1")]),
        arm(
            "-lateral-inhibition",
            &[("SHODH_DISABLE_BOOSTS", "hebbian,lateral_inhibition")],
        ),
        arm(
            "-graph-potentiation",
            &[("SHODH_DISABLE_BOOSTS", "hebbian,graph_potentiation")],
        ),
        arm("+hebbian", &[("SHODH_DISABLE_BOOSTS", "")]),
        arm(
            "+graph-boost-mult+expand",
            &[
                ("SHODH_GRAPH_BOOST_MULTIPLICATIVE", "1"),
                ("SHODH_GRAPH_EXPAND_K", "5"),
            ],
        ),
        // Leave-one-out: each removes one leg or stage from the shipped pipeline, so
        // its delta is that component's marginal contribution. The cumulative
        // `--layer all` ladder attributes each stage conditional on the ones before.
        arm("-bm25", &[("SHODH_ABLATE_BM25", "1")]),
        arm("-spreading", &[("SHODH_ABLATE_SPREADING", "1")]),
        arm("-boost-stack", &[("SHODH_DISABLE_BOOSTS", "all")]),
        // Single-leg isolation: one leg's candidates only.
        arm("leg=vector", &[("SHODH_LEG", "vector")]),
        arm("leg=bm25", &[("SHODH_LEG", "bm25")]),
        arm("leg=graph", &[("SHODH_LEG", "graph")]),
        AblationConfig {
            name: "-ce",
            env: vec![],
            rerank_off: true,
            reference: Some(ABLATION_BASELINE),
        },
    ]
}

/// Upstream flags an arm may set that nothing in veld reads, with the reason.
/// A flag here must not be read anywhere in `src/`; porting it means deleting
/// its entry, and the guard test fails until that happens.
pub(crate) const UNAVAILABLE_IN_VELD: &[(&str, &str)] = &[
    (
        "SHODH_GRAPH_EXPAND_K",
        "veld has no graph-driven BM25 cue expansion (upstream appends top-K 1-hop neighbour names to the BM25 query)",
    ),
    (
        "SHODH_GRAPH_EXPAND_MIN_STRENGTH",
        "margin gate of the graph cue expansion, which veld does not have",
    ),
    (
        "SHODH_GRAPH_BOOST_MULTIPLICATIVE",
        "veld's graph leg has no multiplicative boost form; its additive recency/arousal/credibility boosts exist only in the SpreadingActivation-mode path",
    ),
    (
        "SHODH_GRAPH_EDGE_DIR",
        "veld's spreading paths always resolve the true neighbour (GraphEdge::neighbor_of); there is no legacy to_entity-only path to toggle",
    ),
    (
        "SHODH_GRAPH_TRAVERSE",
        "veld has no intent-matching beam traversal (upstream traverse_beam)",
    ),
    (
        "SHODH_ABLATE_BM25",
        "veld's recall reads no leave-one-out switch for the BM25 leg (upstream #378); the switch belongs in memory/recall.rs",
    ),
    (
        "SHODH_ABLATE_SPREADING",
        "veld's recall reads no leave-one-out switch for graph spreading (upstream #378); the switch belongs in memory/recall.rs",
    ),
    (
        "SHODH_LEG",
        "veld's recall has no single-leg isolation (upstream SHODH_LEG)",
    ),
];

/// The row for an arm that was not run: no metrics, and the reason in their place.
fn unavailable_row(config: &AblationConfig, reason: String) -> AblationRow {
    AblationRow {
        name: config.name.to_string(),
        flags: config.flags(),
        recall_at_10: None,
        ndcg_at_10: None,
        mrr: None,
        p_at_1: None,
        by_category_recall: Default::default(),
        retrieval_fingerprint: 0,
        vacuous_vs_reference: false,
        unavailable_in_veld: Some(reason),
        reference: config.reference.map(str::to_string),
        delta_recall_at_10: None,
        delta_p_at_1: None,
        per_case: Vec::new(),
    }
}

/// Why `env` cannot be measured in veld, or `None` when every flag it sets
/// reaches a live code path.
pub(crate) fn arm_unavailable_in_veld(env: &[(&str, &str)]) -> Option<String> {
    use crate::memory::ablation::{DEFAULT_DISABLED, FAMILIES};
    let mut reasons = Vec::new();
    for (k, v) in env {
        if let Some((_, why)) = UNAVAILABLE_IN_VELD.iter().find(|(f, _)| f == k) {
            reasons.push(format!("{k}: {why}"));
        }
        if *k == crate::memory::ablation::DISABLE_BOOSTS_VAR {
            for tok in v.split(',').map(|t| t.trim().to_ascii_lowercase()) {
                if tok.is_empty() {
                    continue;
                }
                if tok == "all" {
                    // `all` can only differ from the default through a family that is
                    // on by default and live under the harness's read-only recall.
                    let live = FAMILIES.iter().any(|f| {
                        !DEFAULT_DISABLED.contains(f)
                            && !(*f == "graph_potentiation" && crate::memory::recall_readonly())
                    });
                    if !live {
                        reasons.push(format!(
                            "{k}=all: veld gates {FAMILIES:?}; {DEFAULT_DISABLED:?} is off by default and SHODH_RECALL_READONLY=1 suppresses graph_potentiation, so `all` disables nothing more than the default"
                        ));
                    }
                } else if let Some(why) = crate::memory::ablation::unavailable_reason(&tok) {
                    reasons.push(format!("{k} family `{tok}`: {why}"));
                } else if !FAMILIES.contains(&tok.as_str()) {
                    reasons.push(format!("{k} family `{tok}` is not declared"));
                } else if tok == "graph_potentiation" && crate::memory::recall_readonly() {
                    reasons.push(format!(
                        "{k} family `{tok}`: SHODH_RECALL_READONLY=1 already suppresses every potentiation write, so this arm cannot differ from baseline"
                    ));
                }
            }
        }
    }
    (!reasons.is_empty()).then(|| reasons.join("; "))
}

/// The keys some arm sets that `is_set` reports as already present, sorted and
/// deduplicated.
fn preset_ablation_keys(
    configs: &[AblationConfig],
    is_set: impl Fn(&str) -> bool,
) -> Vec<&'static str> {
    let mut preset: Vec<&'static str> = configs
        .iter()
        .flat_map(|c| c.env.iter().map(|(k, _)| *k))
        .filter(|k| is_set(k))
        .collect();
    preset.sort_unstable();
    preset.dedup();
    preset
}

/// Refuse to run when the process already carries a key some arm sets.
///
/// Each arm sets its keys and removes them afterwards. A key already in the
/// environment would be on for every arm before the one that sets it and gone for
/// every arm after, so the arms would stop sharing a reference point and the
/// deltas would measure arm order.
fn preflight_ablation_env(configs: &[AblationConfig]) -> Result<()> {
    let preset = preset_ablation_keys(configs, |k| std::env::var_os(k).is_some());
    if !preset.is_empty() {
        anyhow::bail!(
            "ablation matrix refusing to run: {} already set in the environment. Every arm \
             sets and then removes its own keys, so a preset key would be on for some arms \
             and off for others, and the deltas would measure arm order. Unset them, or add \
             an arm that sets them.",
            preset.join(", ")
        );
    }
    Ok(())
}

/// Refuse to measure a reranked pipeline whose cross-encoder does not load.
///
/// `CrossEncoderReranker::rerank` falls back to bi-encoder scores when the model
/// is missing, so without this a run would report the unreranked pipeline under
/// the shipped one's name. Loads the model now and scores one pair. Nothing is
/// checked when the `use_reranking` dial is off, since then no reranker runs.
/// `cfg!(test)` is exempt, as in [`guard_ner_backend`]: lib tests exercise the
/// harness machinery on machines that may lack the model. Returns the id of the
/// cross-encoder that loaded, `None` when none was checked.
pub(crate) fn guard_cross_encoder(manager: &MultiUserMemoryManager) -> Result<Option<String>> {
    if cfg!(test) {
        return Ok(None);
    }
    let system = manager.get_user_memory(EVAL_USER)?;
    let system = system.read();
    if system.cognitive_dials().use_reranking == Some(false) {
        eprintln!("CROSS_ENCODER=off (use_reranking dial)");
        return Ok(None);
    }
    match system.probe_cross_encoder() {
        None => {
            eprintln!("CROSS_ENCODER=none (no reranker configured)");
            Ok(None)
        }
        Some(Ok(id)) => {
            eprintln!("CROSS_ENCODER={id}");
            Ok(Some(id))
        }
        Some(Err(e)) => anyhow::bail!(
            "recall harness refusing to run: the cross-encoder did not load ({e:#}), so \
             reranking would fall back to bi-encoder scores and the run would measure a \
             pipeline veld does not ship. Provide the model, or switch reranking off with \
             VELD_DIALS_USE_RERANKING=false to measure without it."
        ),
    }
}

/// Unified ablation matrix: one ingest, one row per [`ablation_configs`] entry, so
/// the study shows every component's contribution. Each delta is paired against
/// the arm's own reference over cases, with a seeded bootstrap interval. An arm
/// naming a flag veld has no code path for is reported unavailable, not measured.
///
/// Only QUERY-TIME flags are ablatable here (they are read per-recall against the
/// same ingested corpus). Ingest-time flags (concept entities, IDF edges, hub cap)
/// change what is stored and need separate ingests.
pub fn analyze_ablation(inputs: &RunInputs) -> Result<AblationReport> {
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;
    // The same fast-read caps as the recall run and the funnel; unset means full scale.
    let (cases, corpus) = apply_eval_caps(cases, corpus);

    let configs = ablation_configs();
    preflight_ablation_env(&configs)?;

    let manager = build_manager(&inputs.storage_path)?;
    guard_cross_encoder(&manager)?;
    let id_map = ingest_corpus(&manager, &corpus)?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // One case through the live recall path, under whatever env and dials are set now.
    let recall_ids = |case: &SmokeCase| -> Vec<Uuid> {
        let query = Query {
            query_text: Some(case.query.clone()),
            max_results: SMOKE_K,
            layers: Some(LayerMode::Full),
            ..Default::default()
        };
        system
            .read()
            .recall(&query)
            .unwrap_or_default()
            .iter()
            .map(|m| m.id.0)
            .collect()
    };

    let mut rows: Vec<AblationRow> = Vec::with_capacity(configs.len());
    for config in &configs {
        let (name, env) = (config.name, &config.env);
        if let Some(reason) = arm_unavailable_in_veld(env) {
            eprintln!("  -- ARM `{name}` unavailable in veld: {reason}");
            rows.push(unavailable_row(config, reason));
            continue;
        }
        // SAFETY: the harness is single-threaded (pin_harness_threads) and the
        // production server never invokes it, so process-wide env mutation here
        // races no other reader. Set before the query pass, clear after.
        unsafe {
            for (k, v) in env {
                std::env::set_var(k, v);
            }
        }
        let dials_before = config.rerank_off.then(|| {
            let guard = system.read();
            let before = guard.cognitive_dials();
            let mut off = before.clone();
            off.use_reranking = Some(false);
            guard.set_cognitive_dials(off);
            before
        });

        let mut by_cat: HashMap<SmokeCategory, Vec<f64>> = HashMap::new();
        let mut all: Vec<Metrics> = Vec::with_capacity(cases.len());
        let mut per_case: Vec<AblationCaseRow> = Vec::with_capacity(cases.len());
        // Order-sensitive fingerprint over every retrieved id. An arm that
        // matches its reference here provably changed nothing.
        let mut fp = std::collections::hash_map::DefaultHasher::new();
        for case in &cases {
            let retrieved = recall_ids(case);
            hash_retrieved(&mut fp, &retrieved);
            let relevance = build_relevance_map(case, &id_map);
            let m = Metrics::compute(&retrieved, &relevance, SMOKE_K);
            by_cat.entry(case.category).or_default().push(m.recall_at_k);
            per_case.push(AblationCaseRow {
                case_id: case.id.clone(),
                category: category_name(case.category).to_string(),
                recall_at_k: m.recall_at_k,
                p_at_1: m.p_at_1,
                ndcg_at_k: m.ndcg_at_k,
            });
            all.push(m);
        }

        if let Some(before) = dials_before {
            system.read().set_cognitive_dials(before);
        }
        unsafe {
            for (k, _) in env {
                std::env::remove_var(k);
            }
        }

        let n = all.len().max(1) as f64;
        let by_category_recall = by_cat
            .iter()
            .map(|(c, rs)| {
                let r = rs.iter().sum::<f64>() / rs.len().max(1) as f64;
                (category_name(*c).to_string(), r)
            })
            .collect();
        rows.push(AblationRow {
            name: name.to_string(),
            flags: config.flags(),
            recall_at_10: Some(all.iter().map(|m| m.recall_at_k).sum::<f64>() / n),
            ndcg_at_10: Some(all.iter().map(|m| m.ndcg_at_k).sum::<f64>() / n),
            mrr: Some(all.iter().map(|m| m.mrr).sum::<f64>() / n),
            p_at_1: Some(all.iter().map(|m| m.p_at_1).sum::<f64>() / n),
            by_category_recall,
            retrieval_fingerprint: std::hash::Hasher::finish(&fp),
            vacuous_vs_reference: false,
            unavailable_in_veld: None,
            reference: config.reference.map(str::to_string),
            delta_recall_at_10: None,
            delta_p_at_1: None,
            per_case,
        });
    }

    // Trailing baseline replicate: the baseline's queries again, with no arm
    // setting, after every arm has run. A flag read once into a cache would stay
    // stuck for every later arm, and those arms would still differ from their
    // reference, so only re-measuring the reference point at the end can show it.
    let replicate_fp = {
        let mut fp = std::collections::hash_map::DefaultHasher::new();
        for case in &cases {
            hash_retrieved(&mut fp, &recall_ids(case));
        }
        std::hash::Hasher::finish(&fp)
    };
    let baseline_replicate_identical = rows
        .iter()
        .find(|r| r.reference.is_none() && r.unavailable_in_veld.is_none())
        .map(|r| r.retrieval_fingerprint == replicate_fp);
    match baseline_replicate_identical {
        Some(true) => eprintln!("  baseline replicate: identical to the first baseline pass"),
        Some(false) => {
            tracing::warn!("ABLATION BASELINE REPLICATE DIFFERS from the first baseline pass");
            eprintln!(
                "  !! baseline replicate DIFFERS from the first pass: an arm left state \
                 behind, or query-time retrieval is not deterministic on a fixed ingest. \
                 Every delta in this matrix carries that difference as noise."
            );
        }
        None => {}
    }

    resolve_against_references(&mut rows)?;

    Ok(AblationReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        case_count: cases.len(),
        rows,
        baseline_replicate_identical,
    })
}

/// Fold one case's ranked ids into an order-sensitive run fingerprint.
fn hash_retrieved(fp: &mut std::collections::hash_map::DefaultHasher, ids: &[Uuid]) {
    use std::hash::Hash;
    for id in ids {
        id.hash(fp);
    }
    // Separator so [a],[b] and [a,b] cannot collide.
    0xFFu8.hash(fp);
}

/// Fill each measured arm's paired deltas and vacuity flag from the arm it names
/// as its reference.
///
/// Vacuity: an arm whose retrieved ids are byte-identical to its reference's did
/// not exercise the code path its flags name. It is flagged rather than failed,
/// because an arm may also be legitimately inert (it executed and changed no
/// ranking), and only the fingerprint can tell the two apart.
fn resolve_against_references(rows: &mut [AblationRow]) -> Result<()> {
    let snapshot: HashMap<String, (u64, Vec<f64>, Vec<f64>, bool)> = rows
        .iter()
        .map(|r| {
            (
                r.name.clone(),
                (
                    r.retrieval_fingerprint,
                    r.per_case.iter().map(|c| c.recall_at_k).collect(),
                    r.per_case.iter().map(|c| c.p_at_1).collect(),
                    r.unavailable_in_veld.is_some(),
                ),
            )
        })
        .collect();
    let ci = |control: &[f64], treatment: &[f64]| {
        paired_bootstrap_ci(
            control,
            treatment,
            ABLATION_BOOTSTRAP_RESAMPLES,
            ABLATION_BOOTSTRAP_SEED,
        )
        .map(|(mean, lo, hi)| DeltaCi { mean, lo, hi })
    };

    for row in rows.iter_mut() {
        let Some(reference) = row.reference.clone() else {
            continue;
        };
        let Some((ref_fp, ref_recall, ref_p1, ref_unavailable)) = snapshot.get(&reference) else {
            anyhow::bail!(
                "ablation arm `{}` names `{reference}` as its reference, which is not an arm",
                row.name
            );
        };
        if row.unavailable_in_veld.is_some() || *ref_unavailable {
            continue;
        }
        let recall: Vec<f64> = row.per_case.iter().map(|c| c.recall_at_k).collect();
        let p1: Vec<f64> = row.per_case.iter().map(|c| c.p_at_1).collect();
        row.delta_recall_at_10 = ci(ref_recall, &recall);
        row.delta_p_at_1 = ci(ref_p1, &p1);

        if row.retrieval_fingerprint == *ref_fp {
            row.vacuous_vs_reference = true;
            tracing::warn!(
                arm = %row.name,
                reference = %reference,
                flags = %row.flags.join(" "),
                "VACUOUS ABLATION ARM: retrieval is byte-identical to its reference, so the \
                 flags it adds did not reach the code path they name. Its delta measures nothing."
            );
            eprintln!(
                "  !! VACUOUS ARM `{}` ({}) -- byte-identical to `{}`, measures nothing",
                row.name,
                row.flags.join(" "),
                reference
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod ablation_config_tests {
    use super::*;

    fn configs_by_name(configs: &[AblationConfig]) -> HashMap<&'static str, &AblationConfig> {
        configs.iter().map(|c| (c.name, c)).collect()
    }

    #[test]
    fn arm_names_are_unique() {
        let configs = ablation_configs();
        assert_eq!(configs_by_name(&configs).len(), configs.len());
    }

    #[test]
    fn only_the_baseline_lacks_a_reference_and_it_sets_nothing() {
        let configs = ablation_configs();
        let roots: Vec<&AblationConfig> =
            configs.iter().filter(|c| c.reference.is_none()).collect();
        assert_eq!(roots.len(), 1, "exactly one arm may be unreferenced");
        assert_eq!(roots[0].name, ABLATION_BASELINE);
        assert!(
            roots[0].env.is_empty() && !roots[0].rerank_off,
            "the baseline must be the pipeline as it ships"
        );
    }

    #[test]
    fn every_arm_is_its_reference_plus_at_least_one_flag() {
        let configs = ablation_configs();
        let by_name = configs_by_name(&configs);
        for c in &configs {
            let Some(reference) = c.reference else {
                continue;
            };
            assert_ne!(reference, c.name, "`{}` references itself", c.name);
            let r = by_name
                .get(reference)
                .unwrap_or_else(|| panic!("`{}` references unknown arm `{reference}`", c.name));
            for pair in &r.env {
                assert!(
                    c.env.contains(pair),
                    "`{}` drops {pair:?} from its reference `{reference}`, so its delta \
                     is not attributable to the flags it adds",
                    c.name
                );
            }
            assert!(
                !r.rerank_off || c.rerank_off,
                "`{}` turns the reranker back on against `{reference}`",
                c.name
            );
            assert!(
                c.flags().len() > r.flags().len(),
                "`{}` adds nothing to its reference `{reference}`",
                c.name
            );
        }
    }

    #[test]
    fn the_rerank_off_arm_names_its_dial_in_its_flags() {
        let configs = ablation_configs();
        let ce = configs
            .iter()
            .find(|c| c.rerank_off)
            .expect("an arm measures the cross-encoder");
        assert_eq!(ce.flags(), vec![RERANK_OFF_FLAG.to_string()]);
        assert_eq!(ce.reference, Some(ABLATION_BASELINE));
    }

    #[test]
    fn preset_keys_are_reported_once_each_and_only_if_an_arm_sets_them() {
        let configs = ablation_configs();
        let preset = preset_ablation_keys(&configs, |k| {
            matches!(
                k,
                "SHODH_LEG" | "SHODH_DISABLE_BOOSTS" | "SHODH_NOT_AN_ARM_KEY"
            )
        });
        assert_eq!(preset, vec!["SHODH_DISABLE_BOOSTS", "SHODH_LEG"]);
        assert!(preset_ablation_keys(&configs, |_| false).is_empty());
    }

    /// `all` reaches nothing the default does not, under the harness's read-only
    /// recall, so `-boost-stack` is reported unavailable instead of run as a
    /// baseline copy.
    #[test]
    fn disabling_all_boosts_is_unavailable_under_read_only_recall() {
        let _guard = crate::memory::ablation::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("VELD_RECALL_READONLY").ok();
        std::env::set_var("VELD_RECALL_READONLY", "1");
        let why = arm_unavailable_in_veld(&[("SHODH_DISABLE_BOOSTS", "all")]);
        std::env::set_var("VELD_RECALL_READONLY", "0");
        let writable = arm_unavailable_in_veld(&[("SHODH_DISABLE_BOOSTS", "all")]);
        match prev {
            Some(v) => std::env::set_var("VELD_RECALL_READONLY", v),
            None => std::env::remove_var("VELD_RECALL_READONLY"),
        }
        assert!(
            why.is_some_and(|w| w.contains("nothing more than the default")),
            "read-only: `all` must be unavailable"
        );
        assert_eq!(
            writable, None,
            "with writes on, `all` also disables graph_potentiation"
        );
    }

    fn row(name: &str, reference: Option<&str>, fingerprint: u64, recalls: &[f64]) -> AblationRow {
        AblationRow {
            name: name.to_string(),
            flags: if reference.is_some() {
                vec![format!("FLAG_FOR_{name}=1")]
            } else {
                vec![]
            },
            recall_at_10: Some(recalls.iter().sum::<f64>() / recalls.len() as f64),
            ndcg_at_10: Some(0.0),
            mrr: Some(0.0),
            p_at_1: Some(0.0),
            by_category_recall: BTreeMap::new(),
            retrieval_fingerprint: fingerprint,
            vacuous_vs_reference: false,
            unavailable_in_veld: None,
            reference: reference.map(str::to_string),
            delta_recall_at_10: None,
            delta_p_at_1: None,
            per_case: recalls
                .iter()
                .enumerate()
                .map(|(i, &r)| AblationCaseRow {
                    case_id: format!("case-{i}"),
                    category: "single_hop".to_string(),
                    recall_at_k: r,
                    p_at_1: r,
                    ndcg_at_k: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn the_fingerprint_keeps_case_boundaries_and_rank_order() {
        let (a, b) = (Uuid::from_u128(1), Uuid::from_u128(2));
        let finish = |cases: &[Vec<Uuid>]| {
            let mut fp = std::collections::hash_map::DefaultHasher::new();
            for ids in cases {
                hash_retrieved(&mut fp, ids);
            }
            std::hash::Hasher::finish(&fp)
        };
        assert_ne!(finish(&[vec![a], vec![b]]), finish(&[vec![a, b]]));
        assert_ne!(finish(&[vec![a, b]]), finish(&[vec![b, a]]));
        assert_eq!(finish(&[vec![a, b]]), finish(&[vec![a, b]]));
    }

    #[test]
    fn an_arm_identical_to_its_reference_is_vacuous_with_a_zero_delta() {
        let mut rows = vec![
            row("base", None, 7, &[0.0, 0.5, 1.0]),
            row("inert", Some("base"), 7, &[0.0, 0.5, 1.0]),
        ];
        resolve_against_references(&mut rows).unwrap();
        assert!(rows[1].vacuous_vs_reference);
        let d = rows[1].delta_recall_at_10.unwrap();
        assert_eq!((d.mean, d.lo, d.hi), (0.0, 0.0, 0.0));
        assert!(
            !rows[0].vacuous_vs_reference,
            "the baseline has nothing to be vacuous against"
        );
        assert!(rows[0].delta_recall_at_10.is_none());
    }

    #[test]
    fn a_nested_arm_is_measured_against_its_own_reference_not_the_baseline() {
        // base 0.2 -> mid 0.6 -> child 0.5. Against its reference the child loses
        // 0.1; against the baseline it would appear to gain 0.3.
        let mut rows = vec![
            row("base", None, 1, &[0.2; 4]),
            row("mid", Some("base"), 2, &[0.6; 4]),
            row("child", Some("mid"), 3, &[0.5; 4]),
        ];
        resolve_against_references(&mut rows).unwrap();
        let child = rows[2].delta_recall_at_10.unwrap();
        assert!((child.mean + 0.1).abs() < 1e-12, "got {}", child.mean);
        assert!(child.excludes_zero());
        let mid = rows[1].delta_recall_at_10.unwrap();
        assert!((mid.mean - 0.4).abs() < 1e-12, "got {}", mid.mean);
        assert!(!rows[2].vacuous_vs_reference);
    }

    #[test]
    fn an_unavailable_arm_gets_no_delta_and_is_never_vacuous() {
        let mut unavailable = row("missing-lever", Some("base"), 0, &[]);
        unavailable.unavailable_in_veld = Some("no code path".into());
        let mut rows = vec![row("base", None, 0, &[0.2, 0.4]), unavailable];
        resolve_against_references(&mut rows).unwrap();
        assert!(rows[1].delta_recall_at_10.is_none());
        assert!(!rows[1].vacuous_vs_reference);
    }

    #[test]
    fn an_unknown_reference_is_an_error_not_a_skipped_row() {
        let mut rows = vec![
            row("base", None, 1, &[0.2]),
            row("orphan", Some("missing"), 2, &[0.3]),
        ];
        let err = resolve_against_references(&mut rows).unwrap_err();
        assert!(err.to_string().contains("missing"), "{err}");
    }
}

/// Whole-word / phrase containment: does `name` appear in `text_lc` at word boundaries?
/// Both args must already be lowercased. Avoids spurious substring hits (e.g. "al" in "also").
fn phrase_in_text(name: &str, text_lc: &str) -> bool {
    if name.len() < 2 {
        return false;
    }
    let bytes = text_lc.as_bytes();
    let mut start = 0;
    while let Some(rel) = text_lc[start..].find(name) {
        let pos = start + rel;
        let before_ok = pos == 0 || !bytes[pos - 1].is_ascii_alphanumeric();
        let end = pos + name.len();
        let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        start = pos + 1;
        if start >= text_lc.len() {
            break;
        }
    }
    false
}

/// Query→graph entity-linking accuracy diagnostic. For each query: NER → mentions → link via
/// `find_entity_by_name` → linked graph nodes; compared against a lexical ground truth (graph
/// entity names that appear as phrases in the query). Reports link precision/recall, no-seed
/// rate, and extraction counts — turning "is NER the bottleneck?" into a number. Low recall ⇒
/// the traversal/graph starts from wrong-or-missing seeds and NER/linking is the ceiling.
pub fn analyze_linking(inputs: &RunInputs) -> Result<LinkingReport> {
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    let manager = build_manager(&inputs.storage_path)?;
    let _id_map = ingest_corpus(&manager, &corpus)?;
    let ner = manager.get_neural_ner();
    let graph = manager.get_user_graph(EVAL_USER)?;

    // Lexical ground truth: all graph entity names (lowercased) — the entities the corpus built.
    let entity_names_lc: Vec<String> = {
        let g = graph.read();
        g.get_all_entities()
            .unwrap_or_default()
            .into_iter()
            .map(|e| e.name.to_lowercase())
            .filter(|n| n.len() >= 2)
            .collect()
    };

    #[derive(Default)]
    struct Acc {
        cases: usize,
        ner_sum: usize,
        linked_sum: usize,
        no_seed: usize,
        lexical_sum: usize,
        tp_sum: usize,
        linked_total: usize,
        lexical_total: usize,
    }
    let mut overall = Acc::default();
    let mut by_cat: BTreeMap<String, Acc> = BTreeMap::new();

    for case in &cases {
        let cat = category_name(case.category).to_string();
        let query_lc = case.query.to_lowercase();

        let mentions: Vec<String> = match ner.extract(&case.query) {
            Ok(es) => es.into_iter().map(|e| e.text).collect(),
            Err(_) => Vec::new(),
        };
        let mut linked: HashSet<String> = HashSet::new();
        {
            let g = graph.read();
            for m in &mentions {
                if let Ok(Some(ent)) = g.find_entity_by_name(m) {
                    linked.insert(ent.name.to_lowercase());
                }
            }
        }
        let lexical: HashSet<String> = entity_names_lc
            .iter()
            .filter(|n| phrase_in_text(n, &query_lc))
            .cloned()
            .collect();
        let tp = linked.iter().filter(|n| lexical.contains(*n)).count();

        let (m, l, lex, no_seed) = (
            mentions.len(),
            linked.len(),
            lexical.len(),
            linked.is_empty(),
        );
        let bump = |acc: &mut Acc| {
            acc.cases += 1;
            acc.ner_sum += m;
            acc.linked_sum += l;
            if no_seed {
                acc.no_seed += 1;
            }
            acc.lexical_sum += lex;
            acc.tp_sum += tp;
            acc.linked_total += l;
            acc.lexical_total += lex;
        };
        bump(&mut overall);
        bump(by_cat.entry(cat).or_default());
    }

    let to_row = |cat: &str, a: &Acc| -> LinkingRow {
        let n = a.cases.max(1) as f64;
        LinkingRow {
            category: cat.to_string(),
            cases: a.cases,
            ner_mean: a.ner_sum as f64 / n,
            linked_mean: a.linked_sum as f64 / n,
            no_seed_pct: 100.0 * a.no_seed as f64 / n,
            lexical_mean: a.lexical_sum as f64 / n,
            link_precision: if a.linked_total == 0 {
                0.0
            } else {
                a.tp_sum as f64 / a.linked_total as f64
            },
            link_recall: if a.lexical_total == 0 {
                0.0
            } else {
                a.tp_sum as f64 / a.lexical_total as f64
            },
        }
    };

    let by_category: BTreeMap<String, LinkingRow> = by_cat
        .iter()
        .map(|(c, a)| (c.clone(), to_row(c, a)))
        .collect();
    Ok(LinkingReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        overall: to_row("ALL", &overall),
        by_category,
    })
}

/// Per-stage gold-rank funnel diagnostic. Runs the suite under Full mode with the gold ids
/// armed, recording at each pipeline-stage boundary (graph → vector → fusion → final) whether
/// the gold is present and its best rank. Aggregates the per-stage drop-off so the step that
/// loses reachable gold is LOCATED, not inferred (e.g. present after `graph`, absent after
/// `fusion` ⇒ RRF is burying it).
/// Apply the optional fast-read subsamples to a loaded `(cases, corpus)`.
/// `SHODH_MAX_CASES` strides the query set down (corpus stays full → distractor
/// difficulty unchanged); then `SHODH_MAX_CORPUS` shrinks the distractor pool while
/// ALWAYS retaining every gold item referenced by the surviving cases (directional
/// only — recall inflates). Both unset → returned unchanged (full scale). Shared so
/// the recall run and the funnel honor the SAME knobs; without it the funnel ran
/// full-scale (1531 × 5882) and timed out. Mirrors the inline blocks in
/// `run_smoke_suite_with_ranks` (dedupe those into this helper is a follow-up).
fn apply_eval_caps(
    cases: Vec<SmokeCase>,
    corpus: Vec<CorpusItem>,
) -> (Vec<SmokeCase>, Vec<CorpusItem>) {
    let cases = match std::env::var("SHODH_MAX_CASES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n < cases.len())
    {
        Some(n) => {
            let total = cases.len();
            let stride = (total / n).max(1);
            let sampled: Vec<_> = cases.into_iter().step_by(stride).take(n).collect();
            tracing::info!(
                "SHODH_MAX_CASES={n}: subsampled {} of {} cases (stride {})",
                sampled.len(),
                total,
                stride
            );
            sampled
        }
        None => cases,
    };
    let corpus = match std::env::var("SHODH_MAX_CORPUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n < corpus.len())
    {
        Some(n) => {
            let gold: std::collections::HashSet<&str> = cases
                .iter()
                .flat_map(|c| c.relevant.iter().map(|r| r.corpus_item_id.as_str()))
                .collect();
            let total = corpus.len();
            let gold_count = corpus
                .iter()
                .filter(|it| gold.contains(it.id.as_str()))
                .count();
            let keep_distractors = n.saturating_sub(gold_count);
            let distractor_total = total - gold_count;
            let stride = if keep_distractors == 0 {
                usize::MAX
            } else {
                (distractor_total / keep_distractors).max(1)
            };
            let mut seen_distractor = 0usize;
            let mut kept_distractors = 0usize;
            let reduced: Vec<_> = corpus
                .into_iter()
                .filter(|it| {
                    if gold.contains(it.id.as_str()) {
                        return true;
                    }
                    let take = kept_distractors < keep_distractors && seen_distractor % stride == 0;
                    seen_distractor += 1;
                    if take {
                        kept_distractors += 1;
                    }
                    take
                })
                .collect();
            tracing::info!(
                "SHODH_MAX_CORPUS={n}: reduced corpus {} → {} ({} gold + {} distractors)",
                total,
                reduced.len(),
                gold_count,
                kept_distractors
            );
            reduced
        }
        None => corpus,
    };
    (cases, corpus)
}

pub fn analyze_funnel(inputs: &RunInputs) -> Result<FunnelReport> {
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    // Honor the same fast-read caps as the recall run, or the funnel runs full-scale
    // (1531 cases × 5882 corpus) per arm and times out. Default unset → full scale.
    let (cases, corpus) = apply_eval_caps(cases, corpus);

    let manager = build_manager(&inputs.storage_path)?;
    let id_map = ingest_corpus(&manager, &corpus)?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // Same ingest→query boundary lever as run_one_pass — the funnel is a separate
    // pass and silently skipped the rebuild (run 27262844551 funnel-arm gate).
    if std::env::var("SHODH_VAMANA_QUALITY_REBUILD")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        system
            .read()
            .force_vector_quality_rebuild()
            .context("vamana quality rebuild at the funnel ingest→query boundary")?;
    }

    let diag_k = std::env::var("RECALL_DIAG_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|k| *k >= SMOKE_K)
        .unwrap_or(SMOKE_K);

    const STAGES: [&str; 5] = ["graph", "vector", "hybrid", "fusion", "final"];

    #[derive(Default, Clone)]
    struct Acc {
        seen: usize,
        present: usize,
        top_k: usize,
        rank_sum: usize,
    }
    impl Acc {
        fn record(&mut self, rank: Option<usize>) {
            self.seen += 1;
            if let Some(r) = rank {
                self.present += 1;
                self.rank_sum += r;
                if r < SMOKE_K {
                    self.top_k += 1;
                }
            }
        }
    }

    let mut overall: BTreeMap<String, Acc> = BTreeMap::new();
    let mut by_cat: BTreeMap<String, BTreeMap<String, Acc>> = BTreeMap::new();
    let mut case_count = 0usize;

    for case in &cases {
        let gold: HashSet<MemoryId> = case
            .relevant
            .iter()
            .filter_map(|r| id_map.get(&r.corpus_item_id).copied())
            .map(MemoryId)
            .collect();
        if gold.is_empty() {
            continue;
        }
        case_count += 1;
        let cat = category_name(case.category).to_string();

        let query = Query {
            query_text: Some(case.query.clone()),
            max_results: diag_k,
            layers: Some(crate::memory::types::LayerMode::Full),
            ..Default::default()
        };

        crate::memory::gold_funnel::begin(gold.clone());
        let _ = system.read().recall(&query);
        let funnel = crate::memory::gold_funnel::take().unwrap_or_default();

        for (stage, rank) in &funnel {
            overall.entry(stage.clone()).or_default().record(*rank);
            by_cat
                .entry(cat.clone())
                .or_default()
                .entry(stage.clone())
                .or_default()
                .record(*rank);
        }
    }

    let to_rows = |m: &BTreeMap<String, Acc>| -> Vec<FunnelStageRow> {
        STAGES
            .iter()
            .filter_map(|s| m.get(*s).map(|a| (*s, a)))
            .map(|(s, a)| {
                let seen = a.seen.max(1) as f64;
                FunnelStageRow {
                    stage: s.to_string(),
                    present_pct: 100.0 * a.present as f64 / seen,
                    top10_pct: 100.0 * a.top_k as f64 / seen,
                    mean_rank_when_present: if a.present == 0 {
                        0.0
                    } else {
                        a.rank_sum as f64 / a.present as f64
                    },
                }
            })
            .collect()
    };

    let by_category: BTreeMap<String, Vec<FunnelStageRow>> = by_cat
        .iter()
        .map(|(c, m)| (c.clone(), to_rows(m)))
        .collect();

    Ok(FunnelReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        case_count,
        overall: to_rows(&overall),
        by_category,
    })
}

pub fn analyze_graph_reachability(inputs: &RunInputs) -> Result<ReachabilityReport> {
    pin_harness_threads();
    const MAX_HOPS: usize = 3;
    // Safety valve against hub-entity blowup on dense graphs. Far above any
    // LoCoMo component size, so it does not bias the result in practice.
    const MAX_VISITED_ENTITIES: usize = 50_000;

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    // Respect the same SHODH_MAX_CASES / SHODH_MAX_CORPUS caps as the recall and
    // funnel paths, so the reachability graph matches the graph those runs
    // measured (entity counts comparable to the 51%-present funnel) and ingest
    // time is bounded. Unset env → full corpus, identical to prior behaviour.
    let (cases, corpus) = apply_eval_caps(cases, corpus);

    let manager = build_manager(&inputs.storage_path)?;
    let id_map = ingest_corpus(&manager, &corpus)?;
    let ner = manager.get_neural_ner();
    let graph = manager.get_user_graph(EVAL_USER)?;
    // Mirror the ingest-side concept gate on the query side so the test is
    // symmetric: concept nodes added to the corpus are only seedable if the
    // query also extracts concepts.
    let concept_query_seeding = std::env::var("SHODH_CONCEPT_ENTITIES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let concept_seed_df_max: Option<usize> = std::env::var("SHODH_CONCEPT_SEED_DF_MAX")
        .ok()
        .and_then(|s| s.parse().ok());
    let keyword_extractor = manager.get_keyword_extractor();

    // Degree distribution of the built graph — the direct scoreboard for
    // anti-hub construction tuning (only visible at corpus scale, where the
    // speaker hubs actually form).
    let graph_structure = {
        const HUB_REPORT_THRESHOLD: usize = 50;
        let g = graph.read();
        let entities = g.get_all_entities().unwrap_or_default();
        let mut degrees: Vec<usize> = entities
            .iter()
            .map(|e| {
                g.get_entity_relationships(&e.uuid)
                    .map(|r| r.len())
                    .unwrap_or(0)
            })
            .collect();
        degrees.sort_unstable_by(|a, b| b.cmp(a));
        let total_entities = degrees.len();
        let degree_sum: usize = degrees.iter().sum();

        // Relation-type histogram over STORED edges. Walked from the entity side
        // and de-duplicated by edge uuid, because get_entity_relationships
        // returns an edge once from each endpoint it is reachable from and
        // counting incidences would double most of them.
        //
        // SYMMETRIC types carry no order, so a path hop through one contributes
        // nothing a composition could be non-commutative about. They are also
        // the ones fully determined by entity->episode incidence, so this split
        // is the same line as "stored vs derivable".
        const SYMMETRIC: [&str; 7] = [
            "CoOccurs",
            "RelatedTo",
            "AssociatedWith",
            "CoRetrieved",
            "WorksWith",
            "Knows",
            "AlternativeTo",
        ];
        let mut relation_types: std::collections::BTreeMap<String, usize> = Default::default();
        let mut seen_edges: std::collections::HashSet<uuid::Uuid> = Default::default();
        for e in &entities {
            for edge in g.get_entity_relationships(&e.uuid).unwrap_or_default() {
                if !seen_edges.insert(edge.uuid) {
                    continue;
                }
                *relation_types
                    .entry(format!("{:?}", edge.relation_type))
                    .or_insert(0) += 1;
            }
        }
        let symmetric_edges: usize = relation_types
            .iter()
            .filter(|(name, _)| SYMMETRIC.contains(&name.as_str()))
            .map(|(_, n)| *n)
            .sum();
        let typed_edges: usize = relation_types.values().sum::<usize>() - symmetric_edges;

        // Entity labels. A label-pair typing rule can be correct and enabled and
        // still never fire, because the labels it matches are not produced.
        let mut entity_labels: std::collections::BTreeMap<String, usize> = Default::default();
        for e in &entities {
            if e.labels.is_empty() {
                *entity_labels.entry("<unlabelled>".to_string()).or_insert(0) += 1;
            }
            for l in &e.labels {
                *entity_labels.entry(format!("{l:?}")).or_insert(0) += 1;
            }
        }

        // Which labels carry the CONNECTIVITY, not just which labels exist.
        // A label that is a small share of nodes and a large share of
        // incidences is a node type doing an attribute's job: everyone mentions
        // "last weekend", so minting it wires every such memory into one star
        // that no query would ever traverse deliberately.
        //
        // Degrees are read post-cap, so a saturated entity reports the cap and
        // its discarded edges are invisible here. That censors the mean
        // downward; the top_hubs list below is where saturation is legible.
        let degree_of = |e: &crate::graph_memory::EntityNode| {
            g.get_entity_relationships(&e.uuid)
                .map(|r| r.len())
                .unwrap_or(0)
        };
        let mut label_degrees: std::collections::BTreeMap<String, (usize, usize, usize)> =
            Default::default();
        for e in &entities {
            let d = degree_of(e);
            let names: Vec<String> = if e.labels.is_empty() {
                vec!["<unlabelled>".to_string()]
            } else {
                e.labels.iter().map(|l| format!("{l:?}")).collect()
            };
            // Multi-label entities count under each of their labels, so these
            // sums exceed the totals. Per-label shares stay meaningful; a
            // partition would not.
            for name in names {
                let slot = label_degrees.entry(name).or_insert((0, 0, 0));
                slot.0 += 1;
                slot.1 += d;
                slot.2 = slot.2.max(d);
            }
        }

        // Identities of the top hubs. Ties broken by name so the list is
        // reproducible across ingests -- without it, equal-degree entities
        // would be ordered by whatever the entity scan happened to yield.
        let mut ranked: Vec<(usize, &crate::graph_memory::EntityNode)> =
            entities.iter().map(|e| (degree_of(e), e)).collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        let top_hubs: Vec<(String, String, usize)> = ranked
            .iter()
            .take(25)
            .map(|(d, e)| {
                let labels = if e.labels.is_empty() {
                    "<unlabelled>".to_string()
                } else {
                    e.labels
                        .iter()
                        .map(|l| format!("{l:?}"))
                        .collect::<Vec<_>>()
                        .join("+")
                };
                (e.name.clone(), labels, *d)
            })
            .collect();

        // Undirected adjacency, built once and reused for every component pass.
        let index: std::collections::HashMap<uuid::Uuid, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.uuid, i))
            .collect();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); entities.len()];
        let mut typed_adj: Vec<Vec<usize>> = vec![Vec::new(); entities.len()];
        let mut counted: std::collections::HashSet<uuid::Uuid> = Default::default();
        for e in &entities {
            for edge in g.get_entity_relationships(&e.uuid).unwrap_or_default() {
                if !counted.insert(edge.uuid) {
                    continue;
                }
                let (Some(&a), Some(&b)) =
                    (index.get(&edge.from_entity), index.get(&edge.to_entity))
                else {
                    continue;
                };
                adj[a].push(b);
                adj[b].push(a);
                if !SYMMETRIC.contains(&format!("{:?}", edge.relation_type).as_str()) {
                    typed_adj[a].push(b);
                    typed_adj[b].push(a);
                }
            }
        }

        // Components by BFS over an adjacency list, with a set of deleted
        // vertices. Iterative: a hub-heavy graph would blow a recursive stack.
        let components = |adj: &Vec<Vec<usize>>, removed: &std::collections::HashSet<usize>| {
            let n = adj.len();
            let mut seen = vec![false; n];
            let (mut count, mut largest) = (0usize, 0usize);
            for start in 0..n {
                if seen[start] || removed.contains(&start) {
                    continue;
                }
                count += 1;
                let mut size = 0usize;
                let mut stack = vec![start];
                seen[start] = true;
                while let Some(v) = stack.pop() {
                    size += 1;
                    for &w in &adj[v] {
                        if !seen[w] && !removed.contains(&w) {
                            seen[w] = true;
                            stack.push(w);
                        }
                    }
                }
                largest = largest.max(size);
            }
            (count, largest)
        };

        let n_ent = entities.len().max(1);
        let none: std::collections::HashSet<usize> = Default::default();
        let (c_all, l_all) = components(&adj, &none);
        let components_all = (c_all, l_all as f64 / n_ent as f64);

        // Delete the top-N by degree. A k-connected graph survives k-1
        // deletions; if this shatters early, connectivity is hub-carried.
        let mut by_degree: Vec<usize> = (0..entities.len()).collect();
        by_degree.sort_unstable_by(|&a, &b| adj[b].len().cmp(&adj[a].len()).then(a.cmp(&b)));
        let mut components_after_hub_removal = Vec::new();
        for &k in &[1usize, 5, 10, 25] {
            if k > by_degree.len() {
                break;
            }
            let removed: std::collections::HashSet<usize> =
                by_degree.iter().take(k).copied().collect();
            let (c, l) = components(&adj, &removed);
            let denom = (entities.len() - k).max(1);
            components_after_hub_removal.push((k, c, l as f64 / denom as f64));
        }

        let (c_typed, l_typed) = components(&typed_adj, &none);
        let isolated_typed = typed_adj.iter().filter(|a| a.is_empty()).count();
        let typed_components = (c_typed, l_typed as f64 / n_ent as f64, isolated_typed);

        GraphStructure {
            relation_types,
            typed_edges,
            symmetric_edges,
            entity_labels,
            label_degrees,
            top_hubs,
            components_all,
            components_after_hub_removal,
            typed_components,
            total_entities,
            total_edges: degree_sum / 2,
            max_degree: degrees.first().copied().unwrap_or(0),
            mean_degree: if total_entities == 0 {
                0.0
            } else {
                degree_sum as f64 / total_entities as f64
            },
            // None: no hub cap in this build. Some(n) the day one lands.
            hub_saturation_skipped_edges: None,
            hub_count: degrees
                .iter()
                .filter(|d| **d > HUB_REPORT_THRESHOLD)
                .count(),
            hub_threshold: HUB_REPORT_THRESHOLD,
            top_degrees: degrees.iter().take(8).copied().collect(),
        }
    };

    let mut by_category: BTreeMap<String, ReachabilityCategory> = BTreeMap::new();
    let mut overall = ReachabilityCategory::default();

    for case in &cases {
        let cat = by_category
            .entry(category_name(case.category).to_string())
            .or_default();
        cat.cases += 1;
        overall.cases += 1;

        // Gold memory uuids for this case.
        let gold: HashSet<Uuid> = case
            .relevant
            .iter()
            .filter_map(|r| id_map.get(&r.corpus_item_id).copied())
            .collect();
        cat.gold_total += gold.len();
        overall.gold_total += gold.len();

        // Seed entities: NER over the query text, resolved to graph nodes by
        // name (same name-keyed lookup the ingest path builds them under).
        let mut seed_names: Vec<String> = match ner.extract(&case.query) {
            Ok(es) => es.into_iter().map(|e| e.text).collect(),
            Err(_) => Vec::new(),
        };
        // Symmetric concept seeding: when concept entities are added to the
        // corpus graph (SHODH_CONCEPT_ENTITIES), the QUERY must also resolve to
        // them or those nodes are unreachable as seeds. Extract YAKE keyphrases
        // from the query (mirroring the ingest-side concept extraction) so a
        // concept→concept seed can form. Without this the corpus concept nodes
        // exist but no query ever lands on them.
        if concept_query_seeding {
            for kw in keyword_extractor.extract(&case.query).into_iter().take(8) {
                let name = kw.text.trim().to_string();
                if name.len() >= 3 {
                    seed_names.push(name);
                }
            }
        }
        let g = graph.read();
        let mut seed_uuids: HashSet<Uuid> = HashSet::new();
        for name in &seed_names {
            if let Ok(Some(ent)) = g.find_entity_by_name(name) {
                // Mirror the live IDF gate (mod.rs): skip generic concept seeds
                // whose corpus mention_count exceeds SHODH_CONCEPT_SEED_DF_MAX.
                if let Some(df_max) = concept_seed_df_max {
                    let is_concept = matches!(
                        ent.labels.first(),
                        Some(crate::graph_memory::EntityLabel::Other(_))
                    );
                    if is_concept && ent.mention_count > df_max {
                        continue;
                    }
                }
                seed_uuids.insert(ent.uuid);
            }
        }
        if seed_uuids.is_empty() {
            cat.cases_no_seed += 1;
            overall.cases_no_seed += 1;
            cat.unreachable += gold.len();
            overall.unreachable += gold.len();
            continue;
        }
        if seed_uuids.len() >= 2 {
            cat.cases_multi_seed += 1;
            overall.cases_multi_seed += 1;
        }

        // BFS over entity→entity edges; record the min hop at which each gold
        // episode is first attached to a reached entity.
        let mut visited: HashSet<Uuid> = seed_uuids.clone();
        let mut frontier: Vec<Uuid> = seed_uuids.iter().copied().collect();
        let mut gold_min_hop: HashMap<Uuid, usize> = HashMap::new();
        // Per-gold set of distinct query seeds directly (1-hop) attached — the
        // G5 multi-seed discrimination signal.
        let mut gold_seed_cov: HashMap<Uuid, HashSet<Uuid>> = HashMap::new();
        for hop in 1..=MAX_HOPS {
            for ent in &frontier {
                // At hop 1 the frontier is exactly the query seeds.
                let ent_is_seed = hop == 1;
                if let Ok(eps) = g.get_episodes_by_entity(ent) {
                    for ep in eps {
                        if gold.contains(&ep.uuid) {
                            gold_min_hop.entry(ep.uuid).or_insert(hop);
                            if ent_is_seed {
                                gold_seed_cov.entry(ep.uuid).or_default().insert(*ent);
                            }
                        }
                    }
                }
            }
            if hop == MAX_HOPS || gold_min_hop.len() == gold.len() {
                break;
            }
            let mut next: Vec<Uuid> = Vec::new();
            for ent in &frontier {
                if let Ok(edges) = g.get_entity_relationships(ent) {
                    for e in edges {
                        for nbr in [e.from_entity, e.to_entity] {
                            if nbr != *ent && visited.insert(nbr) {
                                next.push(nbr);
                            }
                        }
                    }
                }
                if visited.len() >= MAX_VISITED_ENTITIES {
                    break;
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        drop(g);

        for gid in &gold {
            match gold_min_hop.get(gid).copied() {
                Some(1) => {
                    cat.reachable_within_1 += 1;
                    cat.reachable_within_2 += 1;
                    cat.reachable_within_3 += 1;
                    overall.reachable_within_1 += 1;
                    overall.reachable_within_2 += 1;
                    overall.reachable_within_3 += 1;
                }
                Some(2) => {
                    cat.reachable_within_2 += 1;
                    cat.reachable_within_3 += 1;
                    overall.reachable_within_2 += 1;
                    overall.reachable_within_3 += 1;
                }
                Some(_) => {
                    cat.reachable_within_3 += 1;
                    overall.reachable_within_3 += 1;
                }
                None => {
                    cat.unreachable += 1;
                    overall.unreachable += 1;
                }
            }
            // G5 signal availability: is this gold directly attached to ≥2
            // distinct query seeds?
            if gold_seed_cov.get(gid).map(|s| s.len()).unwrap_or(0) >= 2 {
                cat.gold_multi_seed += 1;
                overall.gold_multi_seed += 1;
            }
        }
    }

    Ok(ReachabilityReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        max_hops: MAX_HOPS,
        overall,
        by_category,
        graph: graph_structure,
    })
}

/// Learning-curve diagnostic — see [`LearningCurveReport`]. The flagship test of
/// "smarter with use": does a memory become EASIER to recall as it is used?
///
/// Protocol per case (only those whose cold gold-rank lands in the headroom band
/// — already-top-1 has nowhere to climb, very-deep gold won't surface in a few
/// cycles): recall the query, then apply `Helpful` feedback to the gold via
/// `reinforce_recall` (strengthens the gold's graph edges, importance-gated), and
/// re-measure the gold's rank + score. Repeat `cycles` times. `recall` itself
/// also auto-reinforces every returned memory (access-count + co-activation), so
/// distractors are reinforced too — only the *extra* `Helpful` signal on the gold
/// breaks the symmetry. A genuine associative memory shows the gold's rank fall
/// and score rise across cycles; a static retriever stays flat. Full mode only
/// (Hebbian/LTP writes are gated to `Full`).
/// One reinforcement-outcome arm: fresh ingest, then for each headroom case run
/// `cyc` rounds of recall + `outcome` feedback on the gold, tracking the gold's
/// rank/score per cycle.
fn run_learning_arm(
    storage_path: &Path,
    corpus: &[CorpusItem],
    cases: &[SmokeCase],
    cyc: usize,
    outcome: RetrievalOutcome,
    outcome_name: &str,
) -> Result<LearningCurveArm> {
    const TRACK_K: usize = 50;
    const MIN_COLD_RANK: usize = 2;
    const MAX_COLD_RANK: usize = 30;

    let manager = build_manager(storage_path)?;
    let id_map = ingest_corpus(&manager, corpus)?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // Recall the query (Full mode → reinforcement active) and return the ranked
    // (uuid, score) list, wide enough to observe the gold's rank.
    let recall_ranked = |q: &str| -> Vec<(Uuid, f32)> {
        let query = Query {
            query_text: Some(q.to_string()),
            max_results: TRACK_K,
            layers: Some(LayerMode::Full),
            ..Default::default()
        };
        match system.read().recall(&query) {
            Ok(mems) => mems
                .iter()
                .map(|m| (m.id.0, m.score.unwrap_or(0.0)))
                .collect(),
            Err(_) => Vec::new(),
        }
    };

    let mut rank_sums = vec![0.0f64; cyc + 1];
    let mut score_sums = vec![0.0f64; cyc + 1];
    let mut tracked = 0usize;
    let (mut improved, mut worsened, mut unchanged) = (0usize, 0usize, 0usize);
    let mut rank_delta_sum = 0.0f64;
    let mut score_delta_sum = 0.0f64;

    for case in cases {
        let gold: HashSet<Uuid> = case
            .relevant
            .iter()
            .filter_map(|r| id_map.get(&r.corpus_item_id).copied())
            .collect();
        if gold.is_empty() {
            continue;
        }
        // Best-ranked gold's (rank, score) in a ranked list; 1-based.
        let gold_rank = |ranked: &[(Uuid, f32)]| -> Option<(usize, f32)> {
            ranked
                .iter()
                .enumerate()
                .find(|(_, (u, _))| gold.contains(u))
                .map(|(i, (_, s))| (i + 1, *s))
        };

        let cold = recall_ranked(&case.query);
        let (r0, s0) = match gold_rank(&cold) {
            Some(x) => x,
            None => continue,
        };
        if !(MIN_COLD_RANK..=MAX_COLD_RANK).contains(&r0) {
            continue;
        }

        tracked += 1;
        rank_sums[0] += r0 as f64;
        score_sums[0] += s0 as f64;

        let gold_ids: Vec<MemoryId> = gold.iter().map(|u| MemoryId(*u)).collect();
        let (mut last_rank, mut last_score) = (r0, s0);
        for c in 1..=cyc {
            let _ = system.read().reinforce_recall(&gold_ids, outcome);
            let ranked = recall_ranked(&case.query);
            let (rc, sc) = gold_rank(&ranked).unwrap_or((TRACK_K + 1, 0.0));
            rank_sums[c] += rc as f64;
            score_sums[c] += sc as f64;
            last_rank = rc;
            last_score = sc;
        }

        rank_delta_sum += last_rank as f64 - r0 as f64;
        score_delta_sum += (last_score - s0) as f64;
        match last_rank.cmp(&r0) {
            std::cmp::Ordering::Less => improved += 1,
            std::cmp::Ordering::Greater => worsened += 1,
            std::cmp::Ordering::Equal => unchanged += 1,
        }
    }

    let n = tracked.max(1) as f64;
    Ok(LearningCurveArm {
        outcome: outcome_name.to_string(),
        tracked_cases: tracked,
        mean_rank_by_cycle: rank_sums.iter().map(|s| s / n).collect(),
        mean_score_by_cycle: score_sums.iter().map(|s| s / n).collect(),
        improved,
        worsened,
        unchanged,
        mean_rank_delta: rank_delta_sum / n,
        mean_score_delta: score_delta_sum / n,
    })
}

/// Learning-curve diagnostic — see [`LearningCurveReport`]. Runs three arms
/// (Helpful / Neutral / Misleading), each on a FRESH ingest, to expose the
/// reward GRADIENT: a real reward-modulated memory pushes the gold UP under
/// Helpful, DOWN under Misleading, and flat under Neutral. Amplify the reward
/// learning rate via `SHODH_REWARD_LR_MULT` and re-run to see if a stronger
/// reward moves rank, not just score.
pub fn analyze_learning_curve(inputs: &RunInputs, cycles: usize) -> Result<LearningCurveReport> {
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    let cyc = cycles.max(1);
    let arms_spec = [
        ("Helpful", RetrievalOutcome::Helpful),
        ("Neutral", RetrievalOutcome::Neutral),
        ("Misleading", RetrievalOutcome::Misleading),
    ];
    let mut arms = Vec::with_capacity(arms_spec.len());
    for (name, outcome) in arms_spec {
        // Fresh storage per arm — reinforcement mutates state; arms must not bleed.
        let arm_storage = inputs.storage_path.join(name.to_lowercase());
        let arm = run_learning_arm(&arm_storage, &corpus, &cases, cyc, outcome, name)
            .with_context(|| format!("learning-curve arm {name}"))?;
        arms.push(arm);
    }

    Ok(LearningCurveReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        cycles: cyc,
        arms,
    })
}

/// Map a corpus item's `memory_type` string to an `ExperienceType` variant.
///
/// Unknown types fall back to `Observation`, which is the most neutral
/// option in the existing taxonomy. This keeps fixture authoring lenient
/// while still flowing real type information through the pipeline when it
/// is recognised.
fn experience_type_for(s: &str) -> ExperienceType {
    match s.to_ascii_lowercase().as_str() {
        "decision" => ExperienceType::Decision,
        "task" => ExperienceType::Task,
        "conversation" => ExperienceType::Conversation,
        "error" => ExperienceType::Error,
        "learning" => ExperienceType::Learning,
        "discovery" => ExperienceType::Discovery,
        "pattern" => ExperienceType::Pattern,
        "context" => ExperienceType::Context,
        "event" | "reference" | "observation" => ExperienceType::Observation,
        _ => ExperienceType::Observation,
    }
}

/// Build the graded relevance map for a single case using the ingest id map.
///
/// Items that did not make it into `id_map` are silently skipped here; the
/// caller logs a `case` failure if the resulting map is empty.
fn build_relevance_map(case: &SmokeCase, id_map: &HashMap<String, Uuid>) -> HashMap<Uuid, f32> {
    let mut out = HashMap::with_capacity(case.relevant.len());
    for r in &case.relevant {
        if let Some(uuid) = id_map.get(&r.corpus_item_id) {
            out.insert(*uuid, r.grade as f32);
        }
    }
    out
}

fn category_name(c: SmokeCategory) -> &'static str {
    match c {
        SmokeCategory::Decision => "decision",
        SmokeCategory::Code => "code",
        SmokeCategory::Temporal => "temporal",
        SmokeCategory::Entity => "entity",
        SmokeCategory::MultiHop => "multi_hop",
        SmokeCategory::Negation => "negation",
        SmokeCategory::SingleHop => "single_hop",
        SmokeCategory::OpenDomain => "open_domain",
        SmokeCategory::DirObject => "dir_object",
        SmokeCategory::DirSubject => "dir_subject",
        SmokeCategory::DirPassive => "dir_passive",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_storage_dir(label: &str) -> PathBuf {
        let id = Uuid::new_v4().simple().to_string();
        std::env::temp_dir().join(format!("shodh-recall-{label}-{id}"))
    }

    /// Lineage repro (substrate diagnosis 2026-06-10): root-cause P@1 has been
    /// 0.0 through every fix, and the instrumented CI run produced ZERO edge
    /// provenance lines and ZERO causal-funnel lines — hypothesis: the chain
    /// entities never become graph structure at all. This test pushes one
    /// lineage chain through the REAL harness ingest (ingest_corpus →
    /// process_experience_into_graph) and asserts stage by stage:
    ///   1. the chain entities exist in the graph by name
    ///   2. edges connect them
    ///   3. at least one connecting edge is typed CAUSAL (the walk's food)
    /// Whichever assert fails localizes the lineage zero.
    #[test]
    fn lineage_chain_builds_causal_graph_structure() {
        let dir = unique_storage_dir("lineage-repro");
        let manager = build_manager(&dir).expect("manager");
        let (corpus, _cases) = crate::recall_harness::lineage_harness::generate_lineage_fixtures(1);
        ingest_corpus(&manager, &corpus).expect("ingest");

        let graph = manager.get_user_graph(EVAL_USER).expect("graph");
        let g = graph.read();

        // Chain 0 entities: "the Vornak incident" → "the Meslin incident" →
        // "the Caldor incident" (event(0..3)).
        let names = [
            "the Vornak incident",
            "the Meslin incident",
            "the Caldor incident",
        ];
        let mut uuids = Vec::new();
        for name in names {
            let found = g.find_entity_by_name(name).expect("lookup");
            assert!(
                found.is_some(),
                "STAGE 1 FAIL: chain entity '{name}' missing from graph — \
                 entity creation is the lineage zero"
            );
            uuids.push(found.unwrap().uuid);
        }

        // Edges between consecutive chain entities?
        for w in uuids.windows(2) {
            let edges = g
                .get_entity_relationships_limited(&w[0], Some(64))
                .expect("edges");
            let connecting: Vec<_> = edges
                .iter()
                .filter(|e| {
                    (e.from_entity == w[0] && e.to_entity == w[1])
                        || (e.from_entity == w[1] && e.to_entity == w[0])
                })
                .collect();
            assert!(
                !connecting.is_empty(),
                "STAGE 2 FAIL: no edge between consecutive chain entities — \
                 pair-loop/edge creation is the lineage zero"
            );
            let causal = connecting.iter().any(|e| e.relation_type.is_causal());
            assert!(
                causal,
                "STAGE 3 FAIL: edges exist but none is causal (types: {:?}) — \
                 the typing chain is the lineage zero",
                connecting
                    .iter()
                    .map(|e| e.relation_type.as_str())
                    .collect::<Vec<_>>()
            );
        }

        // STAGE 4: the backward origin walk itself. From the observed effect
        // (Caldor), trace_causal_origins must return the root (Vornak).
        let origins = g.trace_causal_origins(&[uuids[2]], 8).expect("origin walk");
        assert!(
            origins.iter().any(|(id, _)| *id == uuids[0]),
            "STAGE 4 FAIL: backward walk from the effect did not reach the \
             root cause (origins: {} found) — edge direction or walk is the \
             lineage zero",
            origins.len()
        );
        drop(g);

        // STAGE 5: the full recall path. The root memory (which shares NO token
        // with the query other than what the walk provides) must appear in the
        // results for the root-cause query.
        let system = manager.get_user_memory(EVAL_USER).expect("system");
        let query = crate::memory::types::Query {
            query_text: Some("What was the earliest origin behind the Caldor incident?".into()),
            max_results: 10,
            layers: Some(LayerMode::Full),
            ..Default::default()
        };
        let results = system.read().recall(&query).expect("recall");
        let texts: Vec<String> = results
            .iter()
            .map(|m| m.experience.content.clone())
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("Vornak")),
            "STAGE 5 FAIL: walk works but the root memory is not in the top-10 \
             — the origins-USAGE path (BM25 cue expansion) is the lineage zero. \
             Retrieved: {texts:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Ingest must type a causal sentence's pair as causal and leave non-causal
    /// co-mentions untyped. The causal item is the positive control: without it a
    /// typer that never fires would pass the negative half.
    #[test]
    fn ingest_types_only_causal_sentences_as_causal() {
        let dir = unique_storage_dir("causal-typing-fp");
        let manager = build_manager(&dir).expect("manager");
        let base = chrono::Utc::now();
        let items = [
            (
                "the Arvel incident",
                "the Brisk incident",
                "{a} set {b} in motion.",
            ),
            (
                "the Corran incident",
                "the Dellow incident",
                "{a} and {b} met.",
            ),
            (
                "the Eskin incident",
                "the Farrow incident",
                "{a} is near {b}.",
            ),
            (
                "the Garth incident",
                "the Hollin incident",
                "{a} did not cause {b}.",
            ),
        ];
        let corpus: Vec<CorpusItem> = items
            .iter()
            .enumerate()
            .map(|(k, (a, b, tpl))| CorpusItem {
                id: format!("fp-{k}"),
                content: tpl.replace("{a}", a).replace("{b}", b),
                memory_type: "fact".into(),
                tags: vec![a.to_string(), b.to_string()],
                created_at: base + chrono::Duration::minutes(k as i64),
            })
            .collect();
        ingest_corpus(&manager, &corpus).expect("ingest");

        let graph = manager.get_user_graph(EVAL_USER).expect("graph");
        let g = graph.read();
        let pair_edges = |a: &str, b: &str| {
            let ua = g.find_entity_by_name(a).expect("lookup").expect(a).uuid;
            let ub = g.find_entity_by_name(b).expect("lookup").expect(b).uuid;
            let edges: Vec<(bool, bool)> = g
                .get_entity_relationships_limited(&ua, Some(64))
                .expect("edges")
                .iter()
                .filter(|e| {
                    (e.from_entity == ua && e.to_entity == ub)
                        || (e.from_entity == ub && e.to_entity == ua)
                })
                .map(|e| (e.relation_type.is_causal(), e.from_entity == ua))
                .collect();
            edges
        };

        let causal = pair_edges(items[0].0, items[0].1);
        assert!(
            causal.iter().any(|&(c, forward)| c && forward),
            "positive control: '{}' must yield a cause-to-effect causal edge, got {causal:?}",
            corpus[0].content
        );
        for (k, (a, b, _)) in items.iter().enumerate().skip(1) {
            let edges = pair_edges(a, b);
            assert!(
                !edges.is_empty(),
                "'{}' formed no edge at all",
                corpus[k].content
            );
            assert!(
                edges.iter().all(|&(c, _)| !c),
                "'{}' produced a causal edge: {edges:?}",
                corpus[k].content
            );
        }
        drop(g);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Same pipeline at HARNESS scale: with the default chain count every
    /// memory shares the "incident"/"set in motion"/"brought about" phrasing,
    /// so the failure mode (lineage P@1 = 0.0 in CI while the 1-chain repro
    /// passes end-to-end) must be scale-induced — dilution of the walked
    /// origin among cross-chain lexical distractors, or cross-chain entity
    /// resolution bleeding (substring tiers matching "Vornak" ↔ "Vornak2").
    /// Roadmap ② training-data export: runs the capped LoCoMo suite (Full
    /// mode) with the fusion feature export armed, producing candidate-level
    /// (vec, bm25, graph, is_gold) rows for the per-leg calibration fit.
    /// Run explicitly:
    ///   SHODH_MAX_CASES=300 SHODH_MAX_CORPUS=1500 cargo test --release \
    ///     export_fusion_training_data -- --ignored --nocapture
    /// Graph census diff between two harness repeat storages — the
    /// determinism-hunt diagnostic. Opens the persisted entity graphs of two
    /// completed repeats and prints set differences in entities (name+labels)
    /// and edges (from→to, relation type), naming the ingest path that varies.
    /// Run explicitly:
    ///   SHODH_CENSUS_A=<path>/repeat_0/recall-eval/graph \
    ///   SHODH_CENSUS_B=<path>/repeat_1/recall-eval/graph \
    ///   cargo test --release graph_census_diff -- --ignored --nocapture
    #[test]
    #[ignore = "offline diagnostic over existing repeat storages — run explicitly"]
    fn graph_census_diff() {
        let path_a = std::env::var("SHODH_CENSUS_A").expect("SHODH_CENSUS_A unset");
        let path_b = std::env::var("SHODH_CENSUS_B").expect("SHODH_CENSUS_B unset");
        let census = |p: &str| {
            let g = crate::graph_memory::GraphMemory::new(std::path::Path::new(p), None)
                .expect("open graph storage");
            let ents = g.get_all_entities().expect("entities");
            let names: std::collections::BTreeMap<String, String> = ents
                .iter()
                .map(|e| (e.name.clone(), format!("{:?}", e.labels)))
                .collect();
            let by_uuid: std::collections::HashMap<uuid::Uuid, String> =
                ents.iter().map(|e| (e.uuid, e.name.clone())).collect();
            let mut edges: std::collections::BTreeMap<String, String> = Default::default();
            for r in g.get_all_relationships().expect("relationships") {
                let f = by_uuid
                    .get(&r.from_entity)
                    .cloned()
                    .unwrap_or_else(|| r.from_entity.to_string());
                let t = by_uuid
                    .get(&r.to_entity)
                    .cloned()
                    .unwrap_or_else(|| r.to_entity.to_string());
                // Weight included so the diff catches edges that exist in both
                // repeats but accumulated different strengths.
                edges.insert(
                    format!("{f} -[{:?}]-> {t}", r.relation_type),
                    format!("{:.9}", r.strength),
                );
            }
            (names, edges)
        };
        let (ents_a, edges_a) = census(&path_a);
        let (ents_b, edges_b) = census(&path_b);
        println!(
            "A: {} entities, {} edge keys | B: {} entities, {} edge keys",
            ents_a.len(),
            edges_a.len(),
            ents_b.len(),
            edges_b.len()
        );
        for (n, l) in &ents_a {
            match ents_b.get(n) {
                None => println!("ENTITY only in A: {n} {l}"),
                Some(lb) if lb != l => println!("ENTITY labels differ: {n} A={l} B={lb}"),
                _ => {}
            }
        }
        for n in ents_b.keys() {
            if !ents_a.contains_key(n) {
                println!("ENTITY only in B: {n} {}", ents_b[n]);
            }
        }
        for (e, c) in &edges_a {
            match edges_b.get(e) {
                None => println!("EDGE only in A: {e} (x{c})"),
                Some(cb) if cb != c => println!("EDGE count differs: {e} A={c} B={cb}"),
                _ => {}
            }
        }
        for e in edges_b.keys() {
            if !edges_a.contains_key(e) {
                println!("EDGE only in B: {e} (x{})", edges_b[e]);
            }
        }
    }

    #[test]
    #[ignore = "training-data export — run explicitly"]
    fn export_fusion_training_data() {
        let dir = unique_storage_dir("fusion-export");
        let out = dir.join("fusion_features.jsonl");
        // SAFETY: process-wide env; run as a single explicit --ignored test.
        unsafe {
            std::env::set_var("SHODH_FUSION_FEATURE_EXPORT", &out);
        }
        let inputs = RunInputs {
            storage_path: dir.join("run"),
            corpus_path: Some(crate::recall_harness::fixtures::manifest_path(
                crate::recall_harness::fixtures::LOCOMO_CORPUS_PATH,
            )),
            cases_path: Some(crate::recall_harness::fixtures::manifest_path(
                crate::recall_harness::fixtures::LOCOMO_CASES_PATH,
            )),
            suite: "locomo".into(),
            git_sha: "local".into(),
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };
        let report = run_smoke_suite_with_ranks(&inputs).expect("export run");
        unsafe {
            std::env::remove_var("SHODH_FUSION_FEATURE_EXPORT");
        }
        let stable = std::path::Path::new("target/fusion_training.jsonl");
        std::fs::copy(&out, stable).expect("copy export");
        let lines = std::fs::read_to_string(stable).expect("read export");
        let n = lines.lines().count();
        eprintln!(
            "EXPORT_WRITTEN {} queries={} cases_run={}",
            stable.display(),
            n,
            report.ranks.len()
        );
        assert!(n > 0, "export produced no feature lines");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Typed-walk retrieval wiring (#67): a "where does X live" query must
    /// retrieve the memory that placed X somewhere via a typed LocatedIn edge
    /// — the lineage walk's machinery generalized to relation intents. The
    /// planted sentence is TEMPLATE-EXACT against the semrel exemplar
    /// ("x lives in y") so typing is platform-stable (the embedding-argmax
    /// divergence lesson). Env-flag note: SHODH_TYPED_WALK is set/removed
    /// around the recall; the cue gate ("where does") keeps it inert for any
    /// concurrently-running test's queries.
    #[test]
    fn typed_walk_answers_located_in_query() {
        let dir = unique_storage_dir("typed-walk");
        let manager = build_manager(&dir).expect("manager");
        let base = chrono::Utc::now();
        let mut corpus: Vec<CorpusItem> = Vec::new();
        corpus.push(CorpusItem {
            id: "tw-gold".into(),
            content: "Caroline lives in Denver.".into(),
            memory_type: "fact".into(),
            tags: vec![],
            created_at: base,
        });
        // Lexical crowd: many Caroline memories so the gold has competition.
        for i in 0..20 {
            corpus.push(CorpusItem {
                id: format!("tw-d{i:02}"),
                content: format!("Caroline talked about topic number {i} for a while."),
                memory_type: "fact".into(),
                tags: vec![],
                created_at: base + chrono::Duration::minutes(i as i64 + 1),
            });
        }
        let id_map = ingest_corpus(&manager, &corpus).expect("ingest");
        let gold_uuid = id_map.get("tw-gold").copied().expect("gold ingested");

        let system = manager.get_user_memory(EVAL_USER).expect("system");
        std::env::set_var("SHODH_TYPED_WALK", "1");
        let results = system
            .read()
            .recall(&crate::memory::types::Query {
                query_text: Some("Where does Caroline live now?".into()),
                max_results: 10,
                layers: Some(LayerMode::Full),
                ..Default::default()
            })
            .expect("recall");
        std::env::remove_var("SHODH_TYPED_WALK");

        let rank = results.iter().position(|m| m.id.0 == gold_uuid);
        assert!(
            rank.is_some(),
            "typed walk must surface the LocatedIn memory in top-10; got: {:?}",
            results
                .iter()
                .map(|m| m.experience.content.clone())
                .collect::<Vec<_>>()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression guard for the lineage flood (2026-06-11): NER fragment
    /// entities ("Mor", "wen") used to receive causal edges from the pair-typing
    /// loop, forming cross-document causal bridges that fused 57/60 chains
    /// (backward-cone p50 = 48 when the true cone is 1; r@10 pinned at 0.05).
    /// The maximal-mention ingest guard must keep fragments out of causal
    /// typing entirely, and every chain's backward cone must be exactly its own
    /// root, ranked first.
    #[test]
    fn lineage_fragment_bridges_never_form() {
        let dir = unique_storage_dir("lineage-flood-diag");
        let manager = build_manager(&dir).expect("manager");
        let chains = crate::recall_harness::multihop::DEFAULT_CHAINS;
        let (corpus, _cases) =
            crate::recall_harness::lineage_harness::generate_lineage_fixtures(chains);
        ingest_corpus(&manager, &corpus).expect("ingest");

        let graph = manager.get_user_graph(EVAL_USER).expect("graph");
        let g = graph.read();
        let stats = g.get_stats().expect("stats");
        eprintln!(
            "DIAG graph: {} entities, {} edges (expected ~{} event entities)",
            stats.entity_count,
            stats.relationship_count,
            chains * 3
        );

        // FRAGMENT CENSUS: entities that are not full "... incident" event names.
        // For each, count causal IN (others claim to cause it) and causal OUT (it
        // claims to cause others) — a fragment with BOTH is a cross-chain bridge
        // for the backward walk.
        let all = g.get_all_entities().expect("entities");
        let mut bridges = 0usize;
        eprintln!("DIAG fragment census ({} entities total):", all.len());
        for ent in &all {
            // Only the event names themselves are skipped: a phrase that merely contains the
            // head noun ("incident set") is exactly the kind of entity that bridges chains.
            if ent.name.starts_with("the ") && ent.name.ends_with(" incident") {
                continue;
            }
            let edges = g
                .get_entity_relationships_limited(&ent.uuid, Some(256))
                .expect("edges");
            let causal_in = edges
                .iter()
                .filter(|e| e.to_entity == ent.uuid && e.relation_type.is_causal())
                .count();
            let causal_out = edges
                .iter()
                .filter(|e| e.from_entity == ent.uuid && e.relation_type.is_causal())
                .count();
            if causal_in > 0 || causal_out > 0 {
                let bridge = causal_in > 0 && causal_out > 0;
                if bridge {
                    bridges += 1;
                }
                eprintln!(
                    "  frag {:?}: causal_in={causal_in} causal_out={causal_out}{}",
                    ent.name,
                    if bridge { "  <-- BRIDGE" } else { "" }
                );
            }
        }
        eprintln!("DIAG bridge fragments: {bridges}");

        // CONE SWEEP: walk from every chain's effect entity. Corpus layout per
        // chain i: item 2i = root "A set B in motion.", item 2i+1 = mid
        // "B then brought about C.".
        let mut sizes: Vec<usize> = Vec::new();
        let mut root_found = 0usize;
        let mut root_top1 = 0usize;
        let mut fattest: (usize, String, Uuid) = (0, String::new(), Uuid::nil());
        for i in 0..chains {
            let root_name = corpus[2 * i]
                .content
                .split(" set ")
                .next()
                .unwrap()
                .to_string();
            let effect_name = corpus[2 * i + 1]
                .content
                .split(" then brought about ")
                .nth(1)
                .unwrap()
                .trim_end_matches('.')
                .to_string();
            let (Some(root), Some(effect)) = (
                g.find_entity_by_name(&root_name).expect("lookup"),
                g.find_entity_by_name(&effect_name).expect("lookup"),
            ) else {
                eprintln!("  chain {i}: UNRESOLVED ({root_name} / {effect_name})");
                continue;
            };
            let origins = g.trace_causal_origins(&[effect.uuid], 8).expect("walk");
            sizes.push(origins.len());
            let found = origins.iter().any(|(id, _)| *id == root.uuid);
            if found {
                root_found += 1;
                if origins
                    .first()
                    .map(|(id, _)| *id == root.uuid)
                    .unwrap_or(false)
                {
                    root_top1 += 1;
                }
            }
            if origins.len() > fattest.0 {
                fattest = (origins.len(), effect_name.clone(), effect.uuid);
            }
        }
        sizes.sort_unstable();
        let n = sizes.len().max(1);
        eprintln!(
            "DIAG cone sweep over {} chains: min={} p50={} max={} | cones>3: {} | root found: {}/{} | root TOP-1 by score: {}/{}",
            n,
            sizes.first().unwrap_or(&0),
            sizes.get(n / 2).unwrap_or(&0),
            sizes.last().unwrap_or(&0),
            sizes.iter().filter(|s| **s > 3).count(),
            root_found,
            n,
            root_top1,
            n
        );

        // BRIDGE TRACE on the fattest cone: 1-hop and 2-hop backward parents
        // (prints only on failure conditions, as forensics).
        if fattest.0 > 1 {
            eprintln!(
                "DIAG fattest cone: {} origins from {:?} — backward parents:",
                fattest.0, fattest.1
            );
            let edges = g
                .get_entity_relationships_limited(&fattest.2, Some(64))
                .expect("edges");
            for e in edges
                .iter()
                .filter(|e| e.to_entity == fattest.2 && e.relation_type.is_causal())
            {
                let pname = g.get_entity(&e.from_entity).ok().flatten().map(|x| x.name);
                eprintln!("  hop1 parent: {pname:?}");
                let pedges = g
                    .get_entity_relationships_limited(&e.from_entity, Some(64))
                    .expect("edges");
                for pe in pedges
                    .iter()
                    .filter(|pe| pe.to_entity == e.from_entity && pe.relation_type.is_causal())
                {
                    let gname = g.get_entity(&pe.from_entity).ok().flatten().map(|x| x.name);
                    eprintln!("    hop2 parent: {gname:?}");
                }
            }
        }

        // The regression assertions (measured before the maximal-mention guard:
        // bridges=34, cone p50=48, top-1 18/60; after: 0 / 1 / 60).
        assert_eq!(bridges, 0, "fragment entities acquired causal edges again");
        assert_eq!(
            sizes.last().copied().unwrap_or(0),
            1,
            "a backward cone exceeded its own chain — cross-chain causal bleed is back"
        );
        assert_eq!(
            root_found, n,
            "a chain's true root vanished from its backward cone"
        );
        assert_eq!(
            root_top1, n,
            "a chain's true root is no longer the strongest-scored origin"
        );
    }

    #[test]
    #[ignore = "parity gap: the causal walk now reaches the root at 60 chains, but veld recall never consumes trace_causal_origins (upstream's SHODH_CAUSAL_ORIGIN stage in memory/mod.rs), so the root memory is absent from Full top-10"]
    fn lineage_walk_survives_harness_scale() {
        let dir = unique_storage_dir("lineage-scale");
        let manager = build_manager(&dir).expect("manager");
        let chains = crate::recall_harness::multihop::DEFAULT_CHAINS;
        let (corpus, _cases) =
            crate::recall_harness::lineage_harness::generate_lineage_fixtures(chains);
        ingest_corpus(&manager, &corpus).expect("ingest");

        let graph = manager.get_user_graph(EVAL_USER).expect("graph");
        let g = graph.read();
        // Probe a middle chain (i=5): events 15, 16, 17.
        let a = g
            .find_entity_by_name("the Polnar incident")
            .expect("lookup")
            .map(|e| e.uuid);
        let c = g
            .find_entity_by_name("the Selvic incident")
            .expect("lookup")
            .map(|e| e.uuid);
        let (a, c) = match (a, c) {
            (Some(a), Some(c)) => (a, c),
            other => panic!(
                "SCALE STAGE 1 FAIL: chain-5 entities unresolved at {chains} chains: {other:?} \
                 — cross-chain entity-resolution bleeding"
            ),
        };
        let origins = g.trace_causal_origins(&[c], 8).expect("walk");
        if !origins.iter().any(|(id, _)| *id == a) {
            // Diagnostics: what does the effect node's neighbourhood look like?
            let c_edges = g
                .get_entity_relationships_limited(&c, Some(64))
                .expect("edges");
            let incoming_causal = c_edges
                .iter()
                .filter(|e| e.to_entity == c && e.relation_type.is_causal())
                .count();
            let type_counts: std::collections::HashMap<&str, usize> =
                c_edges.iter().fold(Default::default(), |mut m, e| {
                    *m.entry(e.relation_type.as_str()).or_default() += 1;
                    m
                });
            panic!(
                "SCALE STAGE 4 FAIL at {chains} chains: {} origins; effect node has \
                 {} edges (incoming causal: {incoming_causal}; types: {type_counts:?}); \
                 graph stats: {:?}",
                origins.len(),
                c_edges.len(),
                g.get_stats()
                    .map(|s| (s.entity_count, s.relationship_count))
                    .ok()
            );
        }
        drop(g);

        let system = manager.get_user_memory(EVAL_USER).expect("system");

        // MUTATION PROBE: the full harness runs 5 mode passes (~600 queries)
        // before Full and measures 0.0, while a fresh-storage Full query works.
        // Replicate the harness's pre-Full query traffic, then re-check the walk
        // and the Full recall — if they now fail, query-driven graph mutation
        // (CoRetrieved edges / co-access strengthening) is the harness zero.
        for mode in &crate::memory::types::LayerMode::ALL[..5] {
            for case in _cases.iter() {
                let q = crate::memory::types::Query {
                    query_text: Some(case.query.clone()),
                    max_results: 10,
                    layers: Some(*mode),
                    ..Default::default()
                };
                let _ = system.read().recall(&q);
            }
        }
        {
            let g = graph.read();
            let origins_after = g.trace_causal_origins(&[c], 8).expect("walk after traffic");
            assert!(
                origins_after.iter().any(|(id, _)| *id == a),
                "MUTATION CONFIRMED AT THE WALK: origins lost after {} pre-Full \
                 queries (had them on fresh storage)",
                _cases.len() * 5
            );
        }

        let query = crate::memory::types::Query {
            query_text: Some("What was the earliest origin behind the Selvic incident?".into()),
            max_results: 10,
            layers: Some(LayerMode::Full),
            ..Default::default()
        };
        let results = system.read().recall(&query).expect("recall");
        let texts: Vec<String> = results
            .iter()
            .map(|m| m.experience.content.clone())
            .collect();
        let root_rank = texts.iter().position(|t| t.contains("Polnar"));
        assert!(
            root_rank.is_some(),
            "MUTATION CONFIRMED AT USAGE: walk survives but root memory absent \
             from Full top-10 after prior-mode traffic. Retrieved: {texts:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Lineage and multihop analyses side by side, for before/after comparisons
    /// of an ingest change. Run explicitly in release:
    ///   cargo test --release --lib lineage_and_multihop_recall_report -- --ignored --nocapture
    #[test]
    #[ignore = "measurement: lineage + multihop through every layer mode, run with --ignored in release"]
    fn lineage_and_multihop_recall_report() {
        let dir = unique_storage_dir("lineage-multihop-report");
        let inputs = |sub: &str| RunInputs {
            storage_path: dir.join(sub),
            corpus_path: None,
            cases_path: None,
            suite: sub.into(),
            git_sha: "local".into(),
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };
        let lineage = crate::recall_harness::lineage_harness::analyze_lineage(
            &inputs("lineage"),
            crate::recall_harness::lineage_harness::DEFAULT_CHAINS,
        )
        .expect("lineage run");
        let multihop = crate::recall_harness::multihop::analyze_multihop(
            &inputs("multihop"),
            crate::recall_harness::multihop::DEFAULT_CHAINS,
        )
        .expect("multihop run");
        for (name, report) in [("lineage", &lineage), ("multihop", &multihop)] {
            for r in &report.rows {
                eprintln!(
                    "REPORT {name} chains={} layer={} multihop_r@10={:.4} multihop_p@1={:.4} multihop_mrr={:.4} onehop_r@10={:.4} onehop_p@1={:.4}",
                    report.chains,
                    r.layer,
                    r.multihop_recall_at_10,
                    r.multihop_p_at_1,
                    r.multihop_mrr,
                    r.onehop_recall_at_10,
                    r.onehop_p_at_1
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The FULL harness function, exactly as CI runs it (analyze_lineage builds
    /// its own fixtures, runs all six layer modes, measures by uuid relevance).
    /// CI measures 0.0 while every staged local repro passes — this localizes
    /// whether the zero lives in the harness measurement plumbing / multi-mode
    /// sequence or in the CI environment.
    #[test]
    #[ignore = "full lineage harness, six layer modes through the cross-encoder; ran 90+ minutes in a debug build, run with --ignored in release"]
    fn lineage_harness_end_to_end_reproduces_ci() {
        let dir = unique_storage_dir("lineage-harness-e2e");
        let inputs = RunInputs {
            storage_path: dir.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "lineage".into(),
            git_sha: "local".into(),
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };
        let report = crate::recall_harness::lineage_harness::analyze_lineage(&inputs, 60)
            .expect("harness run");
        for r in &report.rows {
            eprintln!(
                "  {} root P@1={:.3} r@10={:.3}",
                r.layer, r.multihop_p_at_1, r.multihop_recall_at_10
            );
        }
        // Measurement-plumbing dump: what did the harness think case
        // lin-why-0005 retrieved, by corpus id? (rank lists persist in the
        // lineage_run storage's ReportWithRanks — easier: re-run one pass here
        // and read the rank list directly.)
        {
            let fixture_dir = dir.join("lineage_fixtures");
            let run_inputs = RunInputs {
                storage_path: dir.join("plumb_run"),
                corpus_path: Some(fixture_dir.join("lineage_corpus.jsonl")),
                cases_path: Some(fixture_dir.join("lineage_cases.jsonl")),
                suite: "lineage".into(),
                git_sha: "local".into(),
                repeats: 1,
                layer_modes: vec![LayerMode::Full],
                age_days: 0.0,
            };
            let out = run_smoke_suite_with_ranks(&run_inputs).expect("plumb run");
            if let Some(l) = out.ranks.iter().find(|l| l.case_id == "lin-why-0005") {
                eprintln!("  PLUMB lin-why-0005 retrieved: {:?}", l.retrieved);
            }
            for (layer, records) in &out.per_case_by_layer {
                if let Some(r) = records.iter().find(|r| r.case_id == "lin-why-0005") {
                    eprintln!(
                        "  PLUMB {layer} lin-why-0005 r@k={:.2} relevant={}/{} missed={:?}",
                        r.recall_at_k, r.relevant_found, r.relevant_total, r.missed
                    );
                }
            }
        }
        let full = report
            .rows
            .iter()
            .find(|r| r.layer == "full")
            .expect("full row");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            full.multihop_recall_at_10 > 0.0,
            "HARNESS-LEVEL ZERO REPRODUCED LOCALLY: full-layer root-cause \
             r@10 = 0 — the zero is in the harness path, not the CI environment"
        );
    }

    #[test]
    fn per_case_records_flag_missed_relevant_items() {
        let cases = vec![
            SmokeCase {
                id: "smoke-001".into(),
                category: SmokeCategory::MultiHop,
                query: "chain query".into(),
                fixture_corpus_id: "shodh-smoke".into(),
                relevant: vec![
                    fixtures::RelevanceJudgement {
                        corpus_item_id: "ssm-001".into(),
                        grade: 3,
                    },
                    fixtures::RelevanceJudgement {
                        corpus_item_id: "ssm-002".into(),
                        grade: 1,
                    },
                ],
            },
            SmokeCase {
                id: "smoke-002".into(),
                category: SmokeCategory::Entity,
                query: "entity query".into(),
                fixture_corpus_id: "shodh-smoke".into(),
                relevant: vec![fixtures::RelevanceJudgement {
                    corpus_item_id: "ssm-010".into(),
                    grade: 3,
                }],
            },
        ];
        let metrics = vec![
            Metrics {
                recall_at_k: 0.5,
                ndcg_at_k: 0.6,
                mrr: 1.0,
                p_at_1: 1.0,
                ..Default::default()
            },
            Metrics {
                recall_at_k: 1.0,
                ndcg_at_k: 1.0,
                mrr: 1.0,
                p_at_1: 1.0,
                ..Default::default()
            },
        ];
        let ranks = vec![
            // Case 1 retrieved ssm-001 (found) but dropped ssm-002 (missed).
            CaseRankList {
                case_id: "smoke-001".into(),
                retrieved: vec!["ssm-001".into(), "ssm-099".into()],
            },
            // Case 2 retrieved its only relevant item.
            CaseRankList {
                case_id: "smoke-002".into(),
                retrieved: vec!["ssm-010".into()],
            },
        ];

        // Diagnostic off: the deep list is a copy of the production list, so
        // the depth fields degrade to plain wider cutoffs of the same list.
        let recs = build_per_case_records(&cases, &metrics, &ranks, &ranks);
        assert_eq!(recs.len(), 2);

        let r0 = &recs[0];
        assert_eq!(r0.case_id, "smoke-001");
        assert_eq!(r0.category, "multi_hop");
        assert_eq!(r0.relevant_total, 2);
        assert_eq!(r0.relevant_found, 1);
        assert_eq!(r0.missed, vec!["ssm-002".to_string()]);
        assert_eq!(r0.recall_at_k, 0.5);
        assert_eq!(r0.ndcg_at_k, 0.6);
        // Wider cutoffs recompute from the deep list vs gold: case 1 found
        // ssm-001 but not ssm-002, so 1/2 at every cutoff.
        assert_eq!(r0.recall_at_50, 0.5);
        assert_eq!(r0.recall_at_100, 0.5);

        let r1 = &recs[1];
        assert_eq!(r1.category, "entity");
        assert_eq!(r1.relevant_found, 1);
        assert!(r1.missed.is_empty());
        assert_eq!(r1.recall_at_50, 1.0);
        assert_eq!(r1.recall_at_100, 1.0);

        // Diagnostic on (RECALL_DIAG_K semantics): the deep list comes from a
        // SEPARATE deeper query. A gold item absent from the production top-k
        // but present in the deep list must still be flagged `missed` (headline
        // fields stay on the production list) while recall@50/@100 count it —
        // the exact split that turns "gold ranked >10 at depth" and "gold never
        // retrieved at any depth" into different numbers.
        let deep_ranks = vec![
            CaseRankList {
                case_id: "smoke-001".into(),
                retrieved: vec!["ssm-001".into(), "ssm-099".into(), "ssm-002".into()],
            },
            CaseRankList {
                case_id: "smoke-002".into(),
                retrieved: vec!["ssm-010".into()],
            },
        ];
        let recs = build_per_case_records(&cases, &metrics, &ranks, &deep_ranks);
        let r0 = &recs[0];
        assert_eq!(r0.missed, vec!["ssm-002".to_string()]);
        assert_eq!(r0.relevant_found, 1);
        assert_eq!(r0.recall_at_k, 0.5);
        assert_eq!(r0.recall_at_50, 1.0);
        assert_eq!(r0.recall_at_100, 1.0);
    }

    /// Smoke test the full runner end-to-end against the canonical fixtures.
    ///
    /// Asserts only that the report is well-formed: 30 cases, six categories
    /// represented, full layer present. Quality numbers themselves are
    /// captured by RH-6 baseline runs, not by unit tests.
    #[test]
    fn runner_executes_smoke_suite_and_produces_well_formed_report() {
        let storage = unique_storage_dir("runner");
        let inputs = RunInputs {
            storage_path: storage.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };
        let report = run_smoke_suite(&inputs).expect("runner must succeed");
        let _ = std::fs::remove_dir_all(&storage);

        assert_eq!(
            report.case_count,
            crate::recall_harness::fixtures::TOTAL_SMOKE_CASES
        );
        assert_eq!(report.repeats, 1);
        assert_eq!(report.suite, "smoke");
        assert_eq!(report.embedder, EMBEDDER_ID);
        assert!(report.layers.contains_key("full"));
        for cat in [
            "decision",
            "code",
            "temporal",
            "entity",
            "multi_hop",
            "negation",
        ] {
            let cr = report
                .by_category
                .get(cat)
                .expect("category must be present");
            assert_eq!(
                cr.case_count,
                crate::recall_harness::fixtures::CASES_PER_CATEGORY,
                "category {cat} must have CASES_PER_CATEGORY cases"
            );
        }
        // No infrastructure-level failures expected on the canonical fixtures.
        assert!(
            report.failures.iter().all(|f| f.kind == "case"),
            "no infrastructure failures expected, got {:?}",
            report.failures
        );
    }

    /// OBSERVATION (FUSION_V2): does Borda+rescue actually lift a graph-reachable
    /// answer that RRF buries? Planted 2-hop: query entity Alice —(bridge)→ Bob;
    /// gold "Bob discovered ..." is reachable via the graph but the lexical
    /// distractor "Alice discovered ..." matches the query better. Print the order
    /// and the gold's rank under RRF vs V2 — observe, don't assert.
    #[test]
    fn observe_fusion_v2_graph_rescue() {
        use crate::recall_harness::fixtures::CorpusItem;
        use chrono::{Duration, TimeZone, Utc};

        let storage = unique_storage_dir("v2-observe");
        let manager = build_manager(&storage).unwrap();
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mk = |id: &str, content: &str, i: i64| CorpusItem {
            id: id.to_string(),
            content: content.to_string(),
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + Duration::minutes(i),
        };
        let mut corpus = vec![
            mk(
                "bridge",
                "Alice collaborates closely with Bob on their research",
                0,
            ),
            mk(
                "gold",
                "Bob discovered a rare luminescent mineral inside the cave",
                1,
            ),
            mk(
                "distract",
                "Alice discovered an ordinary grey rock out in the field",
                2,
            ),
        ];
        // Many noise memories so a graph-only answer is genuinely OUTSIDE the
        // hybrid top-`max_results` window (the rescue gate). Without this the
        // rescue can never fire (corpus must exceed max_results).
        for i in 0..24 {
            corpus.push(mk(
                &format!("noise{i:02}"),
                &format!("Unrelated note number {i} about gardening, cooking, and travel plans"),
                10 + i,
            ));
        }
        let id_map = ingest_corpus(&manager, &corpus).unwrap();
        let rev: HashMap<Uuid, String> = id_map.iter().map(|(c, u)| (*u, c.clone())).collect();
        let system = manager.get_user_memory(EVAL_USER).unwrap();

        let recall_order = |q: &str| -> Vec<(String, f32)> {
            let query = Query {
                query_text: Some(q.to_string()),
                max_results: 3,
                layers: Some(LayerMode::Full),
                ..Default::default()
            };
            system
                .read()
                .recall(&query)
                .map(|ms| {
                    ms.iter()
                        .map(|m| {
                            (
                                rev.get(&m.id.0)
                                    .cloned()
                                    .unwrap_or_else(|| m.id.0.to_string()),
                                m.score.unwrap_or(0.0),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        let q = "What did Alice's collaborator discover?";
        std::env::remove_var("SHODH_FUSION_V2");
        let off = recall_order(q);
        std::env::set_var("SHODH_FUSION_V2", "1");
        let on = recall_order(q);
        std::env::remove_var("SHODH_FUSION_V2");
        let _ = std::fs::remove_dir_all(&storage);

        let rank = |v: &Vec<(String, f32)>, id: &str| v.iter().position(|(i, _)| i == id);
        eprintln!("FULL  OFF (RRF): {off:?}");
        eprintln!("FULL  ON  (V2) : {on:?}");
        eprintln!(
            "gold rank  OFF={:?}  ON={:?}   distract rank OFF={:?} ON={:?}",
            rank(&off, "gold"),
            rank(&on, "gold"),
            rank(&off, "distract"),
            rank(&on, "distract"),
        );
    }

    /// OBSERVATION: the clean A/B showed V2 ontology +facts=0.88 but full=0.083.
    /// Reproduce the harness scenario (short "Alice" vs longer org names, equi-frame)
    /// and check the person's rank at PlusFacts vs Full under V2 — to confirm the
    /// content-length quality gate was the `full`-layer destroyer (and that dropping
    /// the verbosity factor keeps the person #1 at full too).
    #[test]
    fn observe_v2_full_layer_ontology() {
        use crate::recall_harness::fixtures::CorpusItem;
        use chrono::{Duration, TimeZone, Utc};

        let storage = unique_storage_dir("v2-onto");
        let manager = build_manager(&storage).unwrap();
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let frame = |e: &str| format!("{e} was registered for the Paris trade summit.");
        let mk = |id: &str, content: String, i: i64| CorpusItem {
            id: id.to_string(),
            content,
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + Duration::minutes(i),
        };
        let corpus = vec![
            mk("person", frame("Alice"), 0),
            mk("org1", frame("Acme Corporation"), 1),
            mk("org2", frame("Globex Industries"), 2),
            mk("org3", frame("Initech Systems"), 3),
        ];
        let id_map = ingest_corpus(&manager, &corpus).unwrap();
        let rev: HashMap<Uuid, String> = id_map.iter().map(|(c, u)| (*u, c.clone())).collect();
        let system = manager.get_user_memory(EVAL_USER).unwrap();

        let order = |layers: LayerMode| -> Vec<String> {
            let query = Query {
                query_text: Some(
                    "Which person was registered for the Paris trade summit?".to_string(),
                ),
                max_results: 10,
                layers: Some(layers),
                ..Default::default()
            };
            system
                .read()
                .recall(&query)
                .map(|ms| {
                    ms.iter()
                        .map(|m| {
                            rev.get(&m.id.0)
                                .cloned()
                                .unwrap_or_else(|| m.id.0.to_string())
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        std::env::set_var("SHODH_FUSION_V2", "1");
        let pf = order(LayerMode::PlusFacts);
        let full = order(LayerMode::Full);
        std::env::remove_var("SHODH_FUSION_V2");
        let _ = std::fs::remove_dir_all(&storage);

        let rank = |v: &Vec<String>, id: &str| v.iter().position(|i| i == id);
        eprintln!("V2 +facts: {pf:?}");
        eprintln!("V2 full  : {full:?}");
        eprintln!(
            "person rank  +facts={:?}  full={:?}",
            rank(&pf, "person"),
            rank(&full, "person")
        );
    }

    #[test]
    fn shard_windows_tile_the_manifest_exactly_once() {
        // The property the sharded LongMemEval workflow depends on: with a fixed
        // shard count and `per = ceil(len / shards)`, the windows are disjoint
        // and cover every case exactly once. If this breaks, a suite total is
        // silently computed over duplicated or skipped questions.
        const SHARDS: usize = 10;
        for len in [0usize, 1, 7, 9, 10, 11, 47, 50, 149, 150, 479] {
            let per = len.div_ceil(SHARDS).max(1);
            let mut covered = vec![0u32; len];
            for shard in 0..SHARDS {
                let (start, end) = shard_window(len, Some(shard * per), Some(per));
                assert!(start <= end, "len={len} shard={shard}: inverted window");
                assert!(end <= len, "len={len} shard={shard}: window past the end");
                for slot in covered.iter_mut().take(end).skip(start) {
                    *slot += 1;
                }
            }
            assert!(
                covered.iter().all(|&c| c == 1),
                "len={len}: every case must be covered exactly once, got {covered:?}"
            );
        }
    }

    #[test]
    fn out_of_range_shard_yields_an_empty_window_not_a_panic() {
        // Routine, not exceptional: the shard count is fixed at 10 while the
        // question total varies, so the tail shards fall off the end whenever
        // the total is small.
        assert_eq!(shard_window(5, Some(50), Some(5)), (5, 5));
        assert_eq!(shard_window(0, Some(0), Some(5)), (0, 0));
        assert_eq!(shard_window(5, Some(3), Some(99)), (3, 5));
    }

    #[test]
    fn absent_offset_and_limit_select_the_whole_manifest() {
        assert_eq!(shard_window(42, None, None), (0, 42));
        assert_eq!(shard_window(42, None, Some(10)), (0, 10));
        assert_eq!(shard_window(42, Some(10), None), (10, 42));
    }

    #[test]
    fn experience_type_recognises_known_strings() {
        assert!(matches!(
            experience_type_for("decision"),
            ExperienceType::Decision
        ));
        assert!(matches!(
            experience_type_for("Reference"),
            ExperienceType::Observation
        ));
        assert!(matches!(
            experience_type_for("event"),
            ExperienceType::Observation
        ));
        assert!(matches!(
            experience_type_for("nonsense"),
            ExperienceType::Observation
        ));
    }

    /// Record a baseline from THIS tree with THIS process's embedder:
    ///
    ///   VELD_EMBEDDER=http VELD_WRITE_BASELINE=tests/recall/baseline.json \
    ///     cargo test --release --lib -- --ignored record_smoke_baseline --nocapture
    ///
    /// The report names the embedder that produced it (compare_to_baseline refuses a diff across
    /// embedders), and git_sha is read from the tree, so the file says what it measures.
    #[test]
    #[ignore = "writes a baseline file; run explicitly with VELD_WRITE_BASELINE set"]
    fn record_smoke_baseline() {
        let out = std::env::var("VELD_WRITE_BASELINE")
            .expect("VELD_WRITE_BASELINE unset — refusing to guess where a baseline goes");
        let git_sha = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .expect("git rev-parse HEAD must succeed: a baseline without a sha measures nothing");
        let storage = unique_storage_dir("baseline");
        let inputs = RunInputs {
            storage_path: storage.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha,
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };
        let report = run_smoke_suite(&inputs).expect("smoke suite must run");
        let _ = std::fs::remove_dir_all(&storage);
        assert_ne!(
            report.embedder,
            crate::recall_harness::report::EMBEDDER_UNKNOWN,
            "a baseline of unknown provenance is not a baseline"
        );
        let json = serde_json::to_string_pretty(&report).unwrap();
        std::fs::write(&out, json).expect("write baseline");
        let full = report.layers.get("full").expect("full layer");
        eprintln!(
            "baseline written to {out}: embedder={} sha={} cases={} ndcg@10={:.4} recall@10={:.4} mrr={:.4} p@1={:.4}",
            report.embedder, report.git_sha, report.case_count, full.ndcg_at_10, full.recall_at_10, full.mrr, full.p_at_1
        );
    }

    // -------------------- RH-12 (#272) -------------------------------------
    //
    // Run the full smoke suite with N=2 and verify (a) the report records
    // repeats=2, (b) per-repeat storage subdirs exist, (c) quality metrics
    // are bit-identical between the N=1 baseline run and the N=2 run.
    //
    // This test is the load-bearing assertion that RH-12 changed only the
    // measurement, not the metric — if N=1 and N=2 disagree on quality,
    // either the determinism guarantee broke OR `aggregate_layer` is no
    // longer a pure function of its inputs.
    //
    // Marked `#[ignore]` because it pays the ingest cost twice (~12 min on
    // a cold runner). Run via `cargo test --release -- --ignored
    // runner_repeats_2_produces_same_quality_as_repeats_1` before shipping
    // changes that touch the harness or scoring path.
    #[test]
    #[ignore = "expensive: runs the smoke suite twice (~12min). enable with --ignored before shipping harness changes."]
    fn runner_repeats_2_produces_same_quality_as_repeats_1() {
        let storage1 = unique_storage_dir("repeats1");
        let storage2 = unique_storage_dir("repeats2");

        let inputs1 = RunInputs {
            storage_path: storage1.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };
        let inputs2 = RunInputs {
            storage_path: storage2.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            repeats: 2,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };

        let r1 = run_smoke_suite(&inputs1).expect("repeats=1 must succeed");
        let r2 = run_smoke_suite(&inputs2).expect("repeats=2 must succeed");

        assert_eq!(r1.repeats, 1);
        assert_eq!(r2.repeats, 2);

        // Quality metrics — bit-identical (RH-11 determinism).
        let f1 = r1.layers.get("full").expect("r1 full");
        let f2 = r2.layers.get("full").expect("r2 full");
        assert_eq!(f1.ndcg_at_10.to_bits(), f2.ndcg_at_10.to_bits(), "ndcg@10");
        assert_eq!(
            f1.recall_at_10.to_bits(),
            f2.recall_at_10.to_bits(),
            "recall@10"
        );
        assert_eq!(f1.mrr.to_bits(), f2.mrr.to_bits(), "mrr");
        assert_eq!(f1.p_at_1.to_bits(), f2.p_at_1.to_bits(), "p@1");
        assert_eq!(f1.map.to_bits(), f2.map.to_bits(), "map");

        // Per-repeat storage subdirs must have been created.
        assert!(
            storage2.join("repeat_0").exists(),
            "repeat_0 storage missing"
        );
        assert!(
            storage2.join("repeat_1").exists(),
            "repeat_1 storage missing"
        );

        // No infrastructure failures (rank-list divergence) on the
        // canonical fixtures — RH-11 guarantees byte-identical ranks.
        let infra_failures: Vec<_> = r2
            .failures
            .iter()
            .filter(|f| f.kind == "infrastructure")
            .collect();
        assert!(
            infra_failures.is_empty(),
            "no infrastructure failures expected on canonical fixtures, got {:?}",
            infra_failures
        );

        let _ = std::fs::remove_dir_all(&storage1);
        let _ = std::fs::remove_dir_all(&storage2);
    }

    /// RH-8 (#270): running the smoke suite with multiple `LayerMode`s
    /// emits one `layers` entry per mode, in pipeline order, and per-mode
    /// rank lists are byte-identical across repeats.
    ///
    /// Marked `#[ignore]` for the same reason as the other end-to-end
    /// runner tests — it pays the ingest cost once and the query loop six
    /// times. Run via `cargo test --release -- --ignored
    /// runner_layer_all_emits_six_modes_with_per_mode_determinism`
    /// before shipping changes that touch any layer gate.
    #[test]
    #[ignore = "expensive: runs the smoke suite with 6 modes (~6× query time). enable with --ignored before shipping layer-gate changes."]
    fn runner_layer_all_emits_six_modes_with_per_mode_determinism() {
        let storage = unique_storage_dir("layer-all");
        let inputs = RunInputs {
            storage_path: storage.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            // Two repeats so the per-mode determinism check actually runs.
            repeats: 2,
            layer_modes: LayerMode::ALL.to_vec(),
            age_days: 0.0,
        };
        let report = run_smoke_suite(&inputs).expect("layer-all run must succeed");
        let _ = std::fs::remove_dir_all(&storage);

        // Six modes, all present, in their canonical report-key form.
        for mode in LayerMode::ALL {
            assert!(
                report.layers.contains_key(mode.report_key()),
                "missing layer entry for {}",
                mode.report_key()
            );
        }
        assert_eq!(report.layers.len(), LayerMode::ALL.len());

        // Per-mode determinism: any divergence would have been recorded
        // as `infrastructure` failures by the runner.
        let infra: Vec<_> = report
            .failures
            .iter()
            .filter(|f| f.kind == "infrastructure")
            .collect();
        assert!(
            infra.is_empty(),
            "per-mode determinism violated: {:?}",
            infra
        );

        // Sanity: `full` ndcg@10 must be >= `vamana_only` ndcg@10 on the
        // canonical fixtures. This is the #270 acceptance invariant — if a
        // future scoring change ever inverts it, that's a real signal.
        let full = report.layers.get("full").expect("full layer present");
        let vamana = report
            .layers
            .get("vamana_only")
            .expect("vamana_only layer present");
        assert!(
            full.ndcg_at_10 + 1e-6 >= vamana.ndcg_at_10,
            "full ndcg@10 ({}) must be >= vamana_only ndcg@10 ({}) on canonical fixtures",
            full.ndcg_at_10,
            vamana.ndcg_at_10
        );
    }

    /// Direct unit test of the cross-repeat divergence detection logic.
    ///
    /// Builds two synthetic `OnePassResult`s with one diverging case, then
    /// runs the same comparison the public path uses. Cheap (no harness),
    /// runs on every CI build.
    #[test]
    fn synthetic_rank_divergence_surfaces_infrastructure_failure() {
        let pass_a_ranks = [
            CaseRankList {
                case_id: "smoke-001".to_string(),
                retrieved: vec!["doc-a".to_string(), "doc-b".to_string()],
            },
            CaseRankList {
                case_id: "smoke-002".to_string(),
                retrieved: vec!["doc-c".to_string(), "doc-d".to_string()],
            },
        ];
        let pass_b_ranks = [
            CaseRankList {
                case_id: "smoke-001".to_string(),
                retrieved: vec!["doc-a".to_string(), "doc-b".to_string()],
            },
            CaseRankList {
                // Divergence on smoke-002.
                case_id: "smoke-002".to_string(),
                retrieved: vec!["doc-d".to_string(), "doc-c".to_string()],
            },
        ];

        // Re-implement the same comparison loop the public path uses,
        // narrow to the assertion we care about. Keeps the test
        // independent of harness setup cost.
        let mut failures: Vec<Failure> = Vec::new();
        for (k, ref_rank) in pass_a_ranks.iter().enumerate() {
            let cur_rank = &pass_b_ranks[k];
            assert_eq!(ref_rank.case_id, cur_rank.case_id);
            if ref_rank.retrieved != cur_rank.retrieved {
                failures.push(Failure {
                    kind: "infrastructure".to_string(),
                    detail: format!(
                        "non-determinism: case {} rank list diverged",
                        ref_rank.case_id
                    ),
                });
            }
        }

        assert_eq!(failures.len(), 1, "exactly one divergence expected");
        assert_eq!(failures[0].kind, "infrastructure");
        assert!(
            failures[0].detail.contains("smoke-002"),
            "failure must name the diverging case: {}",
            failures[0].detail
        );
    }

    /// Every `.rs` under `src/` except the harness, which sets the flags rather than reading them.
    fn src_outside_harness() -> Vec<(PathBuf, String)> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let harness = root.join("recall_harness");
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("read src dir") {
                let path = entry.expect("src dir entry").path();
                if path.starts_with(&harness) {
                    continue;
                }
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).expect("read source");
                    out.push((path, text));
                }
            }
        }
        out
    }

    /// A quoted `"KEY"` in non-test, non-comment code that is not a `set_var`/`remove_var`.
    /// Test code is cut at a file's first `#[cfg(test)]`; cutting too much only makes a
    /// flag look unread, which fails loudly, never the reverse.
    fn reads_env(sources: &[(PathBuf, String)], key: &str) -> bool {
        let quoted = format!("\"{key}\"");
        sources.iter().any(|(_, text)| {
            text.lines()
                .take_while(|l| l.trim() != "#[cfg(test)]")
                .filter(|l| !l.trim_start().starts_with("//"))
                .filter(|l| !l.contains("set_var") && !l.contains("remove_var"))
                .any(|l| l.contains(&quoted))
        })
    }

    fn gates_family(sources: &[(PathBuf, String)], family: &str) -> bool {
        let enabled = format!("is_enabled(\"{family}\")");
        let disabled = format!("is_disabled(\"{family}\")");
        sources
            .iter()
            .filter(|(p, _)| !p.ends_with("memory/ablation.rs"))
            .any(|(_, text)| {
                text.lines()
                    .take_while(|l| l.trim() != "#[cfg(test)]")
                    .any(|l| l.contains(&enabled) || l.contains(&disabled))
            })
    }

    /// Keeps the harness from drifting back into measuring baseline under another
    /// name: every flag an arm sets is read somewhere in `src/` outside the
    /// harness, or is declared unavailable (and then must not be read), and every
    /// family an arm disables has a live gate.
    #[test]
    fn every_ablation_arm_flag_is_read_in_src_or_declared_unavailable() {
        use crate::memory::ablation::{unavailable_reason, DISABLE_BOOSTS_VAR, FAMILIES};

        let sources = src_outside_harness();
        // A walk that read nothing would pass every "is not read" check below.
        assert!(sources.len() > 100, "scanned only {} files", sources.len());
        assert!(
            reads_env(&sources, "SHODH_RECALL_READONLY"),
            "scanner misses a known reader"
        );
        assert!(
            !reads_env(&sources, "SHODH_PLANTED_NEVER_READ"),
            "scanner invents a reader"
        );

        let arms = ablation_configs();
        for AblationConfig { name, env, .. } in &arms {
            for (k, v) in env {
                let read = reads_env(&sources, k);
                let declared = UNAVAILABLE_IN_VELD.iter().any(|(f, _)| f == k);
                assert!(
                    read || declared,
                    "arm `{name}` sets {k}, which nothing in src/ reads: it would measure baseline under another name"
                );
                assert!(
                    !(read && declared),
                    "{k} is now read in src/; delete its UNAVAILABLE_IN_VELD entry"
                );
                if *k == DISABLE_BOOSTS_VAR {
                    for tok in v.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                        if tok == "all" {
                            continue;
                        }
                        if FAMILIES.contains(&tok) {
                            assert!(gates_family(&sources, tok), "family `{tok}` has no gate");
                        } else {
                            assert!(
                                unavailable_reason(tok).is_some(),
                                "arm `{name}` disables `{tok}`, neither wired nor declared upstream-only"
                            );
                        }
                    }
                }
            }
        }
        for (flag, _) in UNAVAILABLE_IN_VELD {
            assert!(
                arms.iter().any(|c| c.sets(flag)),
                "{flag} is declared unavailable but no arm sets it"
            );
        }
        for family in FAMILIES {
            assert!(
                gates_family(&sources, family),
                "family `{family}` has no gate"
            );
        }
        for pin in [
            "SHODH_EVAL_NOW",
            "SHODH_RECALL_READONLY",
            "SHODH_ONNX_THREADS",
        ] {
            assert!(
                reads_env(&sources, pin),
                "harness pin {pin} is read by nothing"
            );
        }
    }

    #[test]
    fn every_boost_ablation_but_plus_hebbian_keeps_hebbian_off() {
        for AblationConfig { name, env, .. } in ablation_configs() {
            for (k, v) in env {
                if k != crate::memory::ablation::DISABLE_BOOSTS_VAR || name == "+hebbian" {
                    continue;
                }
                assert!(
                    v.split(',')
                        .map(str::trim)
                        .any(|t| t == "hebbian" || t == "all"),
                    "`{name}` sets {k}={v} without `hebbian`, which switches the default-off \
                     hebbian term back on and measures two changes at once"
                );
            }
        }
    }

    #[test]
    fn unavailable_arms_are_reported_not_run_as_baseline() {
        let _guard = crate::memory::ablation::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("VELD_RECALL_READONLY").ok();
        std::env::set_var("VELD_RECALL_READONLY", "0");

        for env in [
            vec![("SHODH_DISABLE_FACT_LAYERS", "1")],
            vec![("SHODH_GRAPH_FUSION_WEIGHT", "0")],
            vec![("SHODH_DISABLE_BOOSTS", "hebbian")],
            vec![("SHODH_DISABLE_BOOSTS", "graph_potentiation")],
            vec![("SHODH_DISABLE_BOOSTS", "")],
        ] {
            assert_eq!(arm_unavailable_in_veld(&env), None, "{env:?} is wired");
        }
        for (env, needle) in [
            (vec![("SHODH_GRAPH_TRAVERSE", "1")], "SHODH_GRAPH_TRAVERSE"),
            (vec![("SHODH_GRAPH_EDGE_DIR", "1")], "neighbor_of"),
            (
                vec![
                    ("SHODH_GRAPH_FUSION_WEIGHT", "0"),
                    ("SHODH_GRAPH_EXPAND_K", "5"),
                ],
                "SHODH_GRAPH_EXPAND_K",
            ),
            (
                vec![("SHODH_DISABLE_BOOSTS", "hebbian,lateral_inhibition")],
                "lateral",
            ),
            (
                vec![("SHODH_DISABLE_BOOSTS", "no_such_family")],
                "not declared",
            ),
        ] {
            let why = arm_unavailable_in_veld(&env).unwrap_or_default();
            assert!(
                why.contains(needle),
                "{env:?}: expected `{needle}` in {why:?}"
            );
        }
        std::env::set_var("VELD_RECALL_READONLY", "1");
        let why = arm_unavailable_in_veld(&[("SHODH_DISABLE_BOOSTS", "graph_potentiation")]);
        assert!(why.is_some_and(|w| w.contains("RECALL_READONLY")));
        match prev {
            Some(v) => std::env::set_var("VELD_RECALL_READONLY", v),
            None => std::env::remove_var("VELD_RECALL_READONLY"),
        }

        let row = unavailable_row(
            &AblationConfig {
                name: "+x",
                env: vec![("SHODH_GRAPH_TRAVERSE", "1")],
                rerank_off: false,
                reference: Some(ABLATION_BASELINE),
            },
            "why".into(),
        );
        let json = serde_json::to_value(&row).unwrap();
        assert!(
            json["recall@10"].is_null(),
            "an unavailable arm must carry no metric"
        );
        assert_eq!(json["unavailable_in_veld"], "why");
    }

    /// Each wired query-time flag must change what recall returns on a fixture where
    /// its layer has work to do. Byte-identical output means the flag did not reach
    /// its code path.
    #[test]
    fn wired_query_time_ablations_change_what_recall_returns() {
        use chrono::TimeZone;
        let _guard = crate::memory::ablation::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for k in [
            "SHODH_DISABLE_BOOSTS",
            "SHODH_DISABLE_FACT_LAYERS",
            "SHODH_GRAPH_FUSION_WEIGHT",
        ] {
            std::env::remove_var(k);
        }
        let dir = unique_storage_dir("ablation-wired");
        let manager = build_manager(&dir).expect("manager");
        let base = chrono::Utc.with_ymd_and_hms(2024, 3, 1, 12, 0, 0).unwrap();
        let mk = |id: &str, content: &str, i: i64| CorpusItem {
            id: id.to_string(),
            content: content.to_string(),
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + chrono::Duration::minutes(i),
        };
        let gold_text = "Melanie ran a marathon yesterday.";
        let corpus = vec![
            mk("gold", gold_text, 0),
            mk(
                "m1",
                "Melanie trains with Caroline at the Denver running club.",
                1,
            ),
            mk(
                "m2",
                "Caroline and Melanie bought new running shoes in Denver.",
                2,
            ),
            mk(
                "m3",
                "Melanie told Caroline about the Boston marathon route.",
                3,
            ),
            mk(
                "m4",
                "Caroline painted a sunrise over the lake in Denver.",
                4,
            ),
            mk(
                "m5",
                "The Denver running club meets every Saturday morning.",
                5,
            ),
        ];
        let id_map = ingest_corpus(&manager, &corpus).expect("ingest");
        let gold = *id_map.get("gold").expect("gold ingested");
        let system = manager.get_user_memory(EVAL_USER).expect("system");
        // Harness ingest skips temporal facts (the HTTP handler stores them).
        let stored = system
            .read()
            .store_temporal_facts_for_memory(
                EVAL_USER,
                &MemoryId(gold),
                gold_text,
                &["Melanie".to_string()],
                base,
            )
            .expect("store facts");
        assert!(stored > 0, "fixture produced no temporal fact to ablate");

        let recall = |env: &[(&str, &str)]| -> Vec<(Uuid, f32)> {
            for (k, v) in env {
                std::env::set_var(k, v);
            }
            let out = system
                .read()
                .recall(&Query {
                    query_text: Some("When did Melanie run a marathon?".to_string()),
                    user_id: Some(EVAL_USER.to_string()),
                    max_results: 10,
                    layers: Some(LayerMode::Full),
                    ..Default::default()
                })
                .expect("recall");
            for (k, _) in env {
                std::env::remove_var(k);
            }
            out.iter()
                .map(|m| (m.id.0, m.score.unwrap_or(0.0)))
                .collect()
        };

        // veld recall is not bit-repeatable even under the harness pins (read-only recall,
        // frozen scoring clock): two identical passes drift in the fourth decimal. So the
        // bar for "this ablation did something" is measured against that drift rather than
        // assumed to be zero.
        let (first, second) = (recall(&[]), recall(&[]));
        assert!(!second.is_empty(), "baseline recall returned nothing");
        let ids = |v: &Vec<(Uuid, f32)>| v.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        assert_eq!(ids(&first), ids(&second), "baseline result set is unstable");
        let spread = |a: &Vec<(Uuid, f32)>, b: &Vec<(Uuid, f32)>| {
            a.iter()
                .zip(b.iter())
                .map(|((_, x), (_, y))| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let noise = spread(&first, &second);
        let tolerance = (noise * 10.0).max(1e-3);
        eprintln!("baseline repeat noise = {noise:e}, tolerance = {tolerance:e}");

        for (k, v) in [
            ("SHODH_DISABLE_FACT_LAYERS", "1"),
            ("SHODH_DISABLE_BOOSTS", ""),
            ("SHODH_GRAPH_FUSION_WEIGHT", "0"),
        ] {
            let arm = recall(&[(k, v)]);
            let moved = ids(&arm) != ids(&second) || spread(&arm, &second) > tolerance;
            assert!(
                moved,
                "{k}={v} moved no score past the {tolerance:e} repeat noise: the flag does not reach its code path\n  baseline {second:?}\n  arm      {arm:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
