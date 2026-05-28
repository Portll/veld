//! rlm-eval — MVP evaluator for RLM-as-refiner vs cross-encoder rerank.
//!
//! Compares three variants over a fixed corpus + query set:
//!
//! - `baseline`: hybrid search + cross-encoder rerank (production default).
//! - `a`:        hybrid search + RLM refiner (cross-encoder bypassed).
//! - `stacked`:  hybrid search + cross-encoder + RLM refiner.
//!
//! The RLM refiner is configured via `VELD_RLM_ENDPOINT`, `VELD_RLM_API_KEY`,
//! and `VELD_RLM_MODEL`. If the endpoint is unset, `a` degrades to RRF-only
//! and `stacked` degrades to cross-encoder-only — the run still completes
//! and is labeled as such in the output so degenerate results are obvious.
//!
//! Metrics: Recall@5, Recall@10, MRR, mean wall-clock latency per query.
//! Per-class breakdown matches the dataset's `class` field on each query.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use veld::embeddings::{load_primary_embedder, Embedder};
use veld::memory::hybrid_search::{HybridSearchConfig, HybridSearchEngine, RefinerMode};
use veld::memory::rlm_refiner::RlmRefiner;
use veld::memory::types::MemoryId;

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Variant {
    Baseline,
    A,
    Stacked,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::Baseline => "baseline",
            Variant::A => "a",
            Variant::Stacked => "stacked",
        }
    }

    fn refiner_mode(self) -> RefinerMode {
        match self {
            Variant::Baseline => RefinerMode::CrossEncoder,
            Variant::A => RefinerMode::Rlm,
            Variant::Stacked => RefinerMode::Stacked,
        }
    }
}

#[derive(Parser)]
#[command(about = "MVP evaluator for RLM refiner vs cross-encoder rerank")]
struct Args {
    /// Path to the eval dataset JSON
    #[arg(long)]
    dataset: PathBuf,

    /// Variants to run, comma-separated
    #[arg(long, value_delimiter = ',', default_value = "baseline,a,stacked")]
    variants: Vec<Variant>,

    /// Top-K candidates passed to the refiner / cross-encoder
    #[arg(long, default_value_t = 10)]
    topk: usize,

    /// Embedder identifier (e.g. "minilm", "nomic", or http URL)
    #[arg(long, default_value = "minilm")]
    embedder: String,

    /// Maximum queries to evaluate (0 = all)
    #[arg(long, default_value_t = 0)]
    queries: usize,

    /// Number of candidates retrieved from each retriever before fusion
    #[arg(long, default_value_t = 50)]
    candidate_count: usize,

    /// Write JSON results to this path (stdout if omitted)
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Dataset {
    version: String,
    name: String,
    #[serde(default)]
    description: String,
    memories: Vec<DatasetMemory>,
    queries: Vec<DatasetQuery>,
}

#[derive(Deserialize)]
struct DatasetMemory {
    id: usize,
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    entities: Vec<String>,
}

#[derive(Deserialize)]
struct DatasetQuery {
    id: usize,
    query: String,
    #[serde(default)]
    class: String,
    expected_memory_ids: Vec<usize>,
    #[serde(default)]
    acceptable_memory_ids: Vec<usize>,
}

#[derive(Serialize, Default, Clone)]
struct ClassMetrics {
    n: usize,
    recall_at_5: f64,
    recall_at_10: f64,
    /// Recall@10 against the acceptable set (expected ∪ acceptable).
    /// Lets the dataset author mark "also-fine" answers so a strict miss
    /// against `expected_memory_ids` still credits a plausibly correct
    /// retrieval.
    lenient_recall_at_10: f64,
    mrr: f64,
}

#[derive(Serialize)]
struct VariantResult {
    variant: String,
    refiner_mode: String,
    refiner_attached: bool,
    queries_run: usize,
    recall_at_5: f64,
    recall_at_10: f64,
    lenient_recall_at_10: f64,
    mrr: f64,
    mean_latency_ms: f64,
    per_class: HashMap<String, ClassMetrics>,
}

#[derive(Serialize)]
struct RunReport {
    dataset_name: String,
    dataset_version: String,
    embedder: String,
    topk: usize,
    candidate_count: usize,
    queries_total: usize,
    rlm_model: Option<String>,
    rlm_endpoint_present: bool,
    variants: Vec<VariantResult>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::try_init().ok();
    let args = Args::parse();

    let dataset = load_dataset(&args.dataset)?;
    eprintln!(
        "Loaded dataset '{}' v{} — {} memories, {} queries",
        dataset.name,
        dataset.version,
        dataset.memories.len(),
        dataset.queries.len()
    );
    if !dataset.description.is_empty() {
        eprintln!("  {}", dataset.description);
    }

    let queries: Vec<&DatasetQuery> = if args.queries == 0 {
        dataset.queries.iter().collect()
    } else {
        dataset.queries.iter().take(args.queries).collect()
    };
    eprintln!("Evaluating {} queries", queries.len());

    // Assign deterministic-per-run MemoryId for every dataset memory.
    let id_to_memory: HashMap<usize, MemoryId> = dataset
        .memories
        .iter()
        .map(|m| (m.id, MemoryId(Uuid::new_v4())))
        .collect();
    let memory_to_id: HashMap<MemoryId, usize> = id_to_memory
        .iter()
        .map(|(&dsid, mid)| (mid.clone(), dsid))
        .collect();
    let content_map: HashMap<MemoryId, String> = dataset
        .memories
        .iter()
        .map(|m| (id_to_memory[&m.id].clone(), m.content.clone()))
        .collect();

    eprintln!("Loading embedder '{}'...", args.embedder);
    let embedder: Arc<dyn Embedder> =
        load_primary_embedder(&args.embedder).context("failed to load embedder")?;

    eprintln!("Encoding {} memories...", dataset.memories.len());
    let mut mem_embeddings: HashMap<MemoryId, Vec<f32>> =
        HashMap::with_capacity(dataset.memories.len());
    for mem in &dataset.memories {
        let v = embedder
            .encode(&mem.content)
            .with_context(|| format!("encoding memory {}", mem.id))?;
        mem_embeddings.insert(id_to_memory[&mem.id].clone(), v);
    }

    eprintln!("Encoding {} queries...", queries.len());
    let mut query_embeddings: Vec<Vec<f32>> = Vec::with_capacity(queries.len());
    for q in &queries {
        let v = embedder
            .encode_for_query(&q.query)
            .with_context(|| format!("encoding query {}", q.id))?;
        query_embeddings.push(v);
    }

    let rlm_endpoint_present =
        std::env::var("VELD_RLM_ENDPOINT").is_ok_and(|v| !v.is_empty());
    let rlm_model = std::env::var("VELD_RLM_MODEL").ok();
    if !rlm_endpoint_present {
        eprintln!(
            "WARNING: VELD_RLM_ENDPOINT is not set. Variant 'a' will degrade to RRF-only and 'stacked' to cross-encoder-only. Set VELD_RLM_ENDPOINT (and VELD_RLM_API_KEY, VELD_RLM_MODEL) to exercise the refiner."
        );
    }

    let mut variant_results = Vec::with_capacity(args.variants.len());
    for &variant in &args.variants {
        eprintln!("\n=== Variant: {} ===", variant.label());
        let result = run_variant(
            variant,
            &dataset,
            &queries,
            &id_to_memory,
            &memory_to_id,
            &content_map,
            &mem_embeddings,
            &query_embeddings,
            embedder.clone(),
            args.topk,
            args.candidate_count,
        )?;
        eprintln!(
            "  refiner_attached={}  recall@5={:.3}  recall@10={:.3}  lenient@10={:.3}  MRR={:.3}  mean_latency={:.0}ms",
            result.refiner_attached,
            result.recall_at_5,
            result.recall_at_10,
            result.lenient_recall_at_10,
            result.mrr,
            result.mean_latency_ms,
        );
        if !result.per_class.is_empty() {
            let mut class_names: Vec<&String> = result.per_class.keys().collect();
            class_names.sort();
            for cls in class_names {
                let m = &result.per_class[cls];
                eprintln!(
                    "    [{cls}] n={}  r@5={:.3}  r@10={:.3}  MRR={:.3}",
                    m.n, m.recall_at_5, m.recall_at_10, m.mrr,
                );
            }
        }
        variant_results.push(result);
    }

    let report = RunReport {
        dataset_name: dataset.name.clone(),
        dataset_version: dataset.version.clone(),
        embedder: args.embedder.clone(),
        topk: args.topk,
        candidate_count: args.candidate_count,
        queries_total: queries.len(),
        rlm_model,
        rlm_endpoint_present,
        variants: variant_results,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.out {
        std::fs::write(path, &json)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("\nResults written to {}", path.display());
    } else {
        println!("{json}");
    }

    Ok(())
}

fn load_dataset(path: &Path) -> Result<Dataset> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

#[allow(clippy::too_many_arguments)]
fn run_variant(
    variant: Variant,
    dataset: &Dataset,
    queries: &[&DatasetQuery],
    id_to_memory: &HashMap<usize, MemoryId>,
    memory_to_id: &HashMap<MemoryId, usize>,
    content_map: &HashMap<MemoryId, String>,
    mem_embeddings: &HashMap<MemoryId, Vec<f32>>,
    query_embeddings: &[Vec<f32>],
    embedder: Arc<dyn Embedder>,
    topk: usize,
    candidate_count: usize,
) -> Result<VariantResult> {
    let bm25_dir = std::env::temp_dir().join(format!("rlm_eval_bm25_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&bm25_dir).context("creating BM25 scratch dir")?;

    let config = HybridSearchConfig {
        refiner_mode: variant.refiner_mode(),
        rerank_count: topk,
        candidate_count,
        ..HybridSearchConfig::default()
    };

    let mut engine = HybridSearchEngine::new(&bm25_dir, embedder.clone(), config)?;

    // Attach the RLM refiner for variants that need it. If the env isn't
    // configured, leave it unattached; the engine then degrades cleanly per
    // RefinerMode semantics (Rlm -> RRF-only; Stacked -> cross-encoder-only).
    let mut refiner_attached = false;
    if matches!(variant.refiner_mode(), RefinerMode::Rlm | RefinerMode::Stacked) {
        if let Some(refiner) = RlmRefiner::from_env()? {
            engine = engine.with_refiner(Box::new(refiner));
            refiner_attached = true;
        }
    }

    for mem in &dataset.memories {
        let mid = &id_to_memory[&mem.id];
        engine.index_memory(mid, &mem.content, &mem.tags, &mem.entities)?;
    }
    engine.commit_and_reload()?;

    let get_content = |mid: &MemoryId| -> Option<String> { content_map.get(mid).cloned() };

    let mut total_recall_5 = 0.0_f64;
    let mut total_recall_10 = 0.0_f64;
    let mut total_lenient_10 = 0.0_f64;
    let mut total_mrr = 0.0_f64;
    let mut total_latency_ms = 0.0_f64;
    let mut per_class: HashMap<String, ClassMetrics> = HashMap::new();

    for (qi, q) in queries.iter().enumerate() {
        let q_vec = &query_embeddings[qi];
        let mut vec_scores: Vec<(MemoryId, f32)> = mem_embeddings
            .iter()
            .map(|(mid, mv)| (mid.clone(), cosine(q_vec, mv)))
            .collect();
        vec_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        vec_scores.truncate(candidate_count);

        let start = Instant::now();
        let results = engine.search(&q.query, vec_scores, get_content)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_latency_ms += elapsed_ms;

        let result_ds_ids: Vec<usize> = results
            .iter()
            .take(topk.max(10))
            .filter_map(|r| memory_to_id.get(&r.memory_id).copied())
            .collect();

        let expected: HashSet<usize> = q.expected_memory_ids.iter().copied().collect();
        let lenient: HashSet<usize> = q
            .expected_memory_ids
            .iter()
            .chain(q.acceptable_memory_ids.iter())
            .copied()
            .collect();
        let recall_5 = recall_at(&result_ds_ids, &expected, 5);
        let recall_10 = recall_at(&result_ds_ids, &expected, 10);
        let lenient_10 = recall_at(&result_ds_ids, &lenient, 10);
        let mrr = mrr_score(&result_ds_ids, &expected);

        total_recall_5 += recall_5;
        total_recall_10 += recall_10;
        total_lenient_10 += lenient_10;
        total_mrr += mrr;

        let entry = per_class.entry(q.class.clone()).or_default();
        entry.n += 1;
        entry.recall_at_5 += recall_5;
        entry.recall_at_10 += recall_10;
        entry.lenient_recall_at_10 += lenient_10;
        entry.mrr += mrr;
    }

    let n = queries.len() as f64;
    let per_class_finalized: HashMap<String, ClassMetrics> = per_class
        .into_iter()
        .map(|(k, m)| {
            let cn = m.n as f64;
            (
                k,
                ClassMetrics {
                    n: m.n,
                    recall_at_5: m.recall_at_5 / cn,
                    recall_at_10: m.recall_at_10 / cn,
                    lenient_recall_at_10: m.lenient_recall_at_10 / cn,
                    mrr: m.mrr / cn,
                },
            )
        })
        .collect();

    let result = VariantResult {
        variant: variant.label().to_string(),
        refiner_mode: format!("{:?}", variant.refiner_mode()),
        refiner_attached,
        queries_run: queries.len(),
        recall_at_5: total_recall_5 / n,
        recall_at_10: total_recall_10 / n,
        lenient_recall_at_10: total_lenient_10 / n,
        mrr: total_mrr / n,
        mean_latency_ms: total_latency_ms / n,
        per_class: per_class_finalized,
    };

    // Best-effort cleanup. We don't fail the run if the directory persists —
    // it lives in the system temp dir and will be reclaimed.
    drop(engine);
    let _ = std::fs::remove_dir_all(&bm25_dir);

    Ok(result)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn recall_at(ranked: &[usize], expected: &HashSet<usize>, k: usize) -> f64 {
    if expected.is_empty() {
        return 0.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|id| expected.contains(id))
        .count();
    hits as f64 / expected.len() as f64
}

fn mrr_score(ranked: &[usize], expected: &HashSet<usize>) -> f64 {
    for (rank, id) in ranked.iter().enumerate() {
        if expected.contains(id) {
            return 1.0 / (rank + 1) as f64;
        }
    }
    0.0
}
