//! Dataset-backed retrieval benchmark using nexus_benchmark_v2.json.
//!
//! Extends the smaller LOCOMO-style benchmark with a reviewed corpus containing
//! more query types, explicit absence constraints, contradiction cases, and
//! person-focused questions.
//!
//! Run with: cargo test --test nexus_benchmark -- --ignored --nocapture

#[allow(dead_code)]
mod dataset_loader;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use dataset_loader::{BenchmarkDataset, DatasetMemory, DatasetQuery};
use veld::memory::types::{Experience, ExperienceType, Query};
use veld::memory::{MemoryConfig, MemoryId, MemorySystem};
use tempfile::TempDir;

const OVERALL_COMPOSITE_THRESHOLD: f32 = 0.45;
const OVERALL_MRR_THRESHOLD: f32 = 0.45;
const OVERALL_RECALL_AT_5_THRESHOLD: f32 = 0.45;
const ABSENCE_COMPLIANCE_THRESHOLD: f32 = 0.80;
const MAX_QUERY_LATENCY_MS: u64 = 1_000;

#[derive(Debug)]
struct QueryEvaluation {
    query_id: usize,
    query_text: String,
    query_type: String,
    retrieved_ids: Vec<usize>,
    acceptable_ids: Vec<usize>,
    expected_ids: Vec<usize>,
    absence_ids: Vec<usize>,
    precision_at_5: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    reciprocal_rank: f32,
    absence_compliance: f32,
    latency_ms: u64,
}

#[derive(Debug)]
struct AggregateMetrics {
    precision_at_5: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    mrr: f32,
    absence_compliance: f32,
    composite: f32,
    avg_latency_ms: u64,
}

fn experience_type_from_str(value: &str) -> ExperienceType {
    match value {
        "Conversation" => ExperienceType::Conversation,
        "Decision" => ExperienceType::Decision,
        "Error" => ExperienceType::Error,
        "Learning" => ExperienceType::Learning,
        "Discovery" => ExperienceType::Discovery,
        "Pattern" => ExperienceType::Pattern,
        "Context" => ExperienceType::Context,
        "Task" => ExperienceType::Task,
        "CodeEdit" => ExperienceType::CodeEdit,
        "FileAccess" => ExperienceType::FileAccess,
        "Search" => ExperienceType::Search,
        "Command" => ExperienceType::Command,
        "Observation" => ExperienceType::Observation,
        "Intention" => ExperienceType::Intention,
        _ => ExperienceType::Observation,
    }
}

fn remember_dataset_memory(system: &MemorySystem, memory: &DatasetMemory) -> MemoryId {
    let mut metadata = HashMap::new();
    metadata.insert("importance_hint".to_string(), memory.importance.to_string());
    metadata.insert("session".to_string(), memory.session.to_string());
    if !memory.narrative_notes.is_empty() {
        metadata.insert("narrative_notes".to_string(), memory.narrative_notes.clone());
    }

    let mut entities = memory.tags.clone();
    entities.extend(memory.person_mentions.clone());

    let experience = Experience {
        content: memory.content.clone(),
        experience_type: experience_type_from_str(&memory.experience_type),
        entities,
        tags: memory.tags.clone(),
        metadata,
        ..Default::default()
    };

    system
        .remember(experience, None)
        .expect("Failed to store dataset memory")
}

fn precision_at_k(retrieved: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let top_k = retrieved.iter().take(k);
    let hits = top_k.filter(|id| relevant.contains(id)).count() as f32;
    hits / k as f32
}

fn recall_at_k(retrieved: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
    if relevant.is_empty() {
        return 1.0;
    }
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count() as f32;
    hits / relevant.len() as f32
}

fn reciprocal_rank(retrieved: &[usize], relevant: &HashSet<usize>) -> f32 {
    for (index, memory_id) in retrieved.iter().enumerate() {
        if relevant.contains(memory_id) {
            return 1.0 / (index as f32 + 1.0);
        }
    }
    0.0
}

fn absence_compliance(retrieved: &[usize], absence: &HashSet<usize>, k: usize) -> f32 {
    if absence.is_empty() {
        return 1.0;
    }
    let violations = retrieved
        .iter()
        .take(k)
        .filter(|id| absence.contains(id))
        .count() as f32;
    1.0 - (violations / absence.len() as f32).min(1.0)
}

fn map_results_to_dataset_ids(
    results: &[veld::memory::SharedMemory],
    reverse_map: &HashMap<uuid::Uuid, usize>,
) -> Vec<usize> {
    results
        .iter()
        .filter_map(|memory| reverse_map.get(&memory.id.0).copied())
        .collect()
}

fn evaluate_query(
    system: &MemorySystem,
    query: &DatasetQuery,
    reverse_map: &HashMap<uuid::Uuid, usize>,
) -> QueryEvaluation {
    let recall_query = Query {
        query_text: Some(query.query.clone()),
        max_results: 10,
        ..Default::default()
    };

    let start = Instant::now();
    let results = system.recall(&recall_query).expect("Dataset recall failed");
    let latency_ms = start.elapsed().as_millis() as u64;
    let retrieved_ids = map_results_to_dataset_ids(&results, reverse_map);

    let relevant: HashSet<usize> = query.acceptable_memory_ids.iter().copied().collect();
    let absence: HashSet<usize> = query.absence_memory_ids.iter().copied().collect();

    QueryEvaluation {
        query_id: query.id,
        query_text: query.query.clone(),
        query_type: query.query_type.clone(),
        retrieved_ids,
        acceptable_ids: query.acceptable_memory_ids.clone(),
        expected_ids: query.expected_memory_ids.clone(),
        absence_ids: query.absence_memory_ids.clone(),
        precision_at_5: precision_at_k(&map_results_to_dataset_ids(&results, reverse_map), &relevant, 5),
        recall_at_5: recall_at_k(&map_results_to_dataset_ids(&results, reverse_map), &relevant, 5),
        recall_at_10: recall_at_k(&map_results_to_dataset_ids(&results, reverse_map), &relevant, 10),
        reciprocal_rank: reciprocal_rank(&map_results_to_dataset_ids(&results, reverse_map), &relevant),
        absence_compliance: absence_compliance(&map_results_to_dataset_ids(&results, reverse_map), &absence, 10),
        latency_ms,
    }
}

fn aggregate_metrics(dataset: &BenchmarkDataset, evaluations: &[QueryEvaluation]) -> AggregateMetrics {
    let n = evaluations.len().max(1) as f32;
    let weights = &dataset.scoring.composite_weights;

    let precision_at_5 = evaluations.iter().map(|e| e.precision_at_5).sum::<f32>() / n;
    let recall_at_5 = evaluations.iter().map(|e| e.recall_at_5).sum::<f32>() / n;
    let recall_at_10 = evaluations.iter().map(|e| e.recall_at_10).sum::<f32>() / n;
    let mrr = evaluations.iter().map(|e| e.reciprocal_rank).sum::<f32>() / n;
    let absence_compliance = evaluations.iter().map(|e| e.absence_compliance).sum::<f32>() / n;
    let avg_latency_ms = (evaluations.iter().map(|e| e.latency_ms).sum::<u64>() as f32 / n) as u64;

    let composite = (mrr * weights.mrr)
        + (recall_at_5 * weights.recall_at_5)
        + (recall_at_10 * weights.recall_at_10)
        + (precision_at_5 * weights.precision_at_5)
        + (absence_compliance * weights.absence_compliance);

    AggregateMetrics {
        precision_at_5,
        recall_at_5,
        recall_at_10,
        mrr,
        absence_compliance,
        composite,
        avg_latency_ms,
    }
}

fn print_summary(dataset: &BenchmarkDataset, overall: &AggregateMetrics, evaluations: &[QueryEvaluation]) {
    println!("\n=== NEXUS BENCHMARK V2 ===");
    println!("Dataset: {}", dataset.name);
    println!("Memories: {}", dataset.memories.len());
    println!("Queries: {}", dataset.queries.len());
    println!();
    println!("MRR:               {:.3}", overall.mrr);
    println!("Recall@5:          {:.3}", overall.recall_at_5);
    println!("Recall@10:         {:.3}", overall.recall_at_10);
    println!("Precision@5:       {:.3}", overall.precision_at_5);
    println!("AbsenceCompliance: {:.3}", overall.absence_compliance);
    println!("Composite:         {:.3}", overall.composite);
    println!("Avg latency:       {}ms", overall.avg_latency_ms);

    let max_latency = evaluations.iter().map(|e| e.latency_ms).max().unwrap_or(0);
    println!("Max latency:       {}ms", max_latency);

    println!("\nHardest misses:");
    for evaluation in evaluations
        .iter()
        .filter(|e| e.reciprocal_rank == 0.0 || e.absence_compliance < 1.0)
        .take(10)
    {
        println!(
            "  Q{:02} [{}] rr={:.2} abs={:.2} latency={}ms",
            evaluation.query_id,
            evaluation.query_type,
            evaluation.reciprocal_rank,
            evaluation.absence_compliance,
            evaluation.latency_ms,
        );
        println!("    {}", evaluation.query_text);
        println!("    acceptable={:?}", evaluation.acceptable_ids);
        println!("    expected={:?}", evaluation.expected_ids);
        println!("    absence={:?}", evaluation.absence_ids);
        println!("    retrieved={:?}", evaluation.retrieved_ids);
    }
}

#[test]
#[ignore = "dataset-backed benchmark"]
fn benchmark_nexus_dataset_v2() {
    let dataset_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("datasets")
        .join("nexus_benchmark_v2.json");

    let dataset = BenchmarkDataset::load(&dataset_path).expect("Failed to load benchmark dataset");
    let validation_errors = dataset.validate();
    let hard_errors: Vec<_> = validation_errors
        .iter()
        .filter(|error| error.severity == "ERROR")
        .collect();
    assert!(hard_errors.is_empty(), "Dataset validation failed: {hard_errors:?}");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        collective_store_dir: None,
        working_memory_size: 128,
        session_memory_size_mb: 64,
        max_heap_per_user_mb: 512,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.1,
    };
    let system = MemorySystem::new(config, None).expect("Failed to create memory system");

    let mut reverse_map: HashMap<uuid::Uuid, usize> = HashMap::new();
    for memory in &dataset.memories {
        let stored_id = remember_dataset_memory(&system, memory);
        reverse_map.insert(stored_id.0, memory.id);
    }

    let evaluations: Vec<QueryEvaluation> = dataset
        .queries
        .iter()
        .map(|query| evaluate_query(&system, query, &reverse_map))
        .collect();

    let overall = aggregate_metrics(&dataset, &evaluations);
    print_summary(&dataset, &overall, &evaluations);

    let max_latency = evaluations.iter().map(|e| e.latency_ms).max().unwrap_or(0);

    assert!(
        overall.composite >= OVERALL_COMPOSITE_THRESHOLD,
        "Composite score too low: {:.3} (threshold: {:.3})",
        overall.composite,
        OVERALL_COMPOSITE_THRESHOLD,
    );
    assert!(
        overall.mrr >= OVERALL_MRR_THRESHOLD,
        "Overall MRR too low: {:.3} (threshold: {:.3})",
        overall.mrr,
        OVERALL_MRR_THRESHOLD,
    );
    assert!(
        overall.recall_at_5 >= OVERALL_RECALL_AT_5_THRESHOLD,
        "Overall Recall@5 too low: {:.3} (threshold: {:.3})",
        overall.recall_at_5,
        OVERALL_RECALL_AT_5_THRESHOLD,
    );
    assert!(
        overall.absence_compliance >= ABSENCE_COMPLIANCE_THRESHOLD,
        "Absence compliance too low: {:.3} (threshold: {:.3})",
        overall.absence_compliance,
        ABSENCE_COMPLIANCE_THRESHOLD,
    );
    assert!(
        max_latency < MAX_QUERY_LATENCY_MS,
        "Max query latency too high: {}ms (threshold: {}ms)",
        max_latency,
        MAX_QUERY_LATENCY_MS,
    );
}