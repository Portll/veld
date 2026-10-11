//! Fact-yield instrument.
//!
//! Nothing measured what fact distillation produces, so a store that yields no facts read the same
//! as one with nothing to distil. This seeds a store from a recall fixture corpus, runs
//! distillation, and prints:
//!
//!   * facts per 100 memories (yield),
//!   * active vs invalidated counts (is arbitration ever exercised?),
//!   * the full fact list, so quality is hand-checkable in the test output.
//!
//! It pins invariants (eligibility, idempotence), not yields: yield is a property of corpus content
//! and extraction policy, and pinning it would turn every deliberate policy change into a
//! test-fixing chore. The printed report is the instrument; read it when extraction behaviour is in
//! question:
//!
//!   cargo test --release --test fact_yield_report -- --nocapture
//!   cargo test --release --test fact_yield_report -- --ignored --nocapture
//!
//! A yield of zero on a corpus that restates nothing is the corroboration policy (a fact needs at
//! least two distinct source memories), not a plumbing failure; the `memories eligible` line tells
//! the two apart.

use tempfile::TempDir;

use veld::memory::compression::is_active;
use veld::memory::{Experience, ExperienceType, MemoryConfig, MemorySystem};

static ENV_INIT: std::sync::Once = std::sync::Once::new();

fn setup() -> (MemorySystem, TempDir) {
    // The detached cross-encoder pre-warm would race this short binary's exit.
    ENV_INIT.call_once(|| std::env::set_var("VELD_LAZY_CROSS_ENCODER", "1"));
    let temp_dir = TempDir::new().expect("temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        auto_compress: false,
        ..Default::default()
    };
    let system = MemorySystem::new(config, None).expect("memory system");
    (system, temp_dir)
}

fn exp_type(s: &str) -> ExperienceType {
    match s {
        "conversation" => ExperienceType::Conversation,
        "decision" => ExperienceType::Decision,
        "learning" => ExperienceType::Learning,
        "task" => ExperienceType::Task,
        "error" => ExperienceType::Error,
        "discovery" => ExperienceType::Discovery,
        "pattern" => ExperienceType::Pattern,
        _ => ExperienceType::Observation,
    }
}

/// Seed a store from a recall-fixture corpus (jsonl with content, memory_type, created_at),
/// keeping the fixture timestamps so the corpus is age-eligible as it would be on a mature store.
fn seed_corpus(system: &MemorySystem, path: &str) -> usize {
    let corpus =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("fixture corpus {path}: {e}"));
    let mut n = 0usize;
    for line in corpus.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).expect("jsonl line");
        let content = v["content"].as_str().expect("content").to_string();
        let mtype = v["memory_type"].as_str().unwrap_or("observation");
        let created_at = v["created_at"]
            .as_str()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&chrono::Utc));
        let e = Experience {
            content,
            experience_type: exp_type(mtype),
            ..Default::default()
        };
        system.remember(e, created_at).expect("remember");
        n += 1;
    }
    n
}

fn report(system: &MemorySystem, user: &str, corpus: &str, seeded: usize) {
    let result = system.distill_facts(user, 2, 7).expect("distill");
    let facts = system
        .fact_store()
        .list_filtered(user, 1000, true)
        .expect("facts");
    let now = chrono::Utc::now();
    let active = facts.iter().filter(|f| is_active(f, now)).count();
    let invalidated = facts.len() - active;

    println!("── fact yield: {corpus} ──");
    println!("memories seeded:        {seeded}");
    println!("memories eligible:      {}", result.memories_eligible);
    println!("facts stored:           {}", facts.len());
    println!("  active:               {active}");
    println!("  invalidated:          {invalidated}");
    println!(
        "yield (facts per 100):  {:.2}",
        facts.len() as f64 * 100.0 / seeded.max(1) as f64
    );
    for f in &facts {
        println!(
            "  [{:?}][active={}][sources={}][conf={:.2}] {}",
            f.fact_type,
            is_active(f, now),
            f.source_memories.len(),
            f.confidence,
            f.fact
        );
    }

    // Invariants that hold whatever the corpus: a mature corpus is fully eligible, and
    // re-distilling is a no-op.
    assert_eq!(
        result.memories_eligible, seeded,
        "a fixture corpus with historical timestamps must be fully eligible"
    );
    let before = facts.len();
    system.distill_facts(user, 2, 7).expect("re-distill");
    let after = system
        .fact_store()
        .list_filtered(user, 1000, true)
        .expect("facts")
        .len();
    assert_eq!(
        before, after,
        "re-distilling the same corpus must not change the fact count"
    );
}

/// Engineering-log corpus, the domain the specialist extractors (procedure, definition,
/// error-pattern, preference) were written for.
#[test]
fn smoke_corpus_fact_yield_report() {
    let (system, _dir) = setup();
    let seeded = seed_corpus(&system, "tests/recall/corpora/shodh-smoke.jsonl");
    report(
        &system,
        "yield-smoke",
        "shodh-smoke (engineering log)",
        seeded,
    );
}

/// Conversational corpus (LoCoMo gate): 629 memories embedded through the ONNX encoder at
/// remember() time, so minutes rather than seconds.
#[test]
#[ignore = "measurement harness: 629 memories through the ONNX embedder; run with --ignored when extraction behaviour is in question"]
fn locomo_gate_fact_yield_report() {
    let (system, _dir) = setup();
    let seeded = seed_corpus(&system, "tests/recall/corpora/locomo-gate.jsonl");
    report(
        &system,
        "yield-locomo",
        "locomo-gate (conversation)",
        seeded,
    );
}
