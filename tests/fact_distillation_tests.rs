//! Fact distillation over the full corpus.
//!
//! Extraction used to filter each cycle to memories newer than a watermark, then advance the
//! watermark over everything the filter passed, including memories the age gate excluded: a
//! memory seen once while too young was never extracted, and corroboration could not span two
//! cycles. Extraction now re-runs over the whole eligible corpus, and
//! `SemanticFactStore::ingest_candidate` makes re-deriving a stored fact from evidence it
//! already counts a no-op. These tests fail against the watermark.
//!
//! Memories here are at least four days old: veld's consolidator never takes a memory younger
//! than `HEAT_SCORE_MIN_AGE_DAYS` (3), whatever `min_age_days` says.

use tempfile::TempDir;

use veld::memory::compression::is_active;
use veld::memory::{Experience, ExperienceType, MemoryConfig, MemorySystem, SemanticFact};

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

fn declarative(content: &str) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    }
}

/// `days` in the past at whole milliseconds, so the time round-trips through storage exactly.
fn days_ago(days: i64) -> chrono::DateTime<chrono::Utc> {
    let t = chrono::Utc::now() - chrono::Duration::days(days);
    chrono::DateTime::from_timestamp_millis(t.timestamp_millis()).expect("valid timestamp")
}

fn all_facts(system: &MemorySystem, user: &str) -> Vec<SemanticFact> {
    system
        .fact_store()
        .list_filtered(user, 100, true)
        .expect("facts")
}

/// A run that the age gate leaves empty must not consume the corpus: the retry with the gate
/// lowered still sees every memory. Under the watermark it returned zero, forever.
#[test]
fn distill_retry_after_an_age_gated_run_still_mints() {
    let (system, _dir) = setup();
    let user = "burn-regression";
    for content in [
        "The staging cluster runs on four nodes in the Frankfurt region.",
        "The staging cluster in the Frankfurt region is running on four nodes.",
    ] {
        system
            .remember(declarative(content), Some(days_ago(5)))
            .expect("remember");
    }

    let first = system.distill_facts(user, 2, 7).expect("first distill");
    assert_eq!(
        first.memories_eligible, 0,
        "five-day-old memories miss a 7-day gate"
    );
    assert_eq!(first.facts_extracted, 0);

    let second = system.distill_facts(user, 2, 3).expect("second distill");
    assert_eq!(second.memories_eligible, 2);
    assert!(
        system
            .get_facts(user, 10)
            .expect("facts")
            .iter()
            .any(|f| f.fact.contains("four nodes")),
        "the retry must mint the corroborated statement, got {:?}",
        all_facts(&system, user)
    );
}

/// Memories that arrive behind an earlier heavy cycle (an import with historical times) are
/// extracted by the next one.
#[test]
fn maintenance_extracts_memories_that_arrive_behind_a_prior_cycle() {
    let (system, _dir) = setup();
    let user = "maintenance-backfill";
    system
        .remember(
            declarative("Deploy window for the search service is under review."),
            None,
        )
        .expect("remember young");
    system
        .run_maintenance(1.0, user, true)
        .expect("heavy maintenance on a young store");
    assert!(system.get_facts(user, 10).expect("facts").is_empty());

    for content in [
        "The ingestion pipeline writes batches to the events topic every thirty seconds.",
        "Every thirty seconds the ingestion pipeline writes batches into the events topic.",
    ] {
        system
            .remember(declarative(content), Some(days_ago(8)))
            .expect("remember backfill");
    }
    system
        .run_maintenance(1.0, user, true)
        .expect("heavy maintenance after backfill");

    let facts = system.get_facts(user, 10).expect("facts");
    assert!(
        facts.iter().any(|f| f.fact.contains("ingestion pipeline")),
        "a heavy cycle must extract eligible memories behind an earlier cycle, got {facts:?}"
    );
}

/// One statement does not mint; the same claim from a distinct memory on a later cycle does.
#[test]
fn corroboration_accumulates_across_distill_cycles() {
    let (system, _dir) = setup();
    let user = "cross-cycle";
    system
        .remember(
            declarative(
                "Payment retries were disabled on the checkout service after the duplicate charge incident.",
            ),
            Some(days_ago(8)),
        )
        .expect("remember");
    let first = system.distill_facts(user, 2, 7).expect("first distill");
    assert_eq!(
        first.facts_extracted, 0,
        "one statement is not corroborated"
    );

    system
        .remember(
            declarative(
                "After the duplicate charge incident the checkout service has payment retries disabled.",
            ),
            Some(days_ago(8)),
        )
        .expect("remember corroboration");
    system.distill_facts(user, 2, 7).expect("second distill");
    let facts = system.get_facts(user, 10).expect("facts");
    assert!(
        facts
            .iter()
            .any(|f| f.fact.to_lowercase().contains("payment retries")),
        "corroboration on a later cycle must mint the fact, got {facts:?}"
    );
}

/// A store seeded in one burst with a claim and its later correction: one distillation mints
/// both, since polarity keeps them in separate clusters, and the correction, backed by newer
/// evidence, supersedes the claim.
#[test]
fn a_bulk_seeded_correction_supersedes_its_claim_in_one_pass() {
    let (system, _dir) = setup();
    let user = "bulk-correction";
    for (content, age) in [
        (
            "Initial reports said four crew members were injured in the bridge collapse.",
            40,
        ),
        (
            "Early reports said four crew members were injured in the bridge collapse.",
            40,
        ),
        (
            "Corrected reports confirm no crew members were injured in the bridge collapse.",
            20,
        ),
        (
            "Later reports confirm no crew members were injured in the bridge collapse.",
            20,
        ),
    ] {
        system
            .remember(declarative(content), Some(days_ago(age)))
            .expect("remember");
    }

    let result = system.distill_facts(user, 2, 7).expect("distill");
    assert!(
        result.facts_extracted >= 2,
        "claim and correction must mint separately, got {}",
        result.facts_extracted
    );

    let facts = all_facts(&system, user);
    let claim = facts
        .iter()
        .find(|f| f.fact.contains("four crew members were injured"))
        .unwrap_or_else(|| panic!("claim fact missing: {facts:?}"));
    let correction = facts
        .iter()
        .find(|f| f.fact.contains("no crew members were injured"))
        .unwrap_or_else(|| panic!("correction fact missing: {facts:?}"));
    let now = chrono::Utc::now();
    assert!(
        is_active(correction, now),
        "the newer correction is believed"
    );
    assert!(
        !is_active(claim, now),
        "the claim is superseded, not deleted"
    );
    assert_eq!(claim.superseded_by.as_deref(), Some(correction.id.as_str()));
}

/// Distilling the same corpus again changes nothing: no duplicate rows, no support or
/// confidence gained from evidence already counted.
#[test]
fn repeated_distillation_is_idempotent() {
    let (system, _dir) = setup();
    let user = "idempotence";
    for content in [
        "The telemetry exporter batches spans every five seconds by default.",
        "By default the telemetry exporter batches spans every five seconds.",
    ] {
        system
            .remember(declarative(content), Some(days_ago(8)))
            .expect("remember");
    }

    system.distill_facts(user, 2, 7).expect("first distill");
    let first = all_facts(&system, user);
    assert_eq!(first.len(), 1, "one corroborated fact: {first:?}");

    for _ in 0..3 {
        system.distill_facts(user, 2, 7).expect("re-distill");
        system
            .run_maintenance(1.0, user, true)
            .expect("heavy maintenance");
    }
    let rescanned = all_facts(&system, user);
    assert_eq!(rescanned.len(), 1, "re-scanning mints no duplicates");
    assert_eq!(rescanned[0].id, first[0].id);
    assert_eq!(rescanned[0].support_count, first[0].support_count);
    assert_eq!(rescanned[0].confidence, first[0].confidence);
    assert_eq!(rescanned[0].last_reinforced, first[0].last_reinforced);
}
