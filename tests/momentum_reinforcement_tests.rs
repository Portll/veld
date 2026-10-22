//! Feedback-momentum writes on the reinforcement path, ported from upstream #434.
//!
//! The momentum EMA is retrieval signal 11. It is reached two ways:
//!
//! 1. Explicitly, by a caller asserting an outcome through `POST /api/reinforce`, which runs
//!    `MemorySystem::reinforce_recall_from` with `MomentumPolicy::Apply`, as
//!    `reinforce_recall` does.
//! 2. Implicitly, by `proactive_context`, which applies its own graded signal per memory and
//!    then hands the memories it classified to `reinforce_recall_from` with
//!    `MomentumPolicy::Skip`. That second stage must not land a second signal for the same
//!    observation.

use std::sync::Arc;

use parking_lot::RwLock;
use tempfile::TempDir;

use veld::memory::{
    types::{Experience, ExperienceType},
    FeedbackStore, MemoryConfig, MemoryId, MemorySystem, MomentumPolicy, RetrievalOutcome,
    SignalRecord, SignalTrigger,
};

fn config(path: &std::path::Path) -> MemoryConfig {
    MemoryConfig {
        storage_path: path.to_path_buf(),
        collective_store_dir: None,
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    }
}

/// A memory system with a feedback store wired up, as the server's per-user open does.
fn system_with_feedback() -> (MemorySystem, Arc<RwLock<FeedbackStore>>, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut system = MemorySystem::new(config(temp_dir.path()), None).expect("memory system");
    let store = Arc::new(RwLock::new(FeedbackStore::new()));
    system.set_feedback_store(store.clone());
    (system, store, temp_dir)
}

fn remember(system: &MemorySystem, content: &str) -> MemoryId {
    let exp = Experience {
        experience_type: ExperienceType::Learning,
        content: content.to_string(),
        ..Default::default()
    };
    system.remember(exp, None).expect("remember")
}

/// The graded signal `proactive_context` applies before it classifies memories. An entity
/// overlap of 0.8 classifies as helpful, so the memory is then passed on for reinforcement.
fn graded_implicit_signal() -> SignalRecord {
    SignalRecord::from_entity_overlap(0.8)
}

#[test]
fn explicit_helpful_reinforcement_moves_momentum_positive() {
    let (system, store, _tmp) = system_with_feedback();
    let id = remember(&system, "Vamana graph traversal uses beam search");
    assert!(
        store.read().get_momentum(&id).is_none(),
        "no momentum exists before any feedback"
    );

    system
        .reinforce_recall(std::slice::from_ref(&id), RetrievalOutcome::Helpful)
        .expect("reinforce");

    let momentum = store.read().get_momentum(&id).expect("momentum");
    assert!(
        momentum.ema > 0.0,
        "helpful reinforcement moves momentum positive, got {}",
        momentum.ema
    );
    assert_eq!(momentum.signal_count, 1, "exactly one signal recorded");
}

#[test]
fn explicit_misleading_reinforcement_moves_momentum_negative() {
    let (system, store, _tmp) = system_with_feedback();
    let id = remember(&system, "Vamana alpha is applied during construction");

    system
        .reinforce_recall(std::slice::from_ref(&id), RetrievalOutcome::Misleading)
        .expect("reinforce");

    let momentum = store.read().get_momentum(&id).expect("momentum");
    assert!(
        momentum.ema < 0.0,
        "misleading reinforcement moves momentum negative, got {}",
        momentum.ema
    );
    assert_eq!(momentum.signal_count, 1);
}

#[test]
fn explicit_neutral_reinforcement_writes_no_momentum() {
    // Neutral records an access, not usefulness either way, so the EMA has no evidence to
    // move on.
    let (system, store, _tmp) = system_with_feedback();
    let id = remember(&system, "Neutral observation with no outcome");

    system
        .reinforce_recall(std::slice::from_ref(&id), RetrievalOutcome::Neutral)
        .expect("reinforce");

    assert!(
        store.read().get_momentum(&id).is_none(),
        "neutral creates no momentum entry"
    );
}

#[test]
fn explicit_reinforcement_records_the_explicit_outcome_trigger() {
    // The trigger says which path wrote a signal: every other variant records an inference
    // the system drew, this one an outcome a caller asserted.
    let (system, store, _tmp) = system_with_feedback();
    let id = remember(&system, "Explicit outcome trigger provenance");

    system
        .reinforce_recall(std::slice::from_ref(&id), RetrievalOutcome::Helpful)
        .expect("reinforce");

    let momentum = store.read().get_momentum(&id).expect("momentum");
    match momentum.recent_signals.back().map(|s| &s.trigger) {
        Some(SignalTrigger::ExplicitOutcome { outcome }) => assert_eq!(outcome, "helpful"),
        other => panic!("expected an ExplicitOutcome trigger, got {other:?}"),
    }
}

#[test]
fn repeated_explicit_reinforcement_accumulates_momentum() {
    // The EMA is inertia-damped: repeated use builds momentum without saturating on the
    // first signal.
    let (system, store, _tmp) = system_with_feedback();
    let id = remember(&system, "Repeatedly useful memory");

    let mut previous = 0.0_f32;
    for round in 1..=5 {
        system
            .reinforce_recall(std::slice::from_ref(&id), RetrievalOutcome::Helpful)
            .expect("reinforce");
        let ema = store.read().get_momentum(&id).expect("momentum").ema;
        assert!(
            ema > previous,
            "round {round}: EMA must increase, {previous} -> {ema}"
        );
        assert!(ema < 1.0, "round {round}: EMA must not saturate, got {ema}");
        previous = ema;
    }
}

#[test]
fn implicit_path_applies_exactly_one_momentum_signal() {
    // The proactive_context sequence: the graded signal, then the classified ids handed to
    // the reinforcement stage, which must not add a second signal for the same evidence.
    let (system, store, _tmp) = system_with_feedback();
    let id = remember(&system, "Surfaced by proactive context");

    let ema_after_graded = {
        let mut guard = store.write();
        let momentum = guard.get_or_create_momentum(id.clone(), ExperienceType::Context);
        momentum.update(graded_implicit_signal());
        momentum.ema
    };

    system
        .reinforce_recall_from(
            std::slice::from_ref(&id),
            RetrievalOutcome::Helpful,
            "session-implicit",
            MomentumPolicy::Skip,
        )
        .expect("reinforce");

    let momentum = store.read().get_momentum(&id).expect("momentum");
    assert_eq!(
        momentum.signal_count, 1,
        "one observation produces one momentum signal, not two"
    );
    assert_eq!(
        momentum.ema, ema_after_graded,
        "the graded EMA survives the reinforcement stage untouched"
    );
    assert!(
        matches!(
            momentum.recent_signals.back().map(|s| &s.trigger),
            Some(SignalTrigger::EntityOverlap { .. })
        ),
        "the surviving signal is the graded implicit one"
    );
}

#[test]
fn apply_policy_would_double_count_the_implicit_path() {
    // The control for the test above: Apply after the graded signal is the second charge for
    // one observation that Skip exists to prevent.
    let (system, store, _tmp) = system_with_feedback();
    let id = remember(&system, "Would be double counted");

    {
        let mut guard = store.write();
        let momentum = guard.get_or_create_momentum(id.clone(), ExperienceType::Context);
        momentum.update(graded_implicit_signal());
    }

    system
        .reinforce_recall_from(
            std::slice::from_ref(&id),
            RetrievalOutcome::Helpful,
            "session-implicit",
            MomentumPolicy::Apply,
        )
        .expect("reinforce");

    let momentum = store.read().get_momentum(&id).expect("momentum");
    assert_eq!(
        momentum.signal_count, 2,
        "Apply after a graded signal records a second signal"
    );
}

#[test]
fn the_implicit_reinforcement_stage_still_boosts_importance() {
    // The reinforcement stage owes the implicit path its importance boost, coactivation and
    // entity-edge updates even though momentum was written before it.
    let (system, store, _tmp) = system_with_feedback();
    let id = remember(&system, "Importance still moves on the implicit path");
    let before = system.get_memory(&id).expect("stored memory").importance();

    let stats = system
        .reinforce_recall_from(
            std::slice::from_ref(&id),
            RetrievalOutcome::Helpful,
            "session-implicit",
            MomentumPolicy::Skip,
        )
        .expect("reinforce");

    assert_eq!(stats.importance_boosts, 1, "importance is boosted");
    let after = system.get_memory(&id).expect("stored memory").importance();
    assert!(
        after > before,
        "the boost is persisted: importance {before} -> {after}"
    );
    assert!(
        store.read().get_momentum(&id).is_none(),
        "Skip creates no momentum entry"
    );
}

// FeedbackStore::with_persistence opens a standalone RocksDB store.
#[cfg(feature = "rocksdb")]
#[test]
fn explicit_reinforcement_momentum_survives_restart() {
    // A deployment driven only through POST /api/reinforce never reaches the flush that
    // proactive_context performs, so the momentum must be written through at reinforcement.
    let store_dir = TempDir::new().expect("temp dir");
    let system_dir = TempDir::new().expect("temp dir");

    let id = {
        let mut system = MemorySystem::new(config(system_dir.path()), None).expect("memory system");
        let store = Arc::new(RwLock::new(
            FeedbackStore::with_persistence(store_dir.path()).expect("persistent store"),
        ));
        system.set_feedback_store(store.clone());

        let id = remember(&system, "Reinforced only through the explicit API");
        system
            .reinforce_recall(std::slice::from_ref(&id), RetrievalOutcome::Helpful)
            .expect("reinforce");
        id
    };

    let reopened = FeedbackStore::with_persistence(store_dir.path()).expect("reopen store");
    let momentum = reopened.get_momentum(&id).expect("momentum");
    assert!(
        momentum.ema > 0.0,
        "restored momentum retains the helpful signal, got {}",
        momentum.ema
    );
    assert_eq!(momentum.signal_count, 1);
}
