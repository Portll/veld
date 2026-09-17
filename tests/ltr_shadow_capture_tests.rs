//! R2 (LTR) shadow-capture integration tests.
//!
//! These run the FULL `recall` scoring path on a real `MemorySystem` and assert
//! the retrieval-trace capture is SHADOW-ONLY: it never changes what `recall`
//! returns, and the captured rows are a faithful, 1:1 read-out of the delivered
//! result (same final scores, same order/rank). They also assert the
//! recall -> feedback -> TRAINING join over the live `reinforce_recall` path.
//!
//! Why this shape proves the shadow invariant: the capture write-site reads the
//! already-finalized `memories` vector (after competition / MMR / demotion /
//! re-cue) and the feature side-map, then persists. It holds no mutable handle on
//! `memories`. So if the captured `final_score`/`rank` for every returned memory
//! exactly mirror the returned `score`/position, the capture cannot have perturbed
//! the result it mirrors. (`features.unified_final` is intentionally the
//! pre-demotion closure output and is NOT expected to equal the final score.)

use std::sync::Arc;

use tempfile::TempDir;
use veld::memory::{
    Experience, ExperienceType, MemoryConfig, MemorySystem, Query, RetrievalOutcome,
};

fn setup() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        collective_store_dir: None,
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 200,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.7,
    };
    let memory = MemorySystem::new(config, None).expect("memory system");
    (memory, temp_dir)
}

fn experience(content: &str) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    }
}

fn seed(memory: &mut MemorySystem) {
    for c in [
        "Robot detected obstacle at the warehouse entrance during patrol",
        "Drone completed the perimeter patrol route without incident",
        "Obstacle avoidance system flagged a pallet blocking aisle three",
        "Quarterly OKRs prioritize warehouse automation and safety",
        "Safety protocol requires obstacle clearance before robot transit",
    ] {
        memory.remember(experience(c), None).expect("remember");
    }
}

/// SHADOW INVARIANT (the pass/fail gate): for every returned memory that was
/// scored through the unified closure, the captured PENDING trace row mirrors the
/// DELIVERED result exactly — same final score, same rank/order. Capture is a
/// faithful read-out, never a parallel path that could perturb scores or ordering.
#[test]
fn test_capture_is_shadow_faithful_to_returned_result() {
    let (mut memory, _t) = setup();
    seed(&mut memory);

    let query = Query {
        query_text: Some("obstacle detection during patrol".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory.recall(&query).expect("recall");
    assert!(!results.is_empty(), "expected some recalled memories");

    let pending = memory.retrieval_trace_store().load_pending();
    assert!(
        !pending.is_empty(),
        "capture should have emitted PENDING trace rows for scored memories"
    );

    // Build an id -> trace lookup. Cardinality: at most one PENDING row per
    // (recall, memory) — a single recall must not double-capture a memory.
    use std::collections::HashMap;
    let mut by_id: HashMap<_, _> = HashMap::new();
    for t in &pending {
        assert!(
            by_id.insert(t.memory_id.clone(), t).is_none(),
            "one trace row per (recall, memory): memory {:?} captured twice",
            t.memory_id
        );
    }

    // For each returned memory that carries a captured trace, the trace must
    // mirror the delivered (score, rank) exactly. Re-cued augmentation rows that
    // bypass the closure legitimately have no trace and are skipped.
    let mut mirrored = 0usize;
    for (rank, m) in results.iter().enumerate() {
        if let Some(t) = by_id.get(&m.id) {
            let returned_score = m.score.expect("returned memory has a score");
            assert!(
                (t.final_score - returned_score).abs() < 1e-6,
                "captured final_score {} must equal delivered score {} for {:?}",
                t.final_score,
                returned_score,
                m.id
            );
            assert_eq!(
                t.rank, rank,
                "captured rank must equal the delivered position for {:?}",
                m.id
            );
            assert_eq!(
                t.query_text, "obstacle detection during patrol",
                "captured query text mirrors the recall query"
            );
            mirrored += 1;
        }
    }
    assert!(
        mirrored > 0,
        "at least one delivered memory must have a faithful captured trace"
    );
}

/// Capture is deterministic and cannot reorder: two identical systems given the
/// same seed + query return the SAME (id, score) sequence. Guards against capture
/// introducing any nondeterminism or reordering into the delivered result.
#[test]
fn test_capture_does_not_perturb_order_or_scores() {
    let (mut a, _ta) = setup();
    let (mut b, _tb) = setup();
    seed(&mut a);
    seed(&mut b);

    let query = Query {
        query_text: Some("warehouse obstacle safety".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let ra = a.recall(&query).expect("recall a");
    let rb = b.recall(&query).expect("recall b");

    let seq_a: Vec<(String, f32)> = ra
        .iter()
        .map(|m| (m.experience.content.clone(), m.score.unwrap_or(0.0)))
        .collect();
    let seq_b: Vec<(String, f32)> = rb
        .iter()
        .map(|m| (m.experience.content.clone(), m.score.unwrap_or(0.0)))
        .collect();

    assert_eq!(
        seq_a.len(),
        seq_b.len(),
        "identical inputs yield identical result cardinality"
    );
    for (x, y) in seq_a.iter().zip(seq_b.iter()) {
        assert_eq!(x.0, y.0, "result ORDER is stable under capture");
        assert!(
            (x.1 - y.1).abs() < 1e-6,
            "result SCORES are stable under capture ({} vs {})",
            x.1,
            y.1
        );
    }
}

/// recall -> reinforce_recall(Helpful) graduates the captured PENDING rows for the
/// fed-back memories to TRAINING, stamped with the outcome label. The live
/// feedback path correlates by memory_id (it carries no retrieval/trace id), which
/// is exactly the join key `reinforce_recall` passes through.
#[test]
fn test_recall_then_feedback_graduates_to_training() {
    let (mut memory, _t) = setup();
    seed(&mut memory);

    let query = Query {
        query_text: Some("obstacle clearance safety protocol".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory.recall(&query).expect("recall");
    assert!(!results.is_empty());

    let store = memory.retrieval_trace_store();
    let pending_before = store.pending_count();
    assert!(pending_before > 0, "recall captured pending rows");
    assert_eq!(
        store.training_count(),
        0,
        "no training rows before feedback"
    );

    // Feed back a positive outcome for the top returned memories — the live path.
    let fed: Vec<_> = results.iter().take(2).map(|m| m.id.clone()).collect();
    let stats = memory
        .reinforce_recall(&fed, RetrievalOutcome::Helpful)
        .expect("reinforce");
    assert_eq!(stats.outcome, RetrievalOutcome::Helpful);

    // The fed-back memories' pending rows should have graduated to TRAINING with
    // the Helpful label.
    let training = memory.retrieval_trace_store().load_training();
    assert!(
        !training.is_empty(),
        "feedback should graduate matching pending rows to TRAINING"
    );
    for t in &training {
        assert_eq!(t.outcome, Some(RetrievalOutcome::Helpful));
        assert!(t.labeled_at.is_some());
        assert!(
            fed.contains(&t.memory_id),
            "only fed-back memories graduate"
        );
    }
    // Graduated rows left the PENDING stage.
    assert!(
        memory.retrieval_trace_store().pending_count() < pending_before,
        "graduated rows are removed from PENDING"
    );
}
