//! Branch-subsystem wiring, ported from upstream #301.
//!
//! A project-pivot signal must open a branch and anchor it into the edge graph with a
//! `BranchedFrom` edge tagged by the new `branch_id`, so the branch is reachable through
//! `trace()` and `find_root_cause()`.

use chrono::{DateTime, Duration, Utc};
use tempfile::TempDir;

use veld::memory::lineage::CausalRelation;
use veld::memory::types::{Experience, ExperienceType};
use veld::memory::{Memory, MemoryConfig, MemoryId, MemorySystem};
use veld::uuid::Uuid;

fn create_test_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        collective_store_dir: None,
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    };
    let system = MemorySystem::new(config, None).expect("memory system");
    (system, temp_dir)
}

fn mem(content: &str, created_at: DateTime<Utc>) -> Memory {
    let exp = Experience {
        experience_type: ExperienceType::Decision,
        content: content.to_string(),
        ..Default::default()
    };
    Memory::new(
        MemoryId(Uuid::new_v4()),
        exp,
        0.7,
        None,
        None,
        None,
        Some(created_at),
    )
}

#[test]
fn pivot_signal_opens_branch_and_anchors_branchedfrom_edge() {
    let (system, _tmp) = create_test_system();
    let user = "branch-user";
    let now = Utc::now();
    let prior = mem(
        "Working on the Vamana index integration for vector search.",
        now - Duration::days(1),
    );
    let pivot = mem(
        "We need to abandon this and pivot to a different strategy entirely.",
        now,
    );

    let edges = system
        .infer_lineage_for_memory(user, &pivot, std::slice::from_ref(&prior))
        .expect("infer lineage");

    let branches = system
        .lineage_graph()
        .list_branches(user)
        .expect("branches");
    let pivot_branch = branches
        .iter()
        .find(|b| b.id != "main")
        .unwrap_or_else(|| panic!("expected a pivot branch, got {branches:?}"));
    let bid = pivot_branch.id.clone();

    let branched = edges
        .iter()
        .find(|e| e.relation == CausalRelation::BranchedFrom)
        .unwrap_or_else(|| panic!("expected a BranchedFrom edge, got {edges:?}"));
    assert_eq!(branched.from, pivot.id);
    assert_eq!(branched.to, prior.id);
    assert_eq!(branched.branch_id.as_deref(), Some(bid.as_str()));

    // Persisted, not merely returned.
    let stored = system
        .lineage_graph()
        .get_edges_from(user, &pivot.id)
        .expect("edges from");
    assert!(stored
        .iter()
        .any(|e| e.relation == CausalRelation::BranchedFrom && e.branch_id.is_some()));
}

#[test]
fn reinferring_a_pivot_reuses_its_branch_and_anchor() {
    let (system, _tmp) = create_test_system();
    let user = "reinfer-user";
    let now = Utc::now();
    let prior = mem(
        "Working on the Vamana index integration for vector search.",
        now - Duration::days(1),
    );
    let pivot = mem(
        "We need to abandon this and pivot to a different strategy entirely.",
        now,
    );

    for _ in 0..2 {
        system
            .infer_lineage_for_memory(user, &pivot, std::slice::from_ref(&prior))
            .expect("infer lineage");
    }

    let branches = system
        .lineage_graph()
        .list_branches(user)
        .expect("branches");
    assert_eq!(
        branches.iter().filter(|b| b.id != "main").count(),
        1,
        "one branch per pivot: {branches:?}"
    );
    let anchors = system
        .lineage_graph()
        .get_edges_from(user, &pivot.id)
        .expect("edges from")
        .into_iter()
        .filter(|e| e.relation == CausalRelation::BranchedFrom)
        .count();
    assert_eq!(anchors, 1, "one anchor per pivot");
}

#[test]
fn no_pivot_signal_creates_no_branch() {
    let (system, _tmp) = create_test_system();
    let user = "no-pivot-user";
    let now = Utc::now();
    let a = mem(
        "Implemented the SPANN search dimension guard.",
        now - Duration::days(1),
    );
    let b = mem("Added a unit test for the SPANN guard.", now);

    system
        .infer_lineage_for_memory(user, &b, std::slice::from_ref(&a))
        .expect("infer lineage");

    let branches = system
        .lineage_graph()
        .list_branches(user)
        .expect("branches");
    assert!(
        branches.iter().all(|br| br.id == "main"),
        "unexpected non-main branch: {branches:?}"
    );
}
