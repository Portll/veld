//! Predicate types for the W6 query planner.
//!
//! A `Query` is a tri-source predicate set (relational / vector / graph). The planner
//! consumes these, estimates cardinality per predicate, and emits a `PhysicalPlan` that
//! orders the cheapest selector first as the scan and the rest as filters.
//!
//! These are pure data — no I/O, no traits, no scoring. They round-trip through serde
//! so callers can stage them over the wire (HTTP / IPC) in a later phase.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single planner-level query. Combines relational filters, an optional vector
/// scan, and graph reachability predicates against a single tenant (`user_id`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Tenant scope. The executor always passes this to every querier so backends
    /// can enforce per-user isolation.
    pub user_id: String,
    /// Structured relational predicates (column equality, IN-set, range).
    pub relational: Vec<RelationalPredicate>,
    /// Optional vector ANN predicate. At most one — multi-vector planning is W7.
    pub vector: Option<VectorPredicate>,
    /// Graph reachability predicates (linked-to-entity, shares-entity, episode).
    pub graph: Vec<GraphPredicate>,
    /// Final top-N to return after intersection.
    pub limit: usize,
    /// Per-stage candidate caps. Clamps scan estimates so the planner never asks
    /// a backend to materialize more than its budget allows.
    pub stage_caps: StageCaps,
}

/// Relational filters expressed as column-level predicates.
///
/// `UserIdEquals` is broken out separately from `Equals` because the selectivity
/// heuristic treats it specially (it's effectively a tenant scope, not a filter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationalPredicate {
    Equals {
        column: String,
        value: serde_json::Value,
    },
    In {
        column: String,
        values: Vec<serde_json::Value>,
    },
    Range {
        column: String,
        lo: Option<serde_json::Value>,
        hi: Option<serde_json::Value>,
    },
    UserIdEquals(String),
}

/// ANN vector search predicate. `kind` selects which Vamana/HNSW index to hit
/// (e.g. `"vamana-text-primary"`); concrete resolution happens in `VectorQuerier`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPredicate {
    pub kind: String,
    pub query_vector: Vec<f32>,
    pub top_k: usize,
}

/// Graph reachability predicates against the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphPredicate {
    LinkedToEntity { entity_id: Uuid },
    SharesEntity { other_memory_id: String },
    EpisodeContains { episode_id: Uuid },
}

/// Per-stage candidate caps. The planner clamps every estimate against the matching
/// cap so a 10_000-element vector scan can be budget-trimmed to e.g. 1_000.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StageCaps {
    pub vector: usize,
    pub graph: usize,
    pub relational: usize,
}

impl Default for StageCaps {
    fn default() -> Self {
        Self {
            vector: 1000,
            graph: 1000,
            relational: 10_000,
        }
    }
}
