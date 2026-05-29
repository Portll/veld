//! Logical + physical plan types for the W6 query planner.
//!
//! The planner emits a `PhysicalPlan` which is a strictly ordered sequence of
//! `PhysicalOp`s. The first op is always a `Scan*` (materializes candidate ids);
//! subsequent ops are `Filter*` (probe per-id); the terminal op is `Limit`.
//!
//! `est_card` is `usize` rather than `u64` because every consumer (stage caps,
//! `Vec::len`, `.min(limit)`) is already `usize`; staying in one integer type
//! avoids `as`-casts at every comparison site.

use super::predicate::{GraphPredicate, RelationalPredicate, VectorPredicate};

/// A fully-ordered physical plan. Ops execute left-to-right.
#[derive(Debug, Clone)]
pub struct PhysicalPlan {
    pub ordered_ops: Vec<PhysicalOp>,
    /// Human-readable plan string for `EXPLAIN`-style debugging and tests.
    pub explain: String,
}

/// One step in a `PhysicalPlan`.
///
/// Scans produce a candidate set (capped at `est_card`). Filters probe one id at
/// a time. `Limit` truncates the final result.
#[derive(Debug, Clone)]
pub enum PhysicalOp {
    ScanRelational {
        predicate: RelationalPredicate,
        est_card: usize,
    },
    ScanVector {
        predicate: VectorPredicate,
        est_card: usize,
    },
    ScanGraph {
        predicate: GraphPredicate,
        est_card: usize,
    },
    FilterRelational {
        predicate: RelationalPredicate,
    },
    FilterVector {
        predicate: VectorPredicate,
    },
    FilterGraph {
        predicate: GraphPredicate,
    },
    Limit(usize),
}
